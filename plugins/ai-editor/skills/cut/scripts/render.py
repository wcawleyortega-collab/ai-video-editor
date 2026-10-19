#!/usr/bin/env python3
"""Render the cut with plain ffmpeg: every kept span trimmed and joined.

    python3 render.py <source> <edit_dir> [--crf 18] [--no-hw] [--keep-level]

Reads <edit_dir>/decisions.json and report.json, writes <edit_dir>/cut.mp4 at the
source resolution (rotation applied), H.264 + AAC. HDR sources come out SDR BT.709 and the
speech at -14 LUFS.

How it stays exact on a big 4K source:
  - Each kept span is encoded on its own (video only), in parallel. One
    filter_complex over the whole 4K stream with 100 trims crawls.
  - Frames are counted per segment and again after joining. Frames in == frames out.
  - Audio is cut in ONE pass and muxed on at the end. Separate AAC tracks per
    segment add encoder padding at every join and drift by seconds.
  - Every audio span gets an 8 ms fade in and out, so a join never clicks.
  - Keyframe every 15 frames, so the next step (Remotion) can seek quickly.
Never stream-copy (-c copy) across cuts: that snaps to keyframes, up to 2 s off.
"""
import argparse
import concurrent.futures as cf
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

FADE_S = 0.008
LEAD_S = 0.5     # seek this far early, let the time-based trim pick the frames


def count_frames(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                        "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    return int(r.stdout.strip().split(",")[0] or 0)   # colour-tagged files print "60," here


def starts(path):
    """(video stream start, file start) in seconds. With -copyts the video trim sees the stream's
    own timestamps, which can begin a frame or two in (B-frame delay, an edit list); the audio
    pass is rebased to the file start."""
    def one(entry, *sel):
        r = subprocess.run(["ffprobe", "-v", "error", *sel, "-show_entries", entry, "-of",
                            "default=nw=1:nk=1", str(path)], capture_output=True, text=True)
        try:
            return float(r.stdout.split()[0])
        except (ValueError, IndexError):
            return 0.0
    return one("stream=start_time", "-select_streams", "v:0"), one("format=start_time")


# iPhones film HDR (HLG, Dolby Vision on an HLG base) by default. Squeezed into 8-bit yuv420p
# untouched, the picture comes out grey and washed out: convert to SDR BT.709 on the Mac's
# VideoToolbox (Homebrew's ffmpeg has no zscale) and tag it so.
HDR_TRC = {"arib-std-b67", "smpte2084"}
TONEMAP = ("scale_vt=color_matrix=bt709:color_primaries=bt709:color_transfer=bt709,"
           "hwdownload,format=p010le,format=yuv420p")
SDR_TAGS = ["-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709"]
TARGET_LUFS, PEAK_DB = -14.0, -1.0


def is_hdr(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=color_transfer", "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return r.stdout.strip() in HDR_TRC


def loudness(path, af=None):
    """(integrated LUFS, true peak dB) of path, after the filter af; None when unmeasurable."""
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af",
                        (af + "," if af else "") + "ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True)
    tail = r.stderr[r.stderr.rfind("Summary:"):]
    try:
        return float(tail.split("I:")[1].split()[0]), float(tail.split("Peak:")[1].split()[0])
    except (IndexError, ValueError):
        return None


def level_filter(path):
    """The filter that puts the speech at -14 LUFS (what the apps play): one gain on the whole track,
    plus a lookahead limiter at -1.5 dB only when that gain would push a peak past -1 dB (a plain
    gain alone stops short: one loud plosive caps a quiet take at -24). The limiter takes some level
    back, so it is measured once more and the shortfall added. None: already there, or unmeasurable."""
    m = loudness(path)
    if not m or m[0] < -36 or abs(TARGET_LUFS - m[0]) < 0.5:   # under -36: room tone, not speech
        return None
    gain = min(TARGET_LUFS - m[0], 20.0)
    if m[1] + gain <= PEAK_DB:
        return f"volume={gain:.2f}dB"
    lim = ",alimiter=limit=0.841:level=false"
    again = loudness(path, f"volume={gain:.2f}dB{lim}")
    if again:
        gain = min(gain + TARGET_LUFS - again[0], 20.0)
    return f"volume={gain:.2f}dB{lim}"


def video_window(s, e, fps, vstart):
    """Trim times that select frames s..e-1 of the video stream exactly. Frame k sits at
    vstart + k/fps only up to the container's timestamp rounding (a 1/600 s timescale moves a
    29.97 fps frame by up to 0.8 ms either way), so a trim on the exact frame time keeps or drops
    that frame by the rounding: two edges rounded differently give N-1 or N+1. Half a frame
    early on both edges is immune to any rounding under half a frame."""
    return vstart + (s - 0.5) / fps, vstart + (e - 0.5) / fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("edit_dir")
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--no-hw", action="store_true", help="skip hardware decode on a Mac")
    ap.add_argument("--keep-level", action="store_true", help="leave the take's loudness as recorded")
    a = ap.parse_args()

    d = Path(a.edit_dir)
    rep = json.loads((d / "report.json").read_text())
    fps, frames = rep["fps"], rep["frames"]
    src, out = a.source, d / "cut.mp4"
    hw = sys.platform == "darwin" and not a.no_hw
    vstart, fstart = starts(src)
    off = vstart - fstart     # where frame 0 plays on the audio's timeline
    tmp = Path(tempfile.mkdtemp(prefix="cut_render_"))
    hdr = is_hdr(src) and hw
    print(f"{len(frames)} spans, {sum(e - s for s, e in frames)} frames -> {out}"
          + (" (HDR source: converted to SDR)" if hdr else ""), flush=True)
    if is_hdr(src) and not hw:
        print("  WARN HDR source left as HDR: the SDR conversion needs a Mac's VideoToolbox (no --no-hw)")

    def one(i):
        s, e = frames[i]
        t0, t1 = video_window(s, e, fps, vstart)
        seg = tmp / f"{i:05d}.mp4"
        # -copyts is load-bearing: without it the input seek rebases timestamps
        # to zero and the absolute trim picks the wrong footage at the right length.
        # Frame-numbered timestamps at an explicit -r: after setpts the frames carry no duration,
        # so without -r the muxer gives the last one ~0 s and FFmpeg 7's edit list ends on it,
        # dropping it on read. N/fps keeps the constant-rate clock from duplicating or dropping
        # frames over a 1/600 s source's jittered timestamps.
        cmd = ["ffmpeg", "-v", "error", "-y", "-copyts", *(["-hwaccel", "videotoolbox"] if hw else []),
               *(["-hwaccel_output_format", "videotoolbox_vld"] if hdr else []),
               "-ss", f"{max(0.0, t0 - fstart - LEAD_S):.6f}", "-i", src, "-an",
               "-vf", f"trim=start={max(0.0, t0):.6f}:end={t1:.6f},setpts=N/({fps})/TB"
               + (f",{TONEMAP}" if hdr else ""),
               "-r", str(fps), "-c:v", "libx264", "-crf", str(a.crf), "-preset", "veryfast",
               "-pix_fmt", "yuv420p", *(SDR_TAGS if hdr else []), "-g", "15", str(seg)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            return f"span {i}: {r.stderr[-300:]}"
        got = count_frames(seg)
        return None if got == e - s else f"span {i}: {got} frames, wanted {e - s}"

    workers = max(2, (os.cpu_count() or 8) // (2 if hw else 4))
    with cf.ThreadPoolExecutor(workers) as ex:
        for n, err in enumerate(ex.map(one, range(len(frames))), 1):
            if err:
                sys.exit(f"ERROR {err}  (segments kept in {tmp})")
            if n % 20 == 0:
                print(f"  video {n}/{len(frames)}", flush=True)

    lst = tmp / "list.txt"
    # concat quoting: a ' in the temp path (a user name like O'Brien) is closed, escaped, reopened
    lst.write_text("".join("file '" + str(tmp / f"{i:05d}.mp4").replace("'", "'\\''") + "'\n"
                           for i in range(len(frames))))
    allv = tmp / "all.mp4"
    if subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                       "-c", "copy", str(allv)]).returncode:
        sys.exit("ERROR concat failed")
    want = sum(e - s for s, e in frames)
    if count_frames(allv) != want:
        sys.exit(f"ERROR concat lost frames (kept in {tmp})")

    print("  audio, one pass", flush=True)
    parts = []
    for i, (s, e) in enumerate(frames):
        t0, t1 = off + s / fps, off + e / fps
        fo = max(0.0, t1 - t0 - FADE_S)
        parts.append(f"[0:a:0]atrim=start={t0:.6f}:end={t1:.6f},asetpts=PTS-STARTPTS,"
                     f"afade=t=in:d={FADE_S},afade=t=out:st={fo:.6f}:d={FADE_S}[a{i}]")
    fc = ";".join(parts) + ";" + "".join(f"[a{i}]" for i in range(len(frames))) + \
        f"concat=n={len(frames)}:v=0:a=1[ac]"
    aud = tmp / "a.m4a"
    if subprocess.run(["ffmpeg", "-v", "error", "-y", "-vn", "-i", src,
                       "-filter_complex", fc, "-map", "[ac]",
                       "-c:a", "aac", "-b:a", "192k", str(aud)]).returncode:
        sys.exit("ERROR audio pass failed")
    lf = None if a.keep_level else level_filter(aud)
    if lf:
        lev = tmp / "a_level.m4a"
        if subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(aud), "-af", lf,
                           "-c:a", "aac", "-b:a", "192k", str(lev)]).returncode:
            sys.exit("ERROR level pass failed")
        aud = lev
        print(f"  level: {lf} -> {TARGET_LUFS:g} LUFS (--keep-level to skip)", flush=True)

    if subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(allv), "-i", str(aud),
                       "-map", "0:v", "-map", "1:a", "-c", "copy", "-movflags", "+faststart",
                       str(out)]).returncode:
        sys.exit("ERROR mux failed")
    got = count_frames(out)
    print(f"  check: {got} frames of {want}, {want / fps:.2f}s")
    if got != want:
        sys.exit(f"ERROR frame mismatch (kept in {tmp})")
    shutil.rmtree(tmp)
    print(out)


if __name__ == "__main__":
    main()
