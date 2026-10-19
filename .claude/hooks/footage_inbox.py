#!/usr/bin/env python3
"""Hook (SessionStart + UserPromptSubmit): the takes in footage/ that have no cut yet.

A take is footage/<file>; its cut is edits/<slug>/cut.mp4, slug as the cut skill names it
(IMG_1234.MOV -> img-1234). Prints nothing when every take is cut. Never fails a session.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VIDEO = {".mov", ".mp4", ".m4v"}


def slug(name):
    return re.sub(r"[^a-z0-9]+", "-", Path(name).stem.lower()).strip("-")


def waiting(root=ROOT):
    inbox = root / "footage"
    if not inbox.is_dir():
        return []
    return sorted(f.name for f in inbox.iterdir()
                  if f.suffix.lower() in VIDEO and not (root / "edits" / slug(f.name) / "cut.mp4").exists())


def main():
    try:
        event = json.load(sys.stdin).get("hook_event_name", "UserPromptSubmit")
    except (ValueError, OSError):
        event = "UserPromptSubmit"
    try:
        todo = waiting()
    except OSError:
        return
    if not todo:
        return
    msg = (f"{len(todo)} take(s) in footage/ not cut yet: {', '.join(todo)}. Cut each with the cut skill "
           f"(source footage/<file>, edit dir edits/<slug>) before any styling; never style a raw take.")
    print(json.dumps({"hookSpecificOutput": {"hookEventName": event, "additionalContext": msg}}))


if __name__ == "__main__":
    if sys.argv[1:] == ["test"]:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            r = Path(d)
            (r / "footage").mkdir()
            for n in ("IMG_1234.MOV", "done.mp4", "notes.txt"):
                (r / "footage" / n).write_text("x")
            (r / "edits" / "done").mkdir(parents=True)
            (r / "edits" / "done" / "cut.mp4").write_text("x")
            assert slug("IMG_1234.MOV") == "img-1234"
            assert waiting(r) == ["IMG_1234.MOV"], waiting(r)
            assert waiting(r / "nowhere") == []
        print("ok")
    else:
        main()
