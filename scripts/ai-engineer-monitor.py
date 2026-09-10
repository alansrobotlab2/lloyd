#!/usr/bin/env python3
"""AI Engineer YouTube Monitor — compatibility shim.

The pipeline moved to `youtube_channel_monitor.py` when a second channel
(Discover AI) was added, and digests now run through the `youtube-digest`
worker source; autonomy task #75 was deleted on 2026-09-09 once that had
taken over. The `ai-engineer-monitor` skill still invokes this path with
`--process-one` as the operator fallback, and the vault is not part of this
repository, so the name stays and simply runs the shared script for the AI
Engineer channel. Any flag it accepts works here too.
"""
import os
import runpy
import sys

_TARGET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "youtube_channel_monitor.py")

if __name__ == "__main__":
    sys.argv = [_TARGET, "--channel", "ai-engineer", *sys.argv[1:]]
    runpy.run_path(_TARGET, run_name="__main__")
