"""Measure the backlog list route the way the MC page hits it.

Read-only: nothing here writes to the vault. Prints, per call, the latency and
the byte size, plus a count of frontmatter parses that actually happened — so
"one pass cold, none warm" (item #1199 cause 1) is read off the real corpus
rather than asserted from a unit test.

    python -m scripts.maintenance.backlog_route_probe
"""

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.routers import backlog as BR  # noqa: E402

PARSES: list[str] = []
_REAL_PARSE = BR._backlog_parse_fm


def _counting_parse(*args, **kwargs):
    PARSES.append("x")
    return _REAL_PARSE(*args, **kwargs)


BR._backlog_parse_fm = _counting_parse


def call(label, fn, unit="ms"):
    PARSES.clear()
    t0 = time.perf_counter()
    resp = fn()
    elapsed = (time.perf_counter() - t0) * 1000
    body = bytes(resp.body)
    rows = json.loads(body)
    n = len(rows) if isinstance(rows, list) else 1
    print(f"{label:<34} {elapsed:8.0f} {unit}   {n:>5} rows   "
          f"{len(body):>10,} B   {len(PARSES):>4} parses")
    return rows


def main() -> None:
    print(f"corpus: {BR._BACKLOG_DIR}")
    print(f"files : {sum(1 for f in BR._BACKLOG_DIR.glob('*.md') if BR._BACKLOG_PATTERN.match(f.name))}, "
          f"{sum(f.stat().st_size for f in BR._BACKLOG_DIR.glob('*.md')):,} B on disk\n")

    BR._FM_CACHE.clear()
    call("tasks  cold (cleared cache)", lambda: BR.backlog_tasks())
    call("tasks  warm", lambda: BR.backlog_tasks())
    call("tasks  warm ?board_id=<smallest>",
         lambda: BR.backlog_tasks(board_id=str(min(
             json.loads(bytes(BR.backlog_boards().body)),
             key=lambda b: b["tasks_count"])["id"])))
    call("tasks  warm ?q=event-loop", lambda: BR.backlog_tasks(q="event-loop"))
    call("boards warm", lambda: BR.backlog_boards())

    target = sorted(BR._BACKLOG_DIR.glob("1199-*.md"))
    if not target:
        print("\ndetail: no 1199-*.md on this board, skipped")
        return
    PARSES.clear()
    t0 = time.perf_counter()
    detail = json.loads(bytes(BR.backlog_task_detail(1199).body))
    elapsed = (time.perf_counter() - t0) * 1000
    disk = target[0].read_text(encoding="utf-8").split("---", 2)[2]
    heading = re.search(r"^#\s+.+$", disk, re.M)
    disk_body = disk[heading.end():].strip() if heading else disk.strip()
    same = " ".join(detail["description"].split()) == " ".join(disk_body.split())
    print(f"{'detail #1199':<34} {elapsed:8.1f} ms        1 row   "
          f"{len(detail['description']):>10,} B   {len(PARSES):>4} parses")
    print(f"\ndetail body equals the file's body: {same} "
          f"(file {len(disk_body):,} B, response {len(detail['description']):,} B)")


if __name__ == "__main__":
    main()
