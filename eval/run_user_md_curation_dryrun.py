#!/usr/bin/env python3
"""Dry run of rationale-per-line curation for loaded memory (#1488).

The proposal (`scripts/maintenance/vault-user-md-rationale-curation.patch`):
every line of `lloyd/USER.md` and every `lloyd/MEMORY.md` index line has a
ledger row — why it is loaded (what future behaviour it changes), where it
came from, and the condition that would make it false — kept in a sidecar the
prompt never loads. A nightly curator reads the rows, checks each
`retire_when` against disk, and archives the lines whose reason no longer
holds, so the file gains headroom by selection rather than by refusal.

This script is the curator's first pass, run once against COPIES: it hands the
primary a numbered copy of the file and asks for one ledger row per entry plus
a keep / retire / relocate verdict. It writes nothing in the vault. What it
measures: how many rows a whole-file backfill produces and what they cost,
how many bytes and tokens the proposed retirements would free, and — after a
human checks each proposed retirement against disk — how often the curator is
right to retire (`--audit <labels>`).

    flock -s ~/.local/state/lloyd-automod/primary.lock \\
      .venvs/lloyd/bin/python eval/run_user_md_curation_dryrun.py run \\
        --file ~/lloyd-data/eval/1488/USER.md --out ~/lloyd-data/eval/1488/user.json
    .venvs/lloyd/bin/python eval/run_user_md_curation_dryrun.py score \\
        --file ~/lloyd-data/eval/1488/USER.md --rows ~/lloyd-data/eval/1488/user.json \\
        --audit ~/lloyd-data/eval/1488/user_audit.txt
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

LIVE_MEMORY = Path.home() / "obsidian" / "lloyd"
VERDICTS = ("keep", "retire", "relocate")

PROMPT = """You are the nightly curator of a file that is loaded into EVERY system prompt of a
personal AI agent ("Lloyd"). Its size ceiling is fixed; room for a new entry has to be
made by retiring entries whose reason for being loaded no longer holds.

Below is the file with each entry numbered. For EVERY numbered entry write one ledger row:
- "n": the entry number
- "rationale": <=120 chars. What future behaviour of the agent this line changes. If it
  changes none (a changelog, a status snapshot, a dated count), say so.
- "origin": the source document, item number or date the line itself names, else "unnamed"
- "retire_when": a concrete, checkable condition under which this line becomes false or
  useless (a path that stops existing, an item that closes, a setting that changes)
- "verdict": "keep" | "relocate" | "retire"
    keep     — it changes behaviour and its reason still holds
    relocate — it is true and useful but is detail/history, not a rule; it belongs in a
               topic note that is read on demand, not in every prompt
    retire   — its reason no longer holds, or it duplicates another entry
- "reason": <=160 chars, why this verdict; for retire/relocate name what you checked or
  what makes it stale
- "check": for retire, the one shell command or file read that would confirm it

Rules: archive, never delete — a retired line goes to an archive note. Be conservative:
a rule Alan stated about how he wants to be treated is "keep" unless it is contradicted
by a later entry. Output ONLY a JSON array of rows, one per numbered entry, no prose.

FILE ({name}):
{numbered}
"""


def entries(text: str) -> list[dict]:
    """The file's curatable entries: bullet lines and blockquote paragraphs.

    Headings and front matter are structure, not entries."""
    out = []
    in_fm = False
    for i, line in enumerate(text.splitlines(), 1):
        if i == 1 and line.strip() == "---":
            in_fm = True
            continue
        if in_fm:
            if line.strip() == "---":
                in_fm = False
            continue
        s = line.strip()
        if s.startswith("- ") or (s.startswith(">") and len(s) > 2):
            out.append({"n": len(out) + 1, "line_no": i, "text": line,
                        "bytes": len((line + "\n").encode())})
    return out


def numbered(ents: list[dict]) -> str:
    return "\n".join(f"[{e['n']}] {e['text'].strip()}" for e in ents)


def _primary():
    import httpx
    import yaml
    cfg = yaml.safe_load((HERE.parent / "config.yaml").read_text()) or {}
    base = ((cfg.get("models") or {}).get("primary") or {}).get("base_url") or "http://127.0.0.1:8096"
    model = httpx.get(f"{base}/v1/models", timeout=10).json()["data"][0]["id"]
    return base.rstrip("/"), model


def tokens(text: str) -> int | None:
    try:
        import httpx
        base, model = _primary()
        r = httpx.post(f"{base}/tokenize", json={"model": model, "prompt": text}, timeout=30)
        return int(r.json()["count"])
    except Exception:  # noqa: BLE001
        return None


def _parse_rows(text: str) -> list[dict]:
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        raise ValueError("no JSON array in the curator's answer")
    rows = json.loads(m.group(0))
    return [r for r in rows if isinstance(r, dict) and "n" in r]


def cmd_run(args) -> int:
    import httpx
    path = Path(args.file).expanduser()
    if path.resolve().parent == LIVE_MEMORY.resolve():
        raise SystemExit("refusing the live memory file: copy it first")
    text = path.read_text(encoding="utf-8")
    ents = entries(text)
    base, model = _primary()
    body = {"model": model, "temperature": 0.0, "max_tokens": args.max_tokens,
            "messages": [{"role": "user", "content": PROMPT.format(name=path.name, numbered=numbered(ents))}],
            "chat_template_kwargs": {"enable_thinking": bool(args.thinking)}}
    r = httpx.post(f"{base}/v1/chat/completions", json=body, timeout=1800)
    r.raise_for_status()
    choice = r.json()["choices"][0]
    msg = choice["message"].get("content") or ""
    raw = Path(args.out).expanduser().with_suffix(".raw.json")
    raw.write_text(json.dumps(r.json(), ensure_ascii=False))
    if not msg.strip():
        raise SystemExit(f"empty answer (finish_reason={choice.get('finish_reason')}); "
                         f"raw response in {raw}")
    rows = _parse_rows(msg)
    usage = r.json().get("usage", {})
    Path(args.out).expanduser().write_text(json.dumps(
        {"file": str(path), "entries": ents, "rows": rows, "usage": usage,
         "thinking": bool(args.thinking)}, indent=1, ensure_ascii=False))
    print(json.dumps({"entries": len(ents), "rows": len(rows), "usage": usage}))
    return 0


def cmd_score(args) -> int:
    from eval.stats import wilson_ci
    data = json.loads(Path(args.rows).expanduser().read_text())
    text = Path(args.file).expanduser().read_text(encoding="utf-8")
    ents = {e["n"]: e for e in data["entries"]}
    rows = {int(r["n"]): r for r in data["rows"] if int(r["n"]) in ents}
    by = {v: [n for n, r in rows.items() if r.get("verdict") == v] for v in VERDICTS}
    total = len(text.encode())
    print(f"{Path(args.file).name}: {total} B, {len(ents)} entries, {len(rows)} ledger rows "
          f"({len(ents) - len(rows)} missing)")
    ledger = "\n".join(json.dumps({k: rows[n].get(k) for k in ("rationale", "origin", "retire_when")},
                                  ensure_ascii=False) for n in rows)
    print(f"ledger sidecar size if every row were kept: {len(ledger.encode())} B (never loaded)")
    for v in VERDICTS:
        b = sum(ents[n]["bytes"] for n in by[v])
        print(f"  {v:9s} {len(by[v]):3d} entries, {b:6d} B")
    freed = [n for n in by["retire"] + by["relocate"]]
    kept_text = "\n".join(l for i, l in enumerate(text.splitlines(), 1)
                          if i not in {ents[n]["line_no"] for n in freed}) + "\n"
    t0, t1 = tokens(text), tokens(kept_text)
    print(f"if applied: {total} -> {len(kept_text.encode())} B"
          + (f"; tokens {t0} -> {t1}" if t0 and t1 else ""))
    if args.audit:
        labels = {}
        for line in Path(args.audit).expanduser().read_text().splitlines():
            parts = line.split(None, 2)
            if len(parts) >= 2 and parts[0].isdigit():
                labels[int(parts[0])] = parts[1].lower()
        for v in ("retire", "relocate"):
            judged = [n for n in by[v] if n in labels]
            ok = sum(labels[n] == "ok" for n in judged)
            if judged:
                lo, hi = wilson_ci(ok, len(judged))
                print(f"  audit {v}: {ok}/{len(judged)} right [{lo:.3f}, {hi:.3f}]")
            for n in judged:
                if labels[n] != "ok":
                    print(f"     WRONG {v} [{n}] {ents[n]['text'].strip()[:110]}")
        missed = [n for n, lab in labels.items() if lab == "miss" and n in by["keep"]]
        print(f"  audit keep-but-stale (missed): {len(missed)} {missed}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--file", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--max-tokens", type=int, default=24000)
    r.add_argument("--thinking", action="store_true")
    s = sub.add_parser("score")
    s.add_argument("--file", required=True)
    s.add_argument("--rows", required=True)
    s.add_argument("--audit")
    args = ap.parse_args(argv)
    return {"run": cmd_run, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
