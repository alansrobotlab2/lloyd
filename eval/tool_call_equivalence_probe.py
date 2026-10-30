#!/usr/bin/env python3
"""Ask the engine to decide, not only to write: the canary fold's tool-call leg.

#3188, owed from #2843. `eval/action_equivalence.py` landed 2026-10-16 in `e7e402d5` to
grade what an engine *did* (which tool, which arguments) into four tiers, and for a week
nothing called it outside its own test. The fold both arm routes run —
`agent-services/bin/flash-next-canary-step.sh`, reached from `flash-next-run-arm.sh:144`
and `eval/kv_dtype_arm.py:124` — drove `eval/engine_output_probe.py`, which posts
`{model, prompt, temperature: 0, logprobs: 1}` to `/v1/completions` (:330-331, and the load
path again at :659-660). `/v1/completions` has no `tools` field, so the canary has always
measured prose tokens and never a decision: the bf16 arm's only verdict in
`~/lloyd-data/logs/services/flash-next-canary.jsonl` (2026-10-12T23:16:34+00:00, `past-floor`)
predates the comparator, and the quant/KV-dtype decision question is still unanswered.

What the comparator is, exactly, because this module's whole job is to feed it correctly:
`classify(a, b)` returns ONE tier string from `TIERS` (`action_equivalence.py:102-103`, in
decision precedence order `protocol_invalid, different_tool, args_differ, canonically_equal`),
`explain(a, b)` returns `{"tier", "why"}` (:395-399), and `coerce` (:184-197) accepts ONE
call — a `ToolCall`, a wire dict `{"function": {name, arguments}}`, a bare `{name, arguments}`
dict or a `(name, args)` pair. It is a per-call comparator: it has no tier for "the engine
declined to call at all", so a decline is handed to it as a nameless call and lands in
`protocol_invalid`, whose `protocol_reason` reading is "call has no tool name" (:211-224) —
which is what a decline is, and is reported separately by `capture` as `no_tool_call` too.

This module is a separate file because the existing probe cannot be extended to do it: its
`probe_one` (:328) posts a raw `prompt` string and its `parse_completion` (:294-315) reads
the legacy `logprobs.tokens` shape, both of which a tool call has no part in. A tool call
arrives on `/v1/chat/completions`, in `choices[0].message.tool_calls[].function`, as a
**name and a raw argument STRING** — which is exactly the shape `action_equivalence.py`'s
header says it was written for and refuses to repair ("arguments arrive as raw strings, not
parsed objects, because that is the shape the log holds and the shape a tool boundary
passes: repairing a malformed call here would hide the finding").

Two subcommands, one per leg the fold needs:

* `capture --label ARM` replays every prompt of `eval/tool_call_corpus.yaml` as a
  chat completion at temperature 0 with that prompt's own `tools` array, and **appends** one
  row per prompt to the arm's capture file. Nothing is compared and nothing is judged: it is
  a recording, written by the arm that made the calls.
* `report --label ARM` reads the capture files of every arm that recorded inside the SAME
  window, forms same-prompt pairs of TWO kinds (#3597) — one per pair of arms, which is the
  dtype answer, and one per pair of consecutive captures of the SAME arm, which is the floor
  a dtype answer has to clear — runs `action_equivalence.classify` over each, and appends one
  line of per-tier counts naming every `args_differ` and `different_tool` pair by prompt id
  and tool name, followed by the same-arm tiers and the number of prompts they were measured
  over. A single arm in the window is not a comparison, so it says so instead of inventing a
  verdict.

The corpus is four prompts, each carrying the `tools` array that call needs — two `Read`
signatures, one `Bash`, two key-value lookups, one `Bash`-only decliner — and the
`expect_tool_call` flag that says which answers are the expected ones. Without that flag the
instrument cannot tell a finding from a correct refusal: the fourth prompt asks for the word
`idle`, so prose there is right and a tool call there is the finding (`call_when_none_asked`),
while a prose answer to one of the first three is `no_tool_call_when_asked`. The file lives at
`eval/tool_call_corpus.yaml` — YAML because `.gitignore:42` ignores `*.json`
tree-wide, so a committed JSON corpus would be invisible to the `git add` that archives it —
never inside `eval/engine_output/corpus.yaml`: that file
is the prose canary's measuring stick, it is fed to `/v1/completions`, and clause 1 requires
it and the floor and reference files stay byte-unchanged.

Exit codes are the prose probe's, unchanged, because the fold's non-fatal contract depends
on the pair being identical (#3109's rule: no arm is lost to its instrument):

    0  captured everything / every pair sits in `canonically_equal`
    1  a finding ABOUT THE ENGINE: a prompt that was asked for a call and got prose, a call
       whose raw arguments are not a JSON object, or a pair that came back
       `different_tool` / `args_differ`
    2  the INSTRUMENT could not decide: the engine unreachable, a completion with no
       `choices` to read, the corpus unreadable, or fewer than two arms in one window

Capture ruling, and why it does not touch the no-blob rule
----------------------------------------------------------
Captures go to `$LLOYD_DATA/logs/services/` — the canary log's own directory, so the operator
watching a window sees prose agreement and decision agreement side by side — and they are
**keyed by arm label and prompt id and hold raw argument strings only**: no prompt text, no
response prose, no request body. `app/component_manifest.py:40` is the rule that says
"**No blob store and no `rebuild_request`.**", and this stays on the right side of it: a
capture row records *what the engine decided*, which is the finding itself and cannot be
reconstructed from anything else, and it deliberately omits *what was asked*, which anyone
can re-read from the corpus the row names by sha. `prompt_sha256` and `corpus_sha256` are
what make a pairing honest; the prompt text is not needed for that and is therefore not
stored. The files are bounded by the retention sweep (`scripts/groundskeeper/retention-sweep.py`
store 20), which keeps 14 days of them, so per-arm captures age rather than accumulate.

Pairing is only ever **inside one window** (`CANARY_WINDOW`), and a report never
reconstructs a comparison from stored captures of two different windows. #2683 wrote that
law for the prose side (`engine_output_probe.py:481-489`: a cross-corpus compare "would
silently compare a bf16 arm's answer against a new-kv one's and report agreement of 1.0 —
which is what a comparison of two different questions always looks like") and a sha-less
pair sat in the canary log for four days before it landed. The window is the same guard for
the same reason: the two arms must have been asked the same question in the same boot, and
outside a paired window no claim can be made later — a `within-floor` line can be rebuilt
from stored bytes, a decision cannot.

Both are append-only files the fold never reads back as a gate: `capture` writes rows,
`report` appends one `tool_equivalence` row, and the fold still always exits 0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
LLOYD_HOME = HERE.parent
if str(LLOYD_HOME) not in sys.path:
    sys.path.insert(0, str(LLOYD_HOME))

from eval.action_equivalence import (  # noqa: E402  # the comparator this module exists to call
    ARGS_DIFFER,
    DIFFERENT_TOOL,
    TIERS,
    explain,
)

DEFAULT_BASE_URL = "http://127.0.0.1:8096"  # engine_output_probe.py:98, the one engine
DEFAULT_CORPUS = HERE / "tool_call_corpus.yaml"
CAPTURE_PREFIX = "flash-next-canary-toolcalls-"
KIND_CALL = "tool_call"
KIND_REPORT = "tool_equivalence"
SCHEMA = 1
REQUEST_TIMEOUT_S = 120.0
REACHABILITY_TIMEOUT_S = 3.0

#: What a pair compares. `cross_arm` is the dtype question the window exists to answer.
#: `same_arm` is the SAME arm asked the same prompt twice inside one window, which is the
#: only measurement of how far this engine disagrees with itself at `temperature: 0` — the
#: floor a cross-arm split has to clear before it counts as a finding (#3597). The prose leg
#: has read against a floor like this since #1743; this leg never had one, and the 2026-10-30
#: window is what comparing two arms without one produces: `protocol_invalid=4` with every
#: other tier 0, a line that cannot distinguish a dtype regression from the budget.
CROSS_ARM = "cross_arm"
SAME_ARM = "same_arm"

#: What the report line says when the window holds no second capture of any arm. Not the
#: same thing as a floor of zero: an unmeasured floor absorbs NOTHING (clause 4 of #3597),
#: so a split is never written off against a measurement nobody took.
FLOOR_UNMEASURED = "unmeasured"

#: Flags that make a capture a finding ABOUT THE ENGINE, and the ones that do not.
#: `no_tool_call` is on this list's other side on purpose: it records that an arm declined,
#: which is the correct answer for the corpus's negative control and a finding only for a
#: prompt that asked (`no_tool_call_when_asked`). Counting the bare fact instead is how an
#: instrument ends up reporting a finding on every single run and gets ignored.
FINDING_FLAGS = frozenset({
    "no_tool_call_when_asked",
    "call_when_none_asked",
    "unnamed_call",
    "arguments_not_a_string",
    "arguments_not_json",
    "arguments_not_a_json_object",
})

EXIT_OK = 0
EXIT_FINDING = 1        # a finding ABOUT THE ENGINE — same meaning as the prose probe's 1
EXIT_INSTRUMENT = 2     # the instrument could not decide — same meaning as the prose 2


class ProbeRefused(Exception):
    """The instrument could not decide. Exit 2, and the fold keeps walking.

    `eval/engine_output_probe.py:163-165` makes this distinction on the prose side and the
    fold depends on both probes keeping it: a probe whose socket times out is a probe that
    did not run, and a probe that did not run has no verdict to report about the engine.
    """


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


# ----------------------------------------------------------------------------- paths
def services_dir() -> Path:
    """`$LLOYD_DATA/logs/services`, the canary log's own directory.

    Resolved per call from the environment rather than from `app.paths.DATA_ROOT`, to be the
    SAME expression the fold uses (`${LLOYD_DATA:-$HOME/lloyd-data}/logs/services`): a
    capture written somewhere the operator is not reading is a capture nobody reads, and
    `app.paths` additionally insists on the root marker. A test points `LLOYD_DATA` at a
    tmp dir and the probe follows it, which is how every node here avoids the real store.
    """
    root = os.environ.get("LLOYD_DATA") or (Path.home() / "lloyd-data")
    return Path(root) / "logs" / "services"


def _label_slug(arm_label: str) -> str:
    """An arm label is operator text (`kv-bfloat16`, `new-a98247ab-refloored`); it becomes a
    filename, so anything a path would read as structure is flattened, the way the prose
    probe flattens `--label` into a filename at `record_path` (:287-289)."""
    return "".join(c if (c.isalnum() or c in "-_.") else "-" for c in arm_label.strip()) or "arm"


def capture_path(arm_label: str) -> Path:
    return services_dir() / f"{CAPTURE_PREFIX}{_label_slug(arm_label)}.jsonl"


def window_id() -> str:
    """The window these calls belong to, or "" when nobody opened one.

    `flash-next-run-arm.sh:30` stamps `CANARY_WINDOW` once per sweep and passes it to every
    step, so in the sweep route every arm of one window shares it; the fold stamps its own
    when run standalone, which then makes each arm its own window and a report of "one arm
    here" — the honest answer, and the one that used to be a silent false agreement.
    """
    return (os.environ.get("CANARY_WINDOW") or "").strip()


def append_rows(path: Path, rows: list[dict]) -> None:
    """One file per arm, append-only — the prose probe's `out.write_text` overwrites a
    record and this must not: an arm's captures are read beside each other across a window."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_rows(path: Path) -> list[dict]:
    """Every JSON line of a capture, unparseable and blank lines skipped: a half-written
    line is a crash's residue, and it must not cost the rest of the file."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


# ----------------------------------------------------------------------------- corpus
def _load_yaml(raw: str) -> dict:
    """`yaml.safe_load`, imported the way `engine_output_probe.py:165-170` imports it: a missing
    PyYAML is a refused run with the same words, never a half-loaded corpus."""
    try:
        import yaml
    except ImportError as exc:
        raise ProbeRefused(f"PyYAML is required to read the corpus ({exc})") from exc
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ProbeRefused(f"corpus is not parseable YAML: {exc}") from exc
    return data if isinstance(data, dict) else {}


def corpus_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_corpus(path: Path = DEFAULT_CORPUS) -> dict:
    """Read and validate the tool-requesting corpus. Raises `ProbeRefused` — the instrument
    cannot decide, it does not accuse the engine — on anything unreadable or malformed."""
    try:
        data = _load_yaml(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ProbeRefused(f"corpus {path} is not on disk") from None
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        raise ProbeRefused(f"corpus {path} is unreadable: {exc}") from None
    if not isinstance(data, dict) or not isinstance(data.get("prompts"), list) \
            or not data["prompts"]:
        raise ProbeRefused(f"corpus {path} has no `prompts` list to replay")
    seen: set[str] = set()
    for spec in data["prompts"]:
        if not isinstance(spec, dict):
            raise ProbeRefused(f"corpus {path} holds a prompt that is not an object")
        pid = str(spec.get("id") or "")
        text = str(spec.get("text") or "")
        tools = spec.get("tools")
        if not pid or not text:
            raise ProbeRefused(f"corpus {path} holds a prompt with no id or no text")
        if pid in seen:
            raise ProbeRefused(f"corpus {path} names prompt id {pid!r} twice")
        if not isinstance(tools, list) or not tools:
            raise ProbeRefused(f"prompt {pid} carries no `tools` array to offer")
        if not isinstance(spec.get("expect_tool_call"), bool):
            # Required, not defaulted: a corpus that does not say which answers are expected
            # cannot separate a finding from a correct refusal, and defaulting to True would
            # silently turn its own negative control into a standing finding.
            raise ProbeRefused(f"prompt {pid} carries no boolean `expect_tool_call`")
        seen.add(pid)
    data["_path"] = str(path)
    data["_sha256"] = corpus_sha256(path)
    data.setdefault("params", {})
    return data


# ----------------------------------------------------------------------------- engine
class EngineClient:
    """The smallest chat-completions client that can distinguish an unreachable engine from
    an empty answer — the same line `engine_output_probe.py:116-145` draws for `/metrics`.

    Takes the same `post_json` / `get_text` shape as the prose probe's client, so the fold
    and the tests hand both probes one stub and clause 4's claim (the same two records,
    within floor on prose and `args_differ` on the decision) is measured, not asserted.
    """

    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: float = REQUEST_TIMEOUT_S):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def post_json(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            raise ProbeRefused(f"POST {path} -> HTTP {exc.code}: {detail}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ProbeRefused(f"POST {path} unreachable: {exc}") from None
        except json.JSONDecodeError as exc:
            raise ProbeRefused(f"POST {path} returned a body that is not JSON: {exc}") from None

    def get_text(self, path: str) -> str:
        try:
            with urllib.request.urlopen(self.base_url + path, timeout=REACHABILITY_TIMEOUT_S) as r:
                return r.read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001 — reachability only; any failure reads as absent
            return ""


def resolve_model(client: Any, model: str) -> str:
    """`auto` is the model the engine names back on `/v1/models`; any other name passes through.

    `auto` used to be sent to the engine verbatim, and vLLM answers an unknown model name
    with a 404, so every capture the canary step ran with the default refused before its
    first prompt — the bf16 arm of 2026-10-30 (#3188) cost two engine boots and recorded no
    decisions. An engine that names no model is the instrument failing, not a guess."""
    if model != "auto":
        return model
    try:
        data = json.loads(client.get_text("/v1/models") or "{}").get("data") or []
    except (json.JSONDecodeError, AttributeError):
        data = []
    name = data[0].get("id") if data and isinstance(data[0], dict) else None
    if not name:
        raise ProbeRefused("--model auto: GET /v1/models named no served model; pass --model")
    return str(name)


def chat_call(client: Any, spec: dict, *, arm: str, window: str, corpus_sha: str,
              model: str, params: dict) -> dict:
    """Ask one prompt with its own `tools` array, and record the ANSWER, not the question.

    Raw argument strings are kept raw. Repairing them here would be the exact thing
    `action_equivalence.py:10-13` forbids, and a repair would also hide the finding that a
    quantised engine emits an argument string that is not a JSON object at all.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": spec["text"]}],
        "tools": spec["tools"],
        "tool_choice": params.get("tool_choice", "auto"),
        "temperature": 0,                    # clause 1: temperature 0, like the prose probe
        "max_tokens": int(params.get("max_tokens", 64)),
    }
    started = time.time()
    resp = client.post_json("/v1/chat/completions", payload)
    choices = resp.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        # No choices is the instrument failing, not the engine declining: there is nothing
        # here to have decided anything about. The prose probe refuses the same way (#2169).
        raise ProbeRefused(f"prompt {spec['id']}: the completion carried no choices to read")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ProbeRefused(f"prompt {spec['id']}: the choice carried no message object")
    calls = message.get("tool_calls") or []
    recorded, flags = [], []
    for call in calls:
        fn = (call or {}).get("function") or {}
        name = str(fn.get("name") or "")
        raw = fn.get("arguments")
        recorded.append({"name": name, "arguments": raw if isinstance(raw, str) else raw})
        if not name:
            flags.append("unnamed_call")
        if not isinstance(raw, str):
            flags.append("arguments_not_a_string")
        else:
            try:
                parsed = json.loads(raw or "null")
            except json.JSONDecodeError:
                flags.append("arguments_not_json")
            else:
                if not isinstance(parsed, dict):
                    flags.append("arguments_not_a_json_object")
    expected = bool(spec.get("expect_tool_call", True))
    if not calls:
        flags.append("no_tool_call")             # the fact
        if expected:
            # Asked for a call, got prose. A finding about THIS arm's engine, and the row
            # above is its evidence.
            flags.append("no_tool_call_when_asked")
    elif not expected:
        # The corpus's negative control: a prompt whose correct answer is the word `idle`.
        # An engine that called a tool there decided wrong, which is a decision finding of
        # exactly the kind this leg exists to catch.
        flags.append("call_when_none_asked")
    return {
        "kind": KIND_CALL,
        "schema": SCHEMA,
        "ts": _utc_stamp(),
        "arm": arm,
        "window": window,
        "prompt": spec["id"],
        "prompt_sha256": _sha(spec["text"]),
        "corpus_sha256": corpus_sha,
        "model": str(resp.get("model") or model),
        "temperature": 0,
        "expect_tool_call": expected,
        "finish_reason": choices[0].get("finish_reason"),
        "content_chars": len(message.get("content") or ""),   # a count, never the prose
        "n_tool_calls": len(calls),
        "tool_calls": recorded,
        "flags": flags,
        "elapsed_s": round(time.time() - started, 3),
    }


def action_of(row: dict) -> dict:
    """The capture row's decision in the one shape `coerce` reads: `{name, arguments}`.

    The FIRST call is the arm's decision, because the corpus is built one-call-per-prompt
    (every prompt a single request; `max_tokens` 256 since #3597, which is enough for one
    whole call where 64 was not): a row with two calls is itself the
    anomaly, so `count_mismatch` names it instead of this module inventing a tier the
    comparator never defined — the tiers at `action_equivalence.py:96-100` are about one call
    against one call.

    A prompt the engine answered in prose has no name and no arguments, and is passed on as
    nameless: `protocol_reason` then reads it "call has no tool name" and the pair lands in
    `protocol_invalid`. That is the only honest mapping available — `coerce` has no decline
    shape, and reading a decline as `different_tool` would name a tool that was never said.
    """
    calls = row.get("tool_calls") if isinstance(row.get("tool_calls"), list) else []
    first = calls[0] if calls and isinstance(calls[0], dict) else {}
    return {"name": first.get("name"), "arguments": first.get("arguments")}


def _tool_name(row: dict) -> str:
    calls = row.get("tool_calls") if isinstance(row.get("tool_calls"), list) else []
    first = calls[0] if calls and isinstance(calls[0], dict) else {}
    return str(first.get("name") or "")


# ----------------------------------------------------------------------------- compare
def call_rows_by_arm(capture_dir: Path | None = None, *, window: str | None = None,
                     exclude: Path | None = None) -> dict[str, list[dict]]:
    """Every arm with call rows in THIS window, read off disk — keyed by arm label.

    Report rows are skipped (they are the fold's own output, not a recording), and a row
    naming another window is skipped, which is the cross-window pairing this leg must never
    make. `exclude` keeps the arm being reported on from being read twice under two names.
    """
    directory = capture_dir or services_dir()
    want_window = window_id() if window is None else window
    by_arm: dict[str, list[dict]] = {}
    for path in sorted(directory.glob(f"{CAPTURE_PREFIX}*.jsonl")):
        if exclude is not None and path.resolve() == exclude.resolve():
            continue
        for row in read_rows(path):
            if row.get("kind") != KIND_CALL:
                continue
            if want_window and str(row.get("window") or "") != want_window:
                continue
            arm = str(row.get("arm") or path.stem.replace(CAPTURE_PREFIX.rstrip("*-"), ""))
            if arm:
                by_arm.setdefault(arm, []).append(row)
    return by_arm


def same_prompt_pairs(rows_by_arm: dict[str, list[dict]]) -> list[dict]:
    """Every pair one window can honestly make, each tagged with the kind of pair it is.

    Two kinds come out, and they answer different questions:

    * `cross_arm` — arm A's answer against arm B's, the dtype question. Per prompt an arm
      contributes its LAST row, the way `compare_records` reads one record per side
      (engine_output_probe.py:571): the arm's latest answer IS its answer.
    * `same_arm` — the SAME arm's consecutive answers to the same prompt, which is the only
      measurement in the window of how far the engine disagrees with itself at
      `temperature: 0` (#3597).

    #3597 clause 2 is why the collapse to `latest` is gone. The dict comprehension that
    built it keyed an arm's rows by prompt id, so a second `capture` under the same
    `--label` — which is exactly what a window does when it re-runs the shipped arm to get a
    floor — OVERWROTE the first row and was never paired with anything: the repeat was
    silently thrown away, in a module whose whole job is to compare recordings. Rows are
    read off disk in file order (`call_rows_by_arm`), so consecutive rows for one prompt are
    consecutive captures, and pairing them that way costs nothing for an arm captured once.

    Both kinds carry the corpus-sha guard: two corpora are two questions (#2683)."""
    by_arm: dict[str, dict[str, list[dict]]] = {
        arm: {} for arm in rows_by_arm}
    for arm, rows in rows_by_arm.items():
        for row in rows:
            by_arm[arm].setdefault(str(row.get("prompt") or ""), []).append(row)

    def _pair(arm_a: str, arm_b: str, pid: str,
              row_a: dict, row_b: dict, kind: str) -> dict | None:
        if str(row_a.get("corpus_sha256") or "") != str(row_b.get("corpus_sha256") or ""):
            return None
        return {"prompt": pid, "arms": [arm_a, arm_b], "kind": kind,
                "corpus_sha256": str(row_a.get("corpus_sha256") or ""),
                "rows": (row_a, row_b)}

    arms = sorted(by_arm)
    pairs: list[dict] = []
    for i, arm_a in enumerate(arms):
        for arm_b in arms[i + 1:]:
            for pid in sorted(set(by_arm[arm_a]) & set(by_arm[arm_b])):
                made = _pair(arm_a, arm_b, pid, by_arm[arm_a][pid][-1],
                             by_arm[arm_b][pid][-1], CROSS_ARM)
                if made is not None:
                    pairs.append(made)
    for arm in arms:
        for pid in sorted(by_arm[arm]):
            seen = by_arm[arm][pid]
            for first, second in zip(seen, seen[1:]):
                made = _pair(arm, arm, pid, first, second, SAME_ARM)
                if made is not None:
                    pairs.append(made)
    return pairs


def _tally(pairs: list[dict]) -> tuple[dict, list[dict], list[dict], list[dict]]:
    """`explain` over a list of pairs: the tier counts and the three naming lists."""
    tiers = {tier: 0 for tier in TIERS}
    args_differ: list[dict] = []
    different_tool: list[dict] = []
    count_mismatch: list[dict] = []
    for pair in pairs:
        row_a, row_b = pair["rows"]
        decided = explain(action_of(row_a), action_of(row_b))
        tiers[decided["tier"]] += 1
        naming = {"prompt": pair["prompt"], "arms": pair["arms"], "why": decided["why"],
                  "tier": decided["tier"]}
        counts = (int(row_a.get("n_tool_calls") or 0), int(row_b.get("n_tool_calls") or 0))
        if counts[0] != counts[1]:
            count_mismatch.append({**naming, "n_tool_calls": list(counts)})
        if decided["tier"] == ARGS_DIFFER:
            args_differ.append({**naming, "tool": _tool_name(row_a) or _tool_name(row_b)})
        elif decided["tier"] == DIFFERENT_TOOL:
            different_tool.append({**naming, "tools": [_tool_name(row_a), _tool_name(row_b)]})
    return tiers, args_differ, different_tool, count_mismatch


def _floor_of(same_arm: list[dict]) -> dict:
    """The shipped arm's own run-to-run disagreement, with the denominator it needs.

    `prompts` is how many prompts the floor was measured over, and it is not decoration: the
    corpus is four prompts, one of them (`tc_no_call_expected`) a nameless decline on both
    arms that `explain` maps to `protocol_invalid` by design, so exactly ONE of those four is
    structural and the decision-bearing tiers divide by the other three. Without the
    denominator a reader takes the floor's `protocol_invalid=1` as a disagreement — or takes
    `args_differ=0` as "this engine is repeatable" when all it says is "not one of the three
    prompts that can disagree did".
    """
    tiers, _, _, _ = _tally(same_arm)
    return {
        "measured": bool(same_arm),
        "arms": sorted({arm for pair in same_arm for arm in pair["arms"]}),
        "prompts": len({pair["prompt"] for pair in same_arm}),
        "pairs": len(same_arm),
        "tiers": tiers,
    }


def split_counts_as_finding(item: dict, floor: dict) -> bool:
    """Does a named cross-arm split count, given what the same arm did to itself?

    The rule is one sentence: a tier the shipped arm ALSO produced against itself is not
    evidence about a dtype, because that is what the engine does anyway. And the
    no-floor case is the loud one — an unmeasured floor absorbs nothing, so with a single
    capture per arm every split still counts exactly as it did before #3597. A report that
    treated "no floor" as "floor of zero" would quietly decide that any disagreement is
    significant one window and none the next, depending on whether somebody re-ran a capture.
    """
    if not floor.get("measured"):
        return True
    return int(floor.get("tiers", {}).get(item.get("tier"), 0)) == 0


def build_report(pairs: list[dict], *, arm: str, window: str) -> dict:
    """Per-tier counts over same-prompt cross-arm pairs, naming every split pair.

    `explain` and not `classify` (`action_equivalence.py:395-399`): the two share `_decide`,
    so the `why` costs nothing and a report that can quote why two argument strings part —
    `at byte 34: '...state.py' != '...worktree.py'` — is the difference between a line an
    operator can act on and a count they have to go and reproduce.

    A pair list may hold both kinds `same_prompt_pairs` emits. The cross-arm tally below is
    the dtype answer and stays exactly the count it was before #3597; the same-arm pairs are
    tallied separately into `floor` and never mixed into `pairs`, because a number that
    averages "the arm disagrees with itself" into "arm A disagrees with arm B" answers neither.
    """
    cross = [p for p in pairs if p.get("kind", CROSS_ARM) == CROSS_ARM]
    same_arm = [p for p in pairs if p.get("kind") == SAME_ARM]
    tiers, args_differ, different_tool, count_mismatch = _tally(cross)
    return {
        "kind": KIND_REPORT,
        "schema": SCHEMA,
        "ts": _utc_stamp(),
        "arm": arm,
        "window": window,
        "pair_arms": sorted({arm for pair in cross for arm in pair["arms"]}),
        "corpus_sha256": (cross or same_arm)[0]["corpus_sha256"] if pairs else "",
        "pairs": len(cross),
        "tiers": tiers,
        "args_differ": args_differ,
        "different_tool": different_tool,
        "count_mismatch": count_mismatch,
        "floor": _floor_of(same_arm),
    }


def render_floor(floor: dict) -> str:
    """The `floor=` segment: the same-arm tiers AND how many prompts they divide by.

    `prompts=N` is not decoration. One of the corpus's four prompts (`tc_no_call_expected`)
    is a nameless decline on every arm and lands in `protocol_invalid` by design, so exactly
    one of the four is structural and the decision-bearing tiers divide by the rest. Without
    the denominator a reader takes the floor's `protocol_invalid=1` as a disagreement, or
    takes `args_differ=0` as proof the engine is repeatable when all it says is that the
    prompts which CAN disagree did not."""
    if not floor.get("measured"):
        return f"floor={FLOOR_UNMEASURED}(no second capture: every split counts)"
    tiers = " ".join(f"{tier}={floor['tiers'][tier]}" for tier in TIERS)
    return (f"floor={'+'.join(floor['arms']) or '-'} prompts={floor['prompts']} "
            f"pairs={floor['pairs']} {tiers}")


def render_report(report: dict) -> str:
    """The one line the fold prints. Counts first, then every split pair named: an operator
    reading a terminal must not have to open the file to learn WHICH prompt split, and a
    per-tier count alone is exactly the reading that made the prose canary's `past-floor`
    row useless for 11 days."""
    counts = " ".join(f"{tier}={report['tiers'][tier]}" for tier in TIERS)
    arms = "+".join(report["pair_arms"])
    # `floor=` sits before ` split=` on purpose: the fold and its tests locate the split list
    # by cutting the line at ` split=`, and moving that anchor to order nothing is a break
    # bought for nothing.
    head = (f"tool-equivalence: window={report['window'] or '-'} arms={arms or '-'} "
            f"pairs={report['pairs']} {counts} {render_floor(report['floor'])}")
    named = [f"{item['prompt']}:{item['tool']}" for item in report["args_differ"]]
    named += [f"{item['prompt']}:{'>'.join(t or 'declined' for t in item['tools'])}"
              for item in report["different_tool"]]
    named += [f"{item['prompt']}:n_tool_calls={'>'.join(str(n) for n in item['n_tool_calls'])}"
              for item in report["count_mismatch"]]
    return f"{head} split=[{', '.join(named)}]" if named else head


# ----------------------------------------------------------------------------- cli
def main(argv: list[str] | None = None, *,
         client_factory: Callable[[str], Any] = lambda url: EngineClient(url)) -> int:
    ap = argparse.ArgumentParser(
        prog="tool_call_equivalence_probe",
        description="Record and compare what the engine DECIDES (tool + raw arguments), "
                    "temperature 0, as the canary fold's tool-call leg.")
    ap.add_argument("command", choices=("capture", "report"))
    ap.add_argument("--label", required=True, help="the arm label the capture is keyed by")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    ap.add_argument("--model", default="auto",
                    help="passed through; 'auto' takes the model the engine names back")
    ap.add_argument("--out", default=None,
                    help="capture file; default $LLOYD_DATA/logs/services/"
                         f"{CAPTURE_PREFIX}<label>.jsonl")
    ap.add_argument("--capture-dir", default=None,
                    help="where report looks for arms' captures; default the services dir")
    args = ap.parse_args(argv)

    arm = _label_slug(args.label)
    window = window_id()
    out_path = Path(args.out) if args.out else capture_path(arm)

    if args.command == "capture":
        try:
            corpus = load_corpus(Path(args.corpus))
        except ProbeRefused as exc:
            print(f"tool-call capture: cannot-decide: {exc}", file=sys.stderr)
            return EXIT_INSTRUMENT           # nothing written: a recording is not a guess
        client = client_factory(args.base_url)
        try:
            model = resolve_model(client, args.model)
        except ProbeRefused as exc:
            print(f"tool-call capture: cannot-decide: {exc}", file=sys.stderr)
            return EXIT_INSTRUMENT
        rows, broken = [], []
        for spec in corpus["prompts"]:
            try:
                rows.append(chat_call(client, spec, arm=arm, window=window,
                                      corpus_sha=corpus["_sha256"], model=model,
                                      params=corpus["params"]))
            except ProbeRefused as exc:
                print(f"tool-call capture: cannot-decide: {spec['id']}: {exc}",
                      file=sys.stderr)
                return EXIT_INSTRUMENT       # partial recording would read as a decision set
            broken += [f"{spec['id']}:{flag}" for flag in rows[-1]["flags"]]
        append_rows(out_path, rows)
        asked = len(rows)
        counts: dict[str, int] = {}
        for item in broken:
            flag = item.split(":", 1)[1]
            counts[flag] = counts.get(flag, 0) + 1
        findings = sum(counts.get(flag, 0) for flag in FINDING_FLAGS)
        print(f"tool-call capture: arm={arm} window={window or '-'} prompts={asked} "
              + " ".join(f"{flag}={counts[flag]}" for flag in sorted(counts))
              + f" findings={findings} out={out_path}")
        # A prompt asked for a call that got prose, or the reverse, or arguments that are not
        # a JSON object, is a finding about the ENGINE (exit 1). The calls are still recorded:
        # that row is the evidence, and `no_tool_call` on a prompt that was never asked to
        # call is a fact, not a finding — which is why the flags are named, not counted.
        return EXIT_FINDING if findings else EXIT_OK

    # report
    try:
        directory = Path(args.capture_dir) if args.capture_dir else out_path.parent
    except OSError as exc:                       # pragma: no cover — defensive on bad paths
        print(f"tool-equivalence: cannot-decide: {exc}", file=sys.stderr)
        return EXIT_INSTRUMENT
    rows_by_arm = call_rows_by_arm(directory, exclude=out_path)
    mine = [r for r in read_rows(out_path) if r.get("kind") == KIND_CALL]
    if mine:
        rows_by_arm.setdefault(arm, []).extend(
            [r for r in mine if not window or str(r.get("window") or "") == window])
    if len(rows_by_arm) < 2:
        print(f"tool-equivalence: cannot-decide: {len(rows_by_arm)} arm(s) "
              f"{sorted(rows_by_arm) or '[]'} have rows in window {window or '-'}; "
              "a comparison needs two arms inside one window")
        return EXIT_INSTRUMENT                   # nothing appended: no verdict, no row
    pairs = same_prompt_pairs(rows_by_arm)
    if not pairs:
        print(f"tool-equivalence: cannot-decide: arms {sorted(rows_by_arm)} share no prompt "
              f"id inside window {window or '-'} (or every pair spanned two corpora)")
        return EXIT_INSTRUMENT
    report = build_report(pairs, arm=arm, window=window)
    line = render_report(report)
    append_rows(out_path, [report])
    print(line)
    # A split counts only past the floor (#3597 clause 4). The fold reads this exit code as
    # "engine regression, do not promote" (`flash-next-canary-step.sh:222-231`), so a tier the
    # shipped arm ALSO produced against itself must stop being a veto — while an unmeasured
    # floor, which is what every window until now has had, keeps counting every split exactly
    # as it did before this change.
    split = [item for item in report["args_differ"] + report["different_tool"]
             if split_counts_as_finding(item, report["floor"])]
    return EXIT_FINDING if split else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
