"""The `summary` display parameter on every advertised tool.

The model writes one short phrase per tool call saying what the call is
doing, and the transcript renders it beside the tool name. It rides in on
the arguments because that is the only channel a tool call has, which
makes it a display field living in a place where everything else is
functional — and that is where every failure mode here comes from:

  * ``session_inject_context`` already has a **required** top-level
    ``summary`` of its own — it is 1 of the 129 tools the aggregator
    advertises today. Treating that one as ours would delete a real
    argument on the way to MCP, and the injected context would arrive
    with no headline.
  * The ``inputSchema`` handed to the injector is the object held in
    ``MCPPool.discovered``, which is process-shared for the life of the
    pool. Injecting into it in place would make the *second* turn read
    `summary` back as the tool's own parameter, skip injection, and stop
    stripping — handing the aggregator an argument no tool declares.
  * The aggregator validates arguments against each tool's real schema,
    so a leaked `summary` is a dispatch error, not a spare field.
"""

import asyncio
import json

from app.harness import events
from app.harness.hooks import HookRegistry
from app.harness.loop import _assistant_message_for_history, _commit_tool_calls
from app.harness.tool_schema import (
    SUMMARY_ARG,
    add_summary_param,
    build_tool_list,
    pop_summary,
)
from app.harness.tool_search import (
    TOOLSEARCH_OPENAI_TOOL,
    TOOLSEARCH_OPENAI_TOOL_WITH_SUMMARY,
    LoadedToolSet,
)
from app.inner_voice import observer_prompt
from app.inner_voice.guards import tool_call_signature


def _mcp_tool(name: str, properties: dict, required: list[str] | None = None) -> dict:
    return {
        "name": name,
        "description": f"{name} does a thing",
        "inputSchema": {
            "type": "object",
            "properties": properties,
            **({"required": required} if required is not None else {}),
        },
    }


_READ = _mcp_tool(
    "Read", {"file_path": {"type": "string"}}, required=["file_path"],
)
# The real shape from agent_mcp/ambient.py — a required `summary` of its own.
_INJECT_CONTEXT = _mcp_tool(
    "session_inject_context",
    {
        "source": {"type": "string"},
        "summary": {"type": "string", "description": "One-line summary (<120 chars ideal)."},
    },
    required=["source", "summary"],
)


def _accumulated(name: str, args: dict, index: int = 0) -> dict:
    """One tool call as ``_accumulate_tool_call`` leaves it."""
    return {
        index: {
            "id": f"call_{name}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }
    }


# ── injection ──────────────────────────────────────────────────────────


def test_summary_is_added_to_every_tool_and_required():
    tools = build_tool_list([("lloyd-mcp", [_READ])], disallowed=set())
    injected = add_summary_param(tools)

    assert injected == {"Read"}
    params = tools[0]["function"]["parameters"]
    assert params["properties"][SUMMARY_ARG]["type"] == "string"
    # First in both lists: property order is the order the schema is shown
    # to the model and roughly the order it emits arguments, so a caption
    # after Bash's `command` is one written after a 40-line heredoc.
    assert list(params["properties"]) == [SUMMARY_ARG, "file_path"]
    assert params["required"] == [SUMMARY_ARG, "file_path"]
    # The tool's own parameters survive untouched.
    assert params["properties"]["file_path"] == {"type": "string"}


def test_a_tool_with_its_own_summary_parameter_is_left_alone():
    tools = build_tool_list([("lloyd-mcp", [_INJECT_CONTEXT])], disallowed=set())
    injected = add_summary_param(tools)

    assert injected == set(), "session_inject_context owns `summary`; it is not ours to strip"
    props = tools[0]["function"]["parameters"]["properties"]
    assert props[SUMMARY_ARG]["description"].startswith("One-line summary")


def test_injection_does_not_mutate_the_shared_discovery_schema():
    """`inputSchema` is the pool's own dict, reused on every turn.

    Injecting in place would leave `summary` in it, so the next turn would
    read it as a real parameter of the tool and stop stripping the value.
    """
    discovered = [("lloyd-mcp", [_READ])]

    first = add_summary_param(build_tool_list(discovered, disallowed=set()))
    assert SUMMARY_ARG not in _READ["inputSchema"]["properties"]
    assert SUMMARY_ARG not in _READ["inputSchema"].get("required", [])

    second = add_summary_param(build_tool_list(discovered, disallowed=set()))
    assert first == second == {"Read"}


def test_toolsearch_gets_its_own_pre_built_summary_variant():
    plain = TOOLSEARCH_OPENAI_TOOL["function"]["parameters"]["properties"]
    with_summary = TOOLSEARCH_OPENAI_TOOL_WITH_SUMMARY["function"]["parameters"]
    assert SUMMARY_ARG not in plain, "the plain constant must stay uninjected"
    assert SUMMARY_ARG in with_summary["properties"]

    on = LoadedToolSet(catalog=[], enabled=True, summaries=True).visible_tools()
    off = LoadedToolSet(catalog=[], enabled=True, summaries=False).visible_tools()
    assert SUMMARY_ARG in on[-1]["function"]["parameters"]["properties"]
    assert SUMMARY_ARG not in off[-1]["function"]["parameters"]["properties"]


# ── extraction ─────────────────────────────────────────────────────────


def test_summary_is_lifted_off_the_dispatch_args_but_kept_in_history():
    """The two records of a call deliberately disagree.

    `_args_dict` is what reaches MCP, which validates against the tool's
    real schema — the caption must not be there. `arguments` is what gets
    replayed to the engine, and it is the only record of this call the
    model will ever see again — the caption MUST be there. Stripping both
    is what broke this on 20260907_184351_ivec8d: the first call of each
    tool carried a summary and every repeat did not, 5/5 vs 0/31, because
    the model imitated its own summary-less example over the schema.
    """
    acc = _accumulated("Read", {"file_path": "server.py", "summary": "Reading server.py"})
    [tc] = _commit_tool_calls(acc, summary_tools={"Read"})

    assert tc["_summary"] == "Reading server.py"
    assert tc["_args_dict"] == {"file_path": "server.py"}, "must not reach MCP"
    assert json.loads(tc["function"]["arguments"]) == {
        "file_path": "server.py", "summary": "Reading server.py",
    }, "must survive into history, or the next call of this tool copies the omission"


def test_the_history_message_carries_the_caption_the_model_wrote():
    """End of the same rule, one layer up: `_assistant_message_for_history`
    is what actually gets sent back to the engine."""
    acc = _accumulated("Bash", {"command": "ls", "summary": "Listing the repo root"})
    committed = _commit_tool_calls(acc, summary_tools={"Bash"})
    msg = _assistant_message_for_history(text="", tool_calls=committed)

    args = json.loads(msg["tool_calls"][0]["function"]["arguments"])
    assert args == {"command": "ls", "summary": "Listing the repo root"}


def test_summary_survives_untouched_on_a_tool_that_owns_the_parameter():
    args = {"source": "autonomy:task-42", "summary": "nightly sweep finished"}
    acc = _accumulated("session_inject_context", args)
    [tc] = _commit_tool_calls(acc, summary_tools={"Read"})

    assert "_summary" not in tc
    assert tc["_args_dict"] == args
    assert json.loads(tc["function"]["arguments"]) == args


def test_a_non_string_summary_is_dropped_from_display_but_still_not_dispatched():
    """Rendering `[object Object]` beside a tool name is worse than nothing,
    and a value the tool never declared still must not reach MCP."""
    acc = _accumulated("Read", {"file_path": "x.py", "summary": {"text": "nope"}})
    [tc] = _commit_tool_calls(acc, summary_tools={"Read"})

    assert "_summary" not in tc
    assert tc["_args_dict"] == {"file_path": "x.py"}


def test_no_summary_emitted_is_not_an_error():
    acc = _accumulated("Read", {"file_path": "x.py"})
    [tc] = _commit_tool_calls(acc, summary_tools={"Read"})

    assert "_summary" not in tc
    assert tc["_args_dict"] == {"file_path": "x.py"}
    assert events.tool_call(
        call_id=tc["id"], name="Read",
        args_json=tc["function"]["arguments"], args_dict=tc["_args_dict"],
        summary=tc.get("_summary", ""),
    )["summary"] == ""


def test_summaries_off_leaves_arguments_exactly_as_the_model_wrote_them():
    args = {"file_path": "x.py", "summary": "Reading x.py"}
    acc = _accumulated("Read", args)
    [tc] = _commit_tool_calls(acc, summary_tools=set())

    assert "_summary" not in tc
    assert tc["_args_dict"] == args


def test_pop_summary_trims_and_only_accepts_strings():
    args = {"a": 1, SUMMARY_ARG: "  Restarting the backend  "}
    assert pop_summary(args) == "Restarting the backend"
    assert args == {"a": 1}

    args = {"a": 1, SUMMARY_ARG: 7}
    assert pop_summary(args) == ""
    assert args == {"a": 1}

    assert pop_summary({"a": 1}) == ""


def test_the_event_carries_the_summary_for_the_transcript():
    evt = events.tool_call(
        call_id="call_1", name="Bash", args_json='{"command":"ls"}',
        args_dict={"command": "ls"}, summary="Listing the repo root",
    )
    assert evt["summary"] == "Listing the repo root"
    assert SUMMARY_ARG not in evt["args_dict"]


# ── what the Inner Voice observer sees ─────────────────────────────────
#
# The observer's job is judging whether the primary is still on the user's
# request. Three of its inputs touch tool calls, and the caption belongs in
# exactly two of them.


def test_the_observer_sees_captions_not_a_wall_of_names():
    """`Tool calls proposed: ['Bash', 'Bash', 'Bash']` is the least
    informative possible input for a drift judgment."""
    tool_calls = [
        {"function": {"name": "Bash"}, "_summary": "Checking root disk usage"},
        {"function": {"name": "Bash"}, "_summary": "Querying GPU names"},
        {"function": {"name": "Bash"}},
    ]
    summary = observer_prompt.build_assistant_message_summary(
        7, "working on it", tool_calls, "tool_calls",
    )
    assert "Bash — Checking root disk usage" in summary
    assert "Bash — Querying GPU names" in summary
    # An uncaptioned call still renders, just bare — the field is best-effort
    # and a missing one must never blank the row.
    assert "'Bash'" in summary


def test_observer_labels_accept_all_three_shapes_a_caption_arrives_in():
    """Live harness event, session-JSON rebuild, and the raw wire string."""
    labels = observer_prompt._tool_call_labels([
        {"function": {"name": "Bash"}, "_summary": "live event"},
        {"function": {"name": "Read"}, "summary": "rebuilt from session json"},
        {"function": {"name": "Grep",
                      "arguments": '{"summary": "raw wire", "pattern": "x"}'}},
        {"function": {"name": "Glob", "arguments": "{not json"}},
    ])
    assert labels == [
        "Bash — live event",
        "Read — rebuilt from session json",
        "Grep — raw wire",
        "Glob",
    ]


def test_the_pretool_prompt_states_the_primarys_intent():
    captioned = observer_prompt.build_pretool_event_summary(
        "Bash", {"command": "df -h /"}, "Checking root disk usage",
    )
    assert 'saying "Checking root disk usage"' in captioned
    # The arguments are still there — the caption is a claim, they are what runs.
    assert "df -h /" in captioned

    plain = observer_prompt.build_pretool_event_summary("Bash", {"command": "df -h /"})
    assert "saying" not in plain
    assert "about to call `Bash`, with args" in plain


def test_the_repetition_guard_never_sees_the_caption():
    """The guard compares what was RUN, not what the primary said about it.

    `exact` is the full `key=value` rendering for every tool except Bash, so
    a caption inside the args would make two byte-identical calls compare as
    different — and rewording is precisely what a looping model does. This is
    why the caption is popped from `_args_dict` and passed to the hook beside
    it rather than inside it.
    """
    def dispatch_args(caption: str) -> dict:
        acc = _accumulated("Read", {"file_path": "server.py", "summary": caption})
        [tc] = _commit_tool_calls(acc, summary_tools={"Read"})
        return tc["_args_dict"]

    a = tool_call_signature("Read", dispatch_args("Reading server.py"))
    b = tool_call_signature("Read", dispatch_args("Taking another look at the server"))
    assert a.exact == b.exact == "file_path='server.py'"
    assert a.idents == b.idents


def test_the_pretool_hook_carries_the_caption_beside_the_args_never_inside():
    seen: dict = {}

    async def _cb(input_data, tool_use_id, _ctx):
        seen.update(input_data)
        return {}

    reg = HookRegistry()
    reg.add_pre_tool_use(None, _cb)
    asyncio.run(reg.fire_pre_tool_use(
        session_id="s1", tool_name="Read",
        tool_input={"file_path": "server.py"},
        tool_use_id="call_1", tool_summary="Reading server.py",
    ))

    assert seen["tool_summary"] == "Reading server.py"
    assert SUMMARY_ARG not in seen["tool_input"], (
        "safety matching and the repetition guard read tool_input; "
        "a free-text caption must never appear there"
    )
