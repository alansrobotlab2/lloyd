"""Shared helper for tests that drive their own copy of `app/routers/messages.py`.

Two files need this: `tests/test_usage_skill_breakdown.py` (#783, the skill
dimension on the usage row) and `tests/test_compaction_record.py` (#1078, the
context-policy dimension on the same row). It used to live inside the first as
`_pristine_messages_module`; the review that refused the #1078 round named the
duplication as an advisory, and the reason it is worth a file is that this
particular helper has one failure mode that is silent:

  Patching attributes on a module copy whose functions were defined against a
  DIFFERENT dict changes nothing for those functions. `mod._run_turn.__globals__`
  is the dict the copy's `compile()` call built — not `vars(app.routers.messages)`.
  So a test that copies the module and then patches the live module's attributes
  (or, the other way round, patches a copy while importing the live module's
  `_run_turn`) drives a turn in which every collaborator it meant to fake is
  real. It does not look broken: the turn completes, an unrelated component
  swallows the surprise, and the assertion downstream fails with a `KeyError` on
  a key that was never written — which is exactly how the last round lost a whole
  review attempt, and why the same test passed when a sibling ran first.

So: get the functions and the patched dict from the same object, and reach that
object through the function's own `__globals__` rather than a remembered attribute
name. `patch_on_copy` below is the only sanctioned way to put a fake in front of
one of these copies.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
from contextlib import ExitStack
from unittest.mock import patch

_SOURCE = "app.routers.messages"
_COPIES: dict[str, object] = {}


def load_messages_copy(monkeypatch, *, name: str, cache: bool = True):
    """A second, private instance of the chat router, importable by no one else.

    `tests/test_session_queue.py` rebinds `app.routers.messages._run_turn` to its
    own recording stubs, so in a full-suite run the live attribute is somebody
    else's fake that writes no usage row at all. A test that drives a turn has to
    own its copy of the router.

    The copy executes the same source file, so it is the same code — and its only
    top-level effects are a logger and its own unused `APIRouter`, so nothing
    registers on the app. It is registered in `sys.modules` under `name` for the
    duration of the test because the module body resolves its own class references
    while executing.

    Caches one copy per `name` for the process. Every fake on it is applied with
    `patch_on_copy` (or `monkeypatch.setattr` on the returned module), both of
    which undo themselves, so a cached copy cannot carry a fake into a later test;
    re-executing a ~2,700-line module per test would only make the suite slower,
    and one `exec_module` raising halfway would leave a dead module object
    registered under the name.
    """
    cached = _COPIES.get(name)
    if cache and cached is not None:
        return cached

    # Imported rather than read out of `sys.modules`: run alone, this test file is
    # the first thing in the process to want the router, and a `KeyError` here is an
    # ordering requirement disguised as a crash — which is the opposite of the point.
    source = importlib.import_module(_SOURCE)
    path = inspect.getsourcefile(source)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader, f"cannot load a copy of {_SOURCE} from {path}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(name, None)
        raise
    # The whole point of the copy: functions and patched globals from one object.
    assert mod._run_turn.__globals__ is vars(mod), (
        f"{name}._run_turn resolves its names somewhere other than vars({name}), "
        "so fakes put on this module would not be seen by the turn under test"
    )
    _COPIES[name] = mod
    return mod


def patch_on_copy(
    stack: ExitStack,
    obj,
    targets: list[tuple[str, object]],
) -> None:
    """Put these fakes into the dict the copy's functions read their names from.

    `obj` is the copy; one `patch.dict` over `vars(obj)` covers every name, and the
    driven `_run_turn`/`post_message` were compiled against that same mapping — the
    loader asserts it. Reaching instead for `patch("app.routers.messages.X")` puts
    the fake in the LIVE router, where the copy's turn never looks: that is the
    mistake that made the last round's driven turn run every real collaborator while
    appearing faked, and it is why this helper takes the copy at all.

    `patch.dict` over the whole mapping rather than `setattr`, because a name the
    copy imported rather than defined (`_build_subliminal_entry` comes from
    `app/routers/_messages_subliminal.py`) has its definition elsewhere: patching the
    function object's own `__globals__` would then leave the copy's binding, the one
    its turn reads, untouched. One `patch.dict` for all the names, not one per name,
    so a target that raises on entry cannot leave an earlier fake behind.

    A name the copy does not carry is a failure, not a skip: `patch.dict` would
    happily install it, and the REAL collaborator would keep running behind the test,
    which reads as a pass with the seam unexercised. Three names this file originally
    patched (`messages._inject_subl_messages`, `messages._drive_inner_voice`,
    `app.sessions_io.load_session_messages`) were attributes of nothing at all, and an
    `except AttributeError` hid that for a whole round; `grep -c` for each in the
    module it named returns 0.
    """
    names = vars(obj)
    missing = [name for name, _ in targets if name not in names]
    if missing:
        raise AttributeError(
            f"the copy {names.get('__name__')!r} carries no {'these names' if len(missing) > 1 else 'name'} "
            f"{missing}. Every fake in this list has to land where the driven code "
            "looks it up — installing one under a name the module does not use leaves "
            "the real collaborator running, and the test would then pass without the "
            "seam under test being exercised at all."
        )
    stack.enter_context(patch.dict(names, dict(targets)))
