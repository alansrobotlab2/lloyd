import { describe, expect, it } from "vitest";

import { activityLabel, hasSessionTitle, sessionLabel } from "./sessionLabel";

// The fallback chain has to be identical on every surface (chat history,
// chat header, dashboard, Inner Voice picker), which is why it is one
// function — and why it is the first thing in web/ with a test.
describe("sessionLabel", () => {
  it("prefers the title, then the preview, then the id", () => {
    expect(sessionLabel({ title: "Setting up TTS", preview: "hey can you" }, "20260906_214751_iv1620"))
      .toBe("Setting up TTS");
    expect(sessionLabel({ title: null, preview: "hey can you" }, "20260906_214751_iv1620"))
      .toBe("hey can you");
    expect(sessionLabel({}, "20260906_214751_iv1620")).toBe("20260906_214751_iv1620");
  });

  it("treats a whitespace title as no title", () => {
    expect(sessionLabel({ title: "   ", preview: " opening words " }, "id")).toBe("opening words");
    expect(hasSessionTitle({ title: "   " })).toBe(false);
    expect(hasSessionTitle({ title: "Named" })).toBe(true);
  });

  it("never returns an empty label", () => {
    expect(sessionLabel({}, null)).toBe("Untitled session");
    expect(sessionLabel({ title: "", preview: "" }, "  ")).toBe("Untitled session");
  });
});

describe("activityLabel", () => {
  it("is empty with no activity, so an idle row renders nothing", () => {
    expect(activityLabel(null)).toBe("");
    expect(activityLabel(undefined)).toBe("");
  });

  it("joins the detail, which is what tells a quick read from a long build", () => {
    expect(activityLabel({ kind: "tool", label: "Bash", detail: "supervisorctl restart", at: 0 } as never))
      .toBe("Bash · supervisorctl restart");
  });

  it("falls back to the kind's own wording when the runner sent no label", () => {
    expect(activityLabel({ kind: "prefill", label: "", detail: "", at: 0 } as never)).toBe("reading context");
  });
});
