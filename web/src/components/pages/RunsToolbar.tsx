// The Runs panel's window line (#2648): what is on screen, how much of the eligible
// store the endpoint withheld from it, and the one control that asks for the withheld
// class.
//
// Deliberately hook-free and dependency-light, for two reasons. The first is honesty: a
// component with no hooks is a function, so `BackgroundPage.test.ts` can call it, walk the
// element it renders and press the button it renders — the click reaches this control's
// own `onClick` rather than a copy of it. The second is that this row is the only part of
// the panel a test can render at all: `web/vitest.config.ts` runs in `environment:
// "node"`, so `react-dom/server` can produce its markup while a mounted
// `BackgroundPage` (contexts, Monaco, lucide) cannot.

import { benchToggleLabel, hiddenBenchSuffix } from './benchRows'

export type RunsToolbarProps = {
  /// Which class is currently showing: false = the bench family is withheld.
  showBench: boolean
  /// How many bench-family transcripts the endpoint withheld from THIS response. Rendered
  /// into both the button's label and the suffix, so the number a reader sees is the
  /// server's count and not a count of the rows that arrived.
  hiddenBench: number
  /// Rows on screen. The panel passes `state.rows.length`; a window of 150 that is
  /// 148 trials deep and a window of 150 that is genuinely full read differently here.
  shown: number
  onToggle: () => void
}

export function RunsToolbar({
  showBench, hiddenBench, shown, onToggle,
}: RunsToolbarProps) {
  return (
    <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
      <button
        type="button"
        onClick={onToggle}
        className="rounded border border-border px-2 py-0.5 hover:bg-muted"
      >
        {benchToggleLabel(showBench, hiddenBench)}
      </button>
      <span>
        {shown} run{shown === 1 ? '' : 's'}
        {hiddenBenchSuffix(hiddenBench)}
      </span>
    </div>
  )
}
