// Which bench-family transcripts the Runs panel is showing, and what it says about
// them (#2648). Pure so the copy can be tested without a DOM — the repo has no React
// test environment, and `tests/test_dashboard_label_contract.py::
// test_the_source_tests_run_and_the_shipped_bundle_builds` runs this file's `.test.ts`
// over the real source.
//
// `run_bench_sdk` began recording one transcript per trial on 2026-09-14, and one day
// of one detached grid wrote 273 of them (2026-10-07) against a listing 150 rows wide.
// So `GET /api/background/sessions` hides the family by default and reports
// `bench_hidden`. The three functions here are that control's whole UI.

export function benchToggleLabel(showBench: boolean, hiddenBench: number): string {
  return showBench
    ? 'hide bench'
    : `show ${hiddenBench} bench run${hiddenBench === 1 ? '' : 's'}`
}

export function hiddenBenchSuffix(hiddenBench: number): string {
  return hiddenBench > 0 ? ` · ${hiddenBench} bench withheld` : ''
}

export function emptyStateCopy(hiddenBench: number): string {
  return hiddenBench > 0
    ? `No non-bench runs in this window — ${hiddenBench} bench transcript`
      + `${hiddenBench === 1 ? '' : 's'} withheld.`
    : 'No background runs recorded yet'
}
