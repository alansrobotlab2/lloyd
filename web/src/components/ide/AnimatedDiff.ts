/**
 * AnimatedDiff — choreographs a top-to-bottom animated reveal of changes
 * between the editor's current content and a new target content.
 *
 * Per hunk:
 *   1. Smooth-scroll to center the hunk.
 *   2. Replace the old lines with [oldLines, newLines] concatenated and
 *      decorate them: old → bold strikethrough red, new → bold green.
 *   3. Hold the configured dwell time.
 *   4. Replace that section with just the new lines (decorations cleared).
 *
 * Top-to-bottom presentation; line offsets accumulate as we go so each
 * hunk's coordinates remain valid.
 *
 * Returns a controller with `cancel()` (skip to final state) and a
 * promise that resolves when playback finishes (or is cancelled).
 */

import { diffLines } from 'diff'
import type { editor as Mon, IRange } from 'monaco-editor'

export interface AnimatedDiffOptions {
  dwellMs?: number          // hold time per hunk while old + new are visible
  scrollSettleMs?: number   // pause for smooth-scroll to land
  maxHunks?: number         // skip animation entirely if exceeded
  onProgress?: (i: number, total: number) => void
  onDone?: () => void
}

export interface AnimatedDiffController {
  promise: Promise<void>
  cancel: () => void
}

interface Hunk {
  // 1-based line numbers in the *current* (pre-animation) editor content.
  removeStart: number
  removeEnd: number       // exclusive
  oldLines: string[]
  newLines: string[]
}

function computeHunks(oldText: string, newText: string): Hunk[] {
  const parts = diffLines(oldText, newText)
  const hunks: Hunk[] = []
  let cursor = 1   // 1-based current-line cursor through `oldText`
  let i = 0
  while (i < parts.length) {
    const p = parts[i]
    if (!p.added && !p.removed) {
      // Equal block — advance cursor.
      cursor += p.count ?? p.value.split('\n').length - 1
      i++
      continue
    }
    // Group adjacent add/remove parts into a single hunk so an in-place
    // edit (remove then add at the same spot) renders as one section.
    const removed: string[] = []
    const added: string[] = []
    while (i < parts.length && (parts[i].added || parts[i].removed)) {
      const linesInPart = splitKeepEmpty(parts[i].value)
      if (parts[i].removed) {
        removed.push(...linesInPart)
      } else if (parts[i].added) {
        added.push(...linesInPart)
      }
      i++
    }
    const removeStart = cursor
    const removeEnd = cursor + removed.length   // exclusive
    cursor = removeEnd
    hunks.push({
      removeStart,
      removeEnd,
      oldLines: removed,
      newLines: added,
    })
  }
  return hunks
}

// `diff` library hunks include trailing newlines on each part except the
// last. Splitting via "\n" and dropping a trailing empty element gives us
// the *line* count for that part regardless of trailing-newline state.
function splitKeepEmpty(text: string): string[] {
  if (text === '') return []
  const out = text.split('\n')
  if (out.length > 0 && out[out.length - 1] === '') out.pop()
  return out
}

const ADD_DECO_CLASS = 'ide-diff-add'
const REMOVE_DECO_CLASS = 'ide-diff-remove'

function delay(ms: number): Promise<void> {
  return new Promise(res => setTimeout(res, ms))
}

export function animateDiff(
  editor: Mon.IStandaloneCodeEditor,
  monaco: typeof import('monaco-editor'),
  newContent: string,
  options: AnimatedDiffOptions = {},
): AnimatedDiffController {
  const dwellMs = options.dwellMs ?? 1500
  const scrollSettleMs = options.scrollSettleMs ?? 220
  const maxHunks = options.maxHunks ?? 8

  const model = editor.getModel()
  if (!model) {
    return makeNoop(newContent, editor, options)
  }

  const oldContent = model.getValue()
  if (oldContent === newContent) {
    return makeNoop(newContent, editor, options)
  }

  const hunks = computeHunks(oldContent, newContent)
  if (hunks.length === 0 || hunks.length > maxHunks) {
    // Big or empty diff → skip animation, set value directly.
    return makeNoop(newContent, editor, options)
  }

  let cancelled = false
  const decorationCollections: Mon.IEditorDecorationsCollection[] = []
  const wasReadOnly = editor.getOption(monaco.editor.EditorOption.readOnly) as boolean
  editor.updateOptions({ readOnly: true })

  const promise = (async () => {
    let lineDelta = 0   // running offset from prior hunks
    try {
      for (let i = 0; i < hunks.length; i++) {
        if (cancelled) break
        const h = hunks[i]
        const startLine = h.removeStart + lineDelta
        const removeCount = h.oldLines.length

        // Build the merged-in-place text that shows old + new together.
        const mergedLines = [...h.oldLines, ...h.newLines]

        // Replace the original old-lines region with [old + new]. If old
        // was empty (pure add), insert at startLine before existing content.
        const range = makeLineRange(monaco, startLine, removeCount, mergedLines)
        const editText = mergedLines.join('\n') + (range.trailingNewline ? '\n' : '')
        editor.executeEdits('animated-diff/expand', [{
          range: range.replaceRange,
          text: editText,
          forceMoveMarkers: true,
        }])

        // Decorate: oldLines as strikethrough-red, newLines as bold-green.
        const decoCollection = editor.createDecorationsCollection()
        const decos: Mon.IModelDeltaDecoration[] = []
        const oldStart = startLine
        const oldEnd = startLine + h.oldLines.length - 1
        const newStart = oldEnd + 1
        const newEnd = newStart + h.newLines.length - 1
        if (h.oldLines.length > 0) {
          decos.push({
            range: new monaco.Range(oldStart, 1, oldEnd, Number.MAX_SAFE_INTEGER),
            options: {
              isWholeLine: true,
              className: REMOVE_DECO_CLASS,
              inlineClassName: REMOVE_DECO_CLASS + '-inline',
            },
          })
        }
        if (h.newLines.length > 0) {
          decos.push({
            range: new monaco.Range(newStart, 1, newEnd, Number.MAX_SAFE_INTEGER),
            options: {
              isWholeLine: true,
              className: ADD_DECO_CLASS,
              inlineClassName: ADD_DECO_CLASS + '-inline',
            },
          })
        }
        decoCollection.set(decos)
        decorationCollections.push(decoCollection)

        // Center the merged section, then dwell.
        const focusLine = h.oldLines.length > 0
          ? oldStart + Math.floor(h.oldLines.length / 2)
          : newStart + Math.floor(h.newLines.length / 2)
        editor.revealLineInCenter(focusLine, monaco.editor.ScrollType.Smooth)
        await delay(scrollSettleMs)
        if (cancelled) break

        options.onProgress?.(i, hunks.length)
        await delay(dwellMs)
        if (cancelled) break

        // Collapse: drop the old lines, keep only the new ones. Also
        // clears the decorations on those rows because the model rows
        // themselves are gone.
        const collapseRange = new monaco.Range(
          oldStart, 1,
          oldEnd + 1, 1,   // up through the newline that ends oldEnd
        )
        if (h.oldLines.length > 0) {
          editor.executeEdits('animated-diff/collapse', [{
            range: collapseRange,
            text: '',
            forceMoveMarkers: true,
          }])
        }
        // Whatever decorations remain (the add highlight) — clear once
        // the dwell is over so the editor returns to a clean look.
        decoCollection.clear()
        decorationCollections.pop()

        // Update running offset: net change introduced by this hunk.
        lineDelta += h.newLines.length - h.oldLines.length
      }
    } finally {
      // Ensure the model is exactly the target content even if any edit
      // calc was off by an edge case (last-line newline weirdness, etc.)
      if (!cancelled && model.getValue() !== newContent) {
        editor.executeEdits('animated-diff/finalize', [{
          range: model.getFullModelRange(),
          text: newContent,
          forceMoveMarkers: true,
        }])
      }
      if (cancelled && model.getValue() !== newContent) {
        // Skipped: snap to the target so we don't leave the file
        // half-merged.
        editor.executeEdits('animated-diff/cancel-finalize', [{
          range: model.getFullModelRange(),
          text: newContent,
          forceMoveMarkers: true,
        }])
      }
      // Clean up any lingering decoration collections.
      for (const c of decorationCollections) c.clear()
      editor.updateOptions({ readOnly: wasReadOnly })
      options.onDone?.()
    }
  })()

  return {
    promise,
    cancel: () => { cancelled = true },
  }
}

function makeLineRange(
  monaco: typeof import('monaco-editor'),
  startLine: number,
  removeCount: number,
  _mergedLines: string[],
): { replaceRange: IRange; trailingNewline: boolean } {
  if (removeCount > 0) {
    // Replace lines [startLine, startLine + removeCount) — that's
    // (startLine, 1) through (startLine + removeCount, 1).
    return {
      replaceRange: new monaco.Range(startLine, 1, startLine + removeCount, 1),
      trailingNewline: true,
    }
  }
  // Pure insertion — insert *before* startLine. Use a zero-width range.
  return {
    replaceRange: new monaco.Range(startLine, 1, startLine, 1),
    trailingNewline: true,
  }
}

function makeNoop(
  newContent: string,
  editor: Mon.IStandaloneCodeEditor,
  options: AnimatedDiffOptions,
): AnimatedDiffController {
  const model = editor.getModel()
  if (model && model.getValue() !== newContent) {
    editor.executeEdits('animated-diff/noop', [{
      range: model.getFullModelRange(),
      text: newContent,
      forceMoveMarkers: true,
    }])
  }
  options.onDone?.()
  return {
    promise: Promise.resolve(),
    cancel: () => {},
  }
}
