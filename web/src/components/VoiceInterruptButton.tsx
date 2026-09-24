import { useEffect, useState } from 'react'
import { VolumeX } from 'lucide-react'
import { Button } from '@/components/ui/button'

// The speaking gate is a volume threshold, so it drops to false in every gap
// between words. A button that followed it directly would go disabled under
// the cursor mid-sentence; hold it up this long after the last audible frame
// so it stays clickable for the whole utterance.
export const SPEAKING_HOLD_MS = 1200

/** `speaking`, held true for `holdMs` after it last went false. */
export function useSpeakingHold(speaking: boolean, holdMs: number = SPEAKING_HOLD_MS): boolean {
  const [held, setHeld] = useState(speaking)
  useEffect(() => {
    if (speaking) {
      setHeld(true)
      return
    }
    const t = setTimeout(() => setHeld(false), holdMs)
    return () => clearTimeout(t)
  }, [speaking, holdMs])
  return speaking || held
}

/**
 * Click-to-interrupt for voice output — the ruled design (USER.md: "the user
 * clicks the interrupt button, interrupt() → clear_queue()"). VoiceRoom's
 * `interrupt()` tells the worker to drop its TTS queue and cancels the
 * in-flight turn; until this button it had no caller, so Lloyd could not be
 * stopped mid-sentence. Rendered whenever voice is engaged (a control that
 * appears only mid-speech shifts the layout under the pointer), enabled only
 * while Lloyd is speaking.
 */
export function VoiceInterruptButton({
  active,
  onInterrupt,
}: {
  active: boolean
  onInterrupt: () => void
}) {
  return (
    <Button
      variant={active ? 'destructive' : 'outline'}
      size="sm"
      onClick={onInterrupt}
      disabled={!active}
      aria-label="Interrupt Lloyd (stop speaking)"
      className="h-7 text-xs w-full justify-start"
      title="Stop Lloyd speaking — clears queued speech and cancels the turn"
    >
      <VolumeX className="w-3.5 h-3.5 mr-1.5" />
      Interrupt
    </Button>
  )
}
