import { describe, expect, it, vi } from 'vitest'
import { isValidElement, type ReactElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { VoiceInterruptButton } from './VoiceInterruptButton'

// vitest runs in the node environment here (no jsdom), so render to markup
// for what is shown and call the element's own onClick for what it does.

type ButtonEl = ReactElement<{ onClick: () => void; disabled: boolean }>

describe('VoiceInterruptButton', () => {
  it('renders an enabled, labelled button while Lloyd is speaking', () => {
    const html = renderToStaticMarkup(
      <VoiceInterruptButton active={true} onInterrupt={() => {}} />,
    )
    expect(html).toContain('<button')
    expect(html).toContain('aria-label="Interrupt Lloyd (stop speaking)"')
    expect(html).not.toMatch(/disabled=""/)
  })

  it('calls interrupt on click', () => {
    const onInterrupt = vi.fn()
    const el = VoiceInterruptButton({ active: true, onInterrupt }) as ButtonEl
    expect(isValidElement(el)).toBe(true)
    expect(el.props.disabled).toBe(false)
    el.props.onClick()
    expect(onInterrupt).toHaveBeenCalledTimes(1)
  })

  it('is disabled when Lloyd is not speaking', () => {
    const html = renderToStaticMarkup(
      <VoiceInterruptButton active={false} onInterrupt={() => {}} />,
    )
    expect(html).toMatch(/disabled=""/)
    const el = VoiceInterruptButton({ active: false, onInterrupt: () => {} }) as ButtonEl
    expect(el.props.disabled).toBe(true)
  })
})
