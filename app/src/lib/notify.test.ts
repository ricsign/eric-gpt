import { describe, expect, it } from 'vitest'
import { ASK_AFTER_CALLS, notifyCopy, shouldAsk } from './notify'

describe('shouldAsk', () => {
  it('never asks before the third completed call', () => {
    // Asking on first run is how an app gets denied permanently before it has
    // earned a yes, and on iOS that denial is effectively irreversible.
    for (let n = 0; n < ASK_AFTER_CALLS; n++) expect(shouldAsk(n)).toBe(false)
  })

  it('does not ask in an environment without Notification', () => {
    // vitest has no Notification global, so this also pins the unsupported path.
    expect(shouldAsk(ASK_AFTER_CALLS)).toBe(false)
    expect(shouldAsk(99)).toBe(false)
  })
})

describe('notifyCopy', () => {
  const teaser = 'A $2,400 repair on a $5,000 car.'

  it('leads with the scenario, never an exhortation', () => {
    const copy = notifyCopy(teaser, 9400)
    expect(copy.startsWith(teaser)).toBe(true)
    expect(copy.toLowerCase()).not.toContain('come back')
    expect(copy.toLowerCase()).not.toContain('don\'t forget')
    expect(copy.toLowerCase()).not.toContain('streak')
  })

  it('adds the crowd only when the count is real', () => {
    expect(notifyCopy(teaser, 9400)).toContain('9,400')
    // A fabricated count in a product whose pitch is that the numbers are real
    // would be the worst possible place to lie.
    expect(notifyCopy(teaser, null)).toBe(teaser)
    expect(notifyCopy(teaser, 0)).toBe(teaser)
  })
})
