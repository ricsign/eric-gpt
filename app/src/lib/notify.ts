/**
 * The daily notification.
 *
 * Two rules, both product decisions rather than technical ones:
 *
 *  1. Permission is asked only after the third completed call. Asking on first
 *     run is how an app gets denied permanently before it has given anyone a
 *     reason to say yes, and on iOS a denial is effectively irreversible without
 *     a trip to Settings.
 *  2. The copy is always the scenario itself, never an exhortation. "A $2,400
 *     repair on a $5,000 car. 9,400 people have already called it." is a reason
 *     to open the app. "Come back!" is a reason to turn notifications off.
 *
 * Web push on iOS additionally requires the app to have been added to the Home
 * Screen, which is a small single-digit share of visitors — so this is a bonus
 * surface, never the retention plan.
 */

/** Completed calls before the permission prompt is allowed to appear. */
export const ASK_AFTER_CALLS = 3

export type NotifyState = 'unsupported' | 'default' | 'granted' | 'denied'

export function notifyState(): NotifyState {
  if (typeof Notification === 'undefined') return 'unsupported'
  return Notification.permission as NotifyState
}

/** Whether to show the opt-in, given how many real calls have been played. */
export function shouldAsk(completedCalls: number): boolean {
  return completedCalls >= ASK_AFTER_CALLS && notifyState() === 'default'
}

/**
 * Requests permission. Must be called from a user gesture, or Safari ignores it
 * silently — which looks identical to the user declining.
 */
export async function requestNotify(): Promise<NotifyState> {
  if (typeof Notification === 'undefined') return 'unsupported'
  try {
    return (await Notification.requestPermission()) as NotifyState
  } catch {
    return 'denied'
  }
}

/**
 * The notification body for a given call.
 *
 * `played` is omitted rather than invented when there is no live count. A
 * fabricated "9,400 people have already called it" in a product whose entire
 * pitch is that the numbers are real would be the worst possible place to lie.
 */
export function notifyCopy(teaser: string, played: number | null): string {
  if (played === null || played <= 0) return teaser
  return `${teaser} ${played.toLocaleString('en-US')} have already called it.`
}
