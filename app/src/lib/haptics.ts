/**
 * Haptic feedback.
 *
 * iOS Safari does not implement `navigator.vibrate` — it is the single biggest
 * tell that a web app is not native. The workaround, which does work on iOS 18+,
 * is the switch-input trick: programmatically clicking a `<label>` bound to an
 * `<input type="checkbox" switch>` makes WebKit play the system toggle haptic.
 *
 * So there are two backends:
 *   - Vibration API where it exists (Android, desktop Chrome with a gamepad, etc.)
 *   - the hidden switch on iOS
 * and a no-op everywhere else. Callers never need to know which is in play.
 *
 * Haptics are a garnish: every call is best-effort and silently does nothing if the
 * platform will not cooperate.
 */

export type HapticStyle = 'light' | 'medium' | 'heavy' | 'success' | 'warning' | 'error' | 'selection'

/** Vibration patterns, in milliseconds. Gaps in the arrays are pauses. */
const PATTERNS: Record<HapticStyle, number | number[]> = {
  selection: 4,
  light: 8,
  medium: 14,
  heavy: 22,
  success: [12, 40, 22],
  warning: [18, 60, 18],
  error: [24, 50, 24, 50, 24],
}

let switchLabel: HTMLLabelElement | null = null
let enabled = true

const supportsVibrate = () =>
  typeof navigator !== 'undefined' && typeof navigator.vibrate === 'function'

/**
 * Builds the hidden switch once, on first use.
 * The elements stay in the DOM but are inert: zero-size, aria-hidden and
 * pointer-events off, so neither VoiceOver nor a stray tap can reach them.
 */
function ensureSwitch(): HTMLLabelElement | null {
  if (switchLabel) return switchLabel
  if (typeof document === 'undefined') return null

  const input = document.createElement('input')
  input.type = 'checkbox'
  // `switch` is the attribute WebKit keys the haptic off. It is unknown to other
  // engines, which simply render a checkbox we never show.
  input.setAttribute('switch', '')
  input.id = 'haptic-switch'

  const label = document.createElement('label')
  label.htmlFor = input.id

  const host = document.createElement('div')
  host.setAttribute('aria-hidden', 'true')
  host.style.cssText =
    'position:fixed;width:0;height:0;overflow:hidden;pointer-events:none;opacity:0;'
  host.append(input, label)
  document.body.append(host)

  switchLabel = label
  return label
}

/** Fires a haptic. Safe to call from any event handler, on any platform. */
export function haptic(style: HapticStyle = 'light'): void {
  if (!enabled) return

  if (supportsVibrate()) {
    try {
      navigator.vibrate(PATTERNS[style])
      return
    } catch {
      // Some browsers throw when the document is not user-activated. Fall through.
    }
  }

  // iOS path. The system toggle haptic has one fixed intensity, so styles collapse
  // to a single tap here; repeating it for the multi-pulse styles would fire the
  // clicks faster than WebKit coalesces them and feels like a stutter, not a pattern.
  const label = ensureSwitch()
  if (label) {
    try {
      label.click()
    } catch {
      /* nothing more to try */
    }
  }
}

/** Respects a user preference stored in settings. */
export function setHapticsEnabled(on: boolean): void {
  enabled = on
}

export function areHapticsEnabled(): boolean {
  return enabled
}
