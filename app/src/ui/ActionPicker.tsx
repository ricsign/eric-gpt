import { haptic } from '../lib/haptics'
import type { ActionState } from '../calls/types'
import './ActionPicker.css'

/**
 * Where someone is with one errand.
 *
 * It lives in ui/ because both screens that hold a list of errands are handed
 * `onToggleAction` and must offer the same three answers over the same stored
 * state. A second copy of a three-state control is the copy that drifts, and
 * the two would then disagree about what a person had already finished.
 */

/**
 * The three answers, in tap order.
 *
 * "Does not apply" is last and is worded as a fact about the question rather
 * than about the reader. A reader with no job at all said five of the ten were
 * not about their life; a list that can only be finished or left undone turns
 * those five into five accusations, and he stops opening it. Saying so has to
 * cost nothing, which is why the third option is a plain statement and not an
 * apology like "skip" or "not for me".
 */
const OPTIONS: { state: ActionState; label: string }[] = [
  { state: 'done', label: 'Done' },
  { state: 'open', label: 'Not yet' },
  { state: 'na', label: 'Does not apply' },
]

export function ActionPicker({
  callId,
  state,
  label,
  onChange,
}: {
  callId: number
  state: ActionState
  /** Names the group for a screen reader, which cannot see the row above it. */
  label: string
  onChange: (callId: number, next: ActionState) => void
}) {
  return (
    <div className="action-picker" role="group" aria-label={label}>
      {OPTIONS.map((option) => (
        <label
          className="action-opt"
          key={option.state}
          data-state={option.state}
          data-on={state === option.state ? '' : undefined}
        >
          {/*
            Native radios rather than buttons: they already carry "exactly one
            of these is true", they already move under the arrow keys, and
            re-picking the answer that is already set does nothing, which is
            the behaviour we want anyway. The input is hidden and the span is
            the chip, so the focus ring is drawn on the span instead.
          */}
          <input
            className="visually-hidden"
            type="radio"
            // Grouped per question, so the arrow keys stay inside one errand
            // instead of walking the length of the list.
            name={`action-${callId}`}
            checked={state === option.state}
            onChange={() => {
              haptic('selection')
              onChange(callId, option.state)
            }}
          />
          <span className="action-opt-t">{option.label}</span>
        </label>
      ))}
    </div>
  )
}
