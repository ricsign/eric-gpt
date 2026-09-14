import { useMemo, useState } from 'react'
import { Napkin } from '../ui/Receipt'
import {
  answerNote,
  answerText,
  dialText,
  sharedFacts,
  shareReceipt,
  type NapkinInput,
} from '../lib/receipt'
import { resolveOptimal, type Call, type Profile } from '../calls/types'
import { formatCallDate } from '../lib/schedule'
import { haptic } from '../lib/haptics'
import './ReceiptScreen.css'

/**
 * The share screen, and the three ways out of it.
 *
 * Image for feeds, plain text for group chats where images die, and a link so
 * the recipient can go and guess the rest. All three ship, because a single
 * share path is a single point of failure for the only thing that spreads this.
 *
 * What the card carries changed completely here, and the reason is the whole
 * job. It used to be the sender's result: a projection to 65, the gap to the
 * right answer, and a stamp reading MONEY LEFT BEHIND. Every reader we showed
 * it to refused to send it, in the same words — it is a picture of them getting
 * money wrong, addressed to people they work with. So the card is now the
 * question and its real answer, which is the thing they said they *would* send,
 * and their own guess is a toggle that starts off.
 */
export function ReceiptScreen({
  call,
  callNo,
  value,
  profile,
  day,
  onNext,
}: {
  call: Call
  /** Position in the set of ten, one-based. Also the deep link on the card. */
  callNo: number
  value: number
  profile: Profile
  day: string
  onNext: () => void
}) {
  const [state, setState] = useState<'idle' | 'shared' | 'copied' | 'failed'>('idle')
  // Off by default. Adding your guess is a decision to put yourself on the
  // card, and the product should not make it for you.
  const [withGuess, setWithGuess] = useState(false)

  const input: NapkinInput = useMemo(
    () => ({
      questionNo: callNo,
      date: formatCallDate(day),
      scene: call.title,
      question: call.question,
      // The profile is resolved here and never handed to the card: only the
      // answer crosses, so nothing downstream can print anyone's pay.
      answer: answerText(resolveOptimal(call.optimal, profile), call.variable),
      note: answerNote(call.variable),
      rule: call.rule,
      givens: sharedFacts(call.fixed),
      guess: withGuess ? dialText(value, call.variable) : undefined,
    }),
    [call, callNo, value, profile, day, withGuess],
  )

  const share = async (mode: 'image' | 'text' | 'link') => {
    haptic('light')
    const outcome = await shareReceipt(input, mode)
    if (outcome === 'cancelled') return
    setState(outcome === 'failed' ? 'failed' : outcome)
    if (outcome !== 'failed') haptic('success')
  }

  return (
    <div className="share scroll">
      <Napkin input={input} />

      <div className="share-actions">
        <button
          className="share-guess press"
          aria-pressed={withGuess}
          onClick={() => {
            haptic('light')
            setWithGuess((on) => !on)
          }}
        >
          <span className="share-guess-box" aria-hidden="true" />
          <span className="data-sm">Put my guess on it</span>
        </button>

        <button className="share-btn share-btn--primary press" onClick={() => share('image')}>
          {state === 'shared' ? 'Sent' : 'Share this'}
        </button>
        <div className="share-btn-row">
          <button className="share-btn press" onClick={() => share('text')}>
            {state === 'copied' ? 'Copied' : 'Copy as text'}
          </button>
          <button className="share-btn press" onClick={() => share('link')}>
            Copy link
          </button>
        </div>
        {state === 'failed' && (
          <p className="share-fail data-sm">Sharing is blocked in this browser.</p>
        )}
      </div>

      <button className="share-next press" onClick={onNext}>
        <span className="data-sm">Tomorrow</span>
        <span className="share-next-teaser">{call.tomorrow}</span>
      </button>
    </div>
  )
}
