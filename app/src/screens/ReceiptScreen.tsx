import { useMemo, useState } from 'react'
import { Receipt } from '../ui/Receipt'
import { shareReceipt, type ReceiptInput } from '../lib/receipt'
import { COMPUTE } from '../calls/compute'
import { judge, referenceValue, type Call, type Profile } from '../calls/types'
import { formatCallDate } from '../lib/schedule'
import { moneyCompact } from '../lib/format'
import { haptic } from '../lib/haptics'
import './ReceiptScreen.css'

/**
 * The receipt, and the three ways out of it.
 *
 * Image for feeds, plain text for group chats where images die, and a deep link
 * so the recipient plays the same call before seeing anyone's answer. All three
 * ship, because a single share path is a single point of failure for the only
 * metric that matters on day one.
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
  callNo: number
  value: number
  profile: Profile
  day: string
  onNext: () => void
}) {
  const [state, setState] = useState<'idle' | 'shared' | 'copied' | 'failed'>('idle')

  const input: ReceiptInput = useMemo(() => {
    const compute = COMPUTE[call.compute]
    const mine = compute(value, profile)
    const best = compute(referenceValue(value, call.optimal, profile), profile)
    return {
      callNo,
      date: formatCallDate(day),
      title: call.title,
      verdict: judge(value, call.optimal, profile),
      value,
      unit: call.variable.unit,
      breakdown: mine.breakdown,
      at65: mine.at65,
      delta: mine.at65 - best.at65,
    }
  }, [call, callNo, value, profile, day])

  const share = async (mode: 'image' | 'text' | 'link') => {
    haptic('light')
    const outcome = await shareReceipt(input, mode)
    if (outcome === 'cancelled') return
    setState(outcome === 'failed' ? 'failed' : outcome)
    if (outcome !== 'failed') haptic('success')
  }

  return (
    <div className="receipt-screen scroll">
      <Receipt input={input} />

      <div className="receipt-actions">
        <button className="receipt-btn receipt-btn--primary press" onClick={() => share('image')}>
          {state === 'shared' ? 'Shared' : 'Share receipt'}
        </button>
        <div className="receipt-btn-row">
          <button className="receipt-btn press" onClick={() => share('text')}>
            {state === 'copied' ? 'Copied' : 'Copy as text'}
          </button>
          <button className="receipt-btn press" onClick={() => share('link')}>
            Copy link
          </button>
        </div>
        {state === 'failed' && (
          <p className="receipt-fail data-sm">Sharing is blocked in this browser.</p>
        )}
      </div>

      <button className="receipt-next press" onClick={onNext}>
        <span className="data-sm">Tomorrow</span>
        <span className="receipt-next-teaser">{call.tomorrow}</span>
      </button>

      <p className="receipt-total data-sm num">
        This call: {input.delta >= 0 ? '+' : '−'}
        {moneyCompact(Math.abs(input.delta))} at 65
      </p>
    </div>
  )
}
