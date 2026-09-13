import { useCallback, useEffect, useMemo, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { Device } from './ui/Device'
import { Salary } from './screens/Salary'
import { CallScreen } from './screens/Call'
import { Outcome } from './screens/Outcome'
import { ReceiptScreen } from './screens/ReceiptScreen'
import { Tomorrow } from './screens/Tomorrow'
import { Tab } from './screens/Tab'
import { Rules } from './screens/Rules'
import { CALLS, callById } from './calls/registry'
import { COMPUTE } from './calls/compute'
import {
  judge,
  optimalValue,
  stepCount,
  type CallResult,
} from './calls/types'
import { callIndex, callNumber, compoundDay } from './lib/schedule'
import { callNoFromPath, pathForCall, setPath } from './lib/route'
import { profileOrDefault, resultFor, scoredResults, useStore } from './state/store'
import './App.css'

type Screen = 'call' | 'outcome' | 'receipt' | 'tomorrow' | 'tab' | 'rules'

/**
 * The whole app is one linear loop plus two side surfaces.
 *
 *   cold open -> drag -> lock in -> outcome -> receipt -> tomorrow
 *
 * There is no tab bar. A persistent nav would put chrome in permanent
 * competition with the control, and the control is the product. The Tab and
 * Rules are reachable from the ends of the loop, where the player has just
 * finished rather than started.
 */
export function App() {
  const profile = useStore((s) => s.profile)
  const results = useStore((s) => s.results)
  const localCrowd = useStore((s) => s.localCrowd)
  const setProfile = useStore((s) => s.setProfile)
  const lockIn = useStore((s) => s.lockIn)

  const day = compoundDay()
  const todayNo = callNumber(day)

  /**
   * A deep link opens the call it names, not today's.
   *
   * That is the whole point of the link on a shared receipt: a link that
   * resolved to "today" would be useless the moment the day rolled over, and
   * the recipient would see a different call from the one they were sent. The
   * path is read once, on mount — changing it later would yank the call out
   * from under someone mid-drag.
   */
  const [linkedNo] = useState(() =>
    typeof location === 'undefined' ? null : callNoFromPath(location.pathname),
  )
  const callNo = linkedNo ?? todayNo
  const call = useMemo(() => CALLS[callIndex(callNo, CALLS.length)], [callNo])

  // Keep the address bar honest, so a refresh or a copied URL lands in the same
  // place. replaceState, not push: the loop is linear and Back should leave the
  // app rather than walk backwards through committed answers.
  useEffect(() => {
    setPath(pathForCall(callNo))
  }, [callNo])

  const existing = resultFor(results, call.id)
  const [screen, setScreen] = useState<Screen>(existing ? 'outcome' : 'call')
  const [answer, setAnswer] = useState<number | null>(existing?.value ?? null)

  const commit = useCallback(
    (value: number) => {
      const compute = COMPUTE[call.compute]
      const p = profileOrDefault(profile)
      const best = optimalValue(call.optimal)
      const verdict = judge(value, call.optimal)
      const at65 = compute(value, p).at65

      const result: CallResult = {
        callId: call.id,
        value,
        verdict,
        delta: at65 - compute(best, p).at65,
        at65,
        day,
        // A call closes once answered. Coming back the same day, or arriving on
        // an older call through a shared link, is practice: it must not move the
        // Tab, or the number stops meaning anything.
        practice: Boolean(existing) || callNo !== todayNo,
      }

      lockIn(result, stepCount(call.variable))
      setAnswer(value)
      setScreen('outcome')
    },
    [call, profile, day, existing, lockIn, callNo, todayNo],
  )

  if (!profile) {
    return (
      <Device>
        <Salary onDone={setProfile} />
      </Device>
    )
  }

  const p = profileOrDefault(profile)
  const nextCall = callById(CALLS[callIndex(callNo + 1, CALLS.length)].id)

  return (
    <Device>
      <div className="app">
        <AnimatePresence mode="wait" initial={false}>
          <motion.div
            key={screen}
            className="app-screen"
            initial={{ opacity: 0, x: 24 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -18 }}
            // Spring, not a fade — 300ms at damping ~28 is the UIKit push feel.
            transition={{ type: 'spring', visualDuration: 0.3, bounce: 0.02 }}
          >
            {screen === 'call' && (
              <CallScreen
                call={call}
                callNo={callNo}
                profile={p}
                playedCount={null}
                onLockIn={commit}
              />
            )}

            {screen === 'outcome' && answer !== null && (
              <Outcome
                call={call}
                callNo={callNo}
                value={answer}
                profile={p}
                localCrowd={localCrowd[call.id]}
                onReceipt={() => setScreen('receipt')}
              />
            )}

            {screen === 'receipt' && answer !== null && (
              <ReceiptScreen
                call={call}
                callNo={callNo}
                value={answer}
                profile={p}
                day={day}
                onNext={() => setScreen('tomorrow')}
              />
            )}

            {screen === 'tomorrow' && (
              <Tomorrow
                teaser={call.tomorrow}
                nextTitle={nextCall?.title ?? ''}
                results={scoredResults(results)}
                onTab={() => setScreen('tab')}
                onRules={() => setScreen('rules')}
              />
            )}

            {screen === 'tab' && (
              <Tab
                results={scoredResults(results)}
                profile={p}
                onClose={() => setScreen('tomorrow')}
              />
            )}

            {screen === 'rules' && (
              <Rules
                results={scoredResults(results)}
                calls={CALLS}
                onClose={() => setScreen('tomorrow')}
              />
            )}
          </motion.div>
        </AnimatePresence>
      </div>
    </Device>
  )
}
