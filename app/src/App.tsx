import { useCallback, useEffect, useMemo, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { Device } from './ui/Device'
import { Home } from './screens/Home'
import { Salary } from './screens/Salary'
import { CallScreen } from './screens/Call'
import { Outcome } from './screens/Outcome'
import { ReceiptScreen } from './screens/ReceiptScreen'
import { Tomorrow } from './screens/Tomorrow'
import { Tab } from './screens/Tab'
import { Rules } from './screens/Rules'
import { CALLS } from './calls/registry'
import { COMPUTE } from './calls/compute'
import { judge, referenceValue, stepCount, type CallResult } from './calls/types'
import { compoundDay, nextQuestionId, unlockedToday } from './lib/schedule'
import { callNoFromPath, pathForCall, setPath } from './lib/route'
import { profileOrDefault, scoredResults, useStore } from './state/store'
import './App.css'

type Screen = 'home' | 'question' | 'outcome' | 'receipt' | 'profile' | 'tomorrow' | 'tab' | 'rules'

/**
 * Ten questions, one a day, and the two surfaces that hold what you keep.
 *
 *   home -> question -> reveal -> share -> home
 *
 * Two things about this shape are deliberate reversals of what shipped before.
 *
 * Home is the entry, not a salary form. The previous build opened on "what do
 * you make a year?" under the words "first run", which asked a stranger for
 * their income before telling them the category, the benefit or the time cost.
 * The income question now lives after the first reveal, where it has been
 * earned, and it collects age too — the old build hardcoded 30 for everyone
 * while every "by 65" figure depended on it.
 *
 * The queue is player-relative. It used to index the library by the global
 * day, which meant a new player's first screen was whatever the shuffle landed
 * on; everyone now starts at question one whenever they arrive.
 */
export function App() {
  const profile = useStore((s) => s.profile)
  const results = useStore((s) => s.results)
  const localCrowd = useStore((s) => s.localCrowd)
  const actions = useStore((s) => s.actions)
  const setProfile = useStore((s) => s.setProfile)
  const setAction = useStore((s) => s.setAction)
  const lockIn = useStore((s) => s.lockIn)

  const day = compoundDay()
  const scored = useMemo(() => scoredResults(results), [results])
  const answered = useMemo(() => new Set(scored.map((r) => r.callId)), [scored])
  const orderedIds = useMemo(() => CALLS.map((c) => c.id), [])
  const nextId = useMemo(() => nextQuestionId(orderedIds, answered), [orderedIds, answered])
  const spentDays = useMemo(() => scored.map((r) => r.day), [scored])
  const openToday = unlockedToday(spentDays)

  /**
   * A shared link opens the question it names, not today's.
   *
   * That is the point of the link on a shared card: one that resolved to
   * "today" would send the recipient somewhere different from the sender. Read
   * once on mount, because changing it later would yank the question out from
   * under someone mid-drag.
   */
  const [linked] = useState(() =>
    typeof location === 'undefined' ? null : callNoFromPath(location.pathname),
  )

  const [openId, setOpenId] = useState<number | null>(linked)
  const [screen, setScreen] = useState<Screen>(linked ? 'question' : 'home')
  const [answer, setAnswer] = useState<number | null>(null)

  const call = useMemo(
    () => CALLS.find((c) => c.id === openId) ?? CALLS[0],
    [openId],
  )
  const questionNo = useMemo(
    () => CALLS.findIndex((c) => c.id === call.id) + 1,
    [call],
  )

  // Keep the address bar honest so a refresh lands in the same place.
  // replaceState, not push: Back should leave the app rather than walk
  // backwards through answers already committed.
  useEffect(() => {
    setPath(screen === 'home' ? '/' : pathForCall(questionNo))
  }, [screen, questionNo])

  const open = useCallback((id: number) => {
    setOpenId(id)
    setAnswer(null)
    setScreen('question')
  }, [])

  const commit = useCallback(
    (value: number) => {
      const compute = COMPUTE[call.compute]
      const p = profileOrDefault(profile)
      const at65 = compute(value, p).at65
      // Measured against the nearest correct answer, so a right answer shows
      // exactly zero rather than a phantom gap to a range's midpoint.
      const best = referenceValue(value, call.optimal, p)

      const result: CallResult = {
        callId: call.id,
        value,
        verdict: judge(value, call.optimal, p),
        delta: at65 - compute(best, p).at65,
        at65,
        day,
        // A question closes once answered. Replaying it, or arriving on one
        // through a shared link, is practice: it must not consume a day or
        // move the count, or neither number means anything.
        practice: answered.has(call.id) || !openToday,
      }

      lockIn(result, stepCount(call.variable))
      setAnswer(value)
      setScreen('outcome')
    },
    [call, profile, day, answered, openToday, lockIn],
  )

  const p = profileOrDefault(profile)

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
            {screen === 'home' && (
              <Home
                calls={CALLS}
                answered={answered}
                nextId={openToday ? nextId : null}
                onStart={open}
              />
            )}

            {screen === 'question' && (
              <CallScreen
                call={call}
                questionNo={questionNo}
                total={CALLS.length}
                profile={p}
                onLockIn={commit}
              />
            )}

            {screen === 'outcome' && answer !== null && (
              <Outcome
                call={call}
                questionNo={questionNo}
                total={CALLS.length}
                value={answer}
                profile={p}
                localCrowd={localCrowd[call.id]}
                onReceipt={() => setScreen('receipt')}
                onPersonalise={() => setScreen('profile')}
                personalised={profile !== null}
              />
            )}

            {screen === 'profile' && (
              <Salary
                initial={p}
                onDone={(next) => {
                  setProfile(next)
                  setScreen('outcome')
                }}
                onBack={() => setScreen('outcome')}
              />
            )}

            {screen === 'receipt' && answer !== null && (
              <ReceiptScreen
                call={call}
                callNo={questionNo}
                value={answer}
                profile={p}
                day={day}
                onNext={() => setScreen('tomorrow')}
              />
            )}

            {screen === 'tomorrow' && (
              <Tomorrow
                teaser={call.tomorrow}
                nextTitle={nextId === null ? '' : (CALLS.find((c) => c.id === nextId)?.title ?? '')}
                results={scored}
                onTab={() => setScreen('tab')}
                onRules={() => setScreen('rules')}
                onHome={() => setScreen('home')}
              />
            )}

            {screen === 'tab' && (
              <Tab
                results={scored}
                calls={CALLS}
                actions={actions}
                onClose={() => setScreen('home')}
                onToggleAction={setAction}
              />
            )}

            {screen === 'rules' && (
              <Rules
                results={scored}
                calls={CALLS}
                actions={actions}
                onClose={() => setScreen('home')}
                onToggleAction={setAction}
              />
            )}
          </motion.div>
        </AnimatePresence>
      </div>
    </Device>
  )
}
