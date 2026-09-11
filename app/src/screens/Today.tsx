import { useEffect, useMemo } from 'react'
import { motion } from 'motion/react'
import { Screen } from '../ui/Screen'
import { Card } from '../ui/Card'
import { Button } from '../ui/Button'
import { Icon } from '../ui/Icon'
import { useNav } from '../ui/NavStack'
import { NumberRoll } from '../ui/NumberRoll'
import { useStore } from '../state/store'
import { DRILLS } from '../data/drills'
import { drillIndex, drillNumber, MAX_ATTEMPTS } from '../lib/daily'
import { rankLessons } from '../data/lessons'
import {
  dateInMonths,
  dateShift,
  dayKey,
  money,
  moneyCompact,
  monthYear,
} from '../lib/format'
import { debtFreeDate, independence } from '../lib/calculators'
import { yearsToTarget } from '../lib/finance'
import { spring } from '../lib/motion'
import { DrillScreen } from './Drill'
import { LessonPlayer } from './LessonPlayer'
import './Today.css'

/**
 * Home.
 *
 * The hero is **a date**, not a score. Which date depends on where the person
 * actually is: someone carrying expensive debt gets a Payoff Day, everyone else
 * gets a Coast Day. It is the only thing in the app that is unambiguously theirs,
 * that moves when they do something real, and that is worth reopening to check.
 *
 * Underneath it is one line saying what moved it since they last looked, with the
 * previous date struck through. That line is the entire retention argument.
 *
 * There is no streak, no XP, no level, and no leaderboard. The one metric-like
 * object in the product lives on the You tab and it goes down, not up.
 */
export function TodayScreen() {
  const { state, confirmCommitment, dispatch } = useStore()
  const nav = useNav()
  const today = dayKey()
  const number = drillNumber(today)
  const drill = DRILLS[drillIndex(number, DRILLS.length)]
  const record = state.drills[number]
  const drillDone = record ? record.solved || record.attempts.length >= MAX_ATTEMPTS : false

  const p = state.profile
  const openCommitments = state.commitments.filter((c) => !c.doneAt).slice(0, 2)

  const nextLesson = useMemo(
    () => rankLessons(p, state.lessons, state.jurisdiction ?? 'US', state.predictions.length)[0],
    [p, state.lessons, state.jurisdiction, state.predictions.length],
  )

  /**
   * The date.
   *
   * Debt outranks everything: a payoff date is concrete, near, and moves visibly
   * when you send an extra payment. A retirement projection for someone paying 23%
   * on a card is the wrong number to lead with.
   */
  const hero = useMemo(() => {
    if ((p.debtBalance ?? 0) > 0) {
      const payment = Math.max(25, (p.debtBalance ?? 0) * 0.02)
      const r = debtFreeDate(
        [
          {
            id: 'main',
            name: 'Highest-rate debt',
            balance: p.debtBalance!,
            apr: p.debtApr ?? 0.229,
            monthlyPayment: payment,
          },
        ],
        0,
      )
      const date = dateInMonths(r.months)
      return {
        label: 'Payoff day',
        date,
        fallback: 'Not on this payment',
        sub: r.neverClears
          ? `At ${money(payment)} a month the interest matches what you send, so the balance does not fall. The Debt-free date tool shows what does clear it.`
          : `On ${money(payment)} a month, with ${money(r.totalInterest)} of interest along the way.`,
        tone: 'drag' as const,
      }
    }

    const realReturn = p.assumedReturn - p.assumedInflation
    const annualSpend = (p.income ?? 65_000) * 0.7
    const r = independence({
      age: p.age ?? 30,
      annualSpend,
      invested: p.invested ?? 0,
      monthly: p.monthly ?? 0,
      takeHomeAnnual: (p.income ?? 65_000) * 0.78,
      realReturn,
      retireAge: p.targetAge ?? 65,
    })

    if (r.alreadyCoasting) {
      return {
        label: 'Coast day',
        date: dateInMonths(0),
        fallback: 'Reached',
        sub: `What you have invested already grows into ${moneyCompact(
          r.target,
        )} by ${p.targetAge ?? 65} with nothing added. Everything from here buys time, not security.`,
        tone: 'growth' as const,
      }
    }

    const years = yearsToTarget(
      { principal: p.invested ?? 0, monthly: p.monthly ?? 0, annualRate: realReturn },
      r.coastTarget,
    )

    return {
      label: 'Coast day',
      date: dateInMonths(years * 12),
      fallback: 'Add a monthly amount to see this',
      sub: `The day you could stop adding entirely and still reach ${moneyCompact(
        r.target,
      )} by ${p.targetAge ?? 65}. It needs ${moneyCompact(
        r.coastTarget,
      )} invested, and it arrives long before the full number does.`,
      tone: 'growth' as const,
    }
  }, [p])

  // Remember the date so the next visit can show what moved. Snapshotting on
  // render is deliberate: "since you last looked" means since you last saw it.
  const shift =
    hero.date && state.lastDate && state.lastDate.value !== hero.date
      ? { from: state.lastDate.value, text: dateShift(state.lastDate.value, hero.date) }
      : null

  useEffect(() => {
    if (!hero.date) return
    // Deferred so the strike-through renders once before it is overwritten.
    const id = window.setTimeout(() => dispatch({ type: 'snapshotDate', value: hero.date! }), 2500)
    return () => window.clearTimeout(id)
  }, [hero.date, dispatch])

  return (
    <Screen
      title="Today"
      eyebrow={new Date().toLocaleDateString('en-US', {
        weekday: 'long',
        month: 'long',
        day: 'numeric',
      })}
    >
      <div className="today">
        {/* ---- The date ---- */}
        <section className="today-hero" data-tone={hero.tone}>
          <p className="today-hero-label">{hero.label}</p>
          <p className="today-hero-date num">
            {hero.date ? monthYear(hero.date) : hero.fallback}
          </p>

          {shift?.text && (
            <motion.p
              className="today-hero-shift"
              initial={{ opacity: 0, y: -4 }}
              animate={{ opacity: 1, y: 0 }}
              transition={spring.smooth}
            >
              <s className="num">{monthYear(shift.from, true)}</s> {shift.text} than last time
            </motion.p>
          )}

          <p className="today-hero-sub">{hero.sub}</p>
        </section>

        {/* ---- Things you said you'd do ---- */}
        {openCommitments.length > 0 && (
          <section className="today-section">
            <h2 className="today-section-title">You said you would</h2>
            {openCommitments.map((c) => (
              <Card key={c.id} className="today-commitment">
                <p className="today-commitment-text">
                  <span className="today-commitment-when">When {c.when},</span> I will {c.then}.
                </p>
                {c.worth != null && c.worth > 0 && (
                  <p className="today-commitment-worth num">
                    Worth about <NumberRoll value={c.worth} format={moneyCompact} duration={0.8} />
                  </p>
                )}
                <div className="today-commitment-actions">
                  <Button
                    size="sm"
                    variant="tinted"
                    onClick={() => confirmCommitment(c.id)}
                    feedback="success"
                  >
                    Done it
                  </Button>
                  <Button size="sm" variant="plain" onClick={() => confirmCommitment(c.id)}>
                    Dismiss
                  </Button>
                </div>
              </Card>
            ))}
          </section>
        )}

        {/* ---- One curve, chosen by relevance ---- */}
        {nextLesson && (
          <section className="today-section">
            <h2 className="today-section-title">
              {state.lessons[nextLesson.id] ? 'Worth drawing again' : 'Draw this one next'}
            </h2>
            <Card
              tone="growth"
              onClick={() => nav.push(nextLesson.id, () => <LessonPlayer lessonId={nextLesson.id} />)}
            >
              <p className="today-lesson-title">{nextLesson.title}</p>
              <p className="today-lesson-competence">{nextLesson.competence}</p>
              <p className="today-lesson-meta">
                {nextLesson.minutes} min
                {(p.debtBalance ?? 0) > 0 && nextLesson.triggers.includes('has-debt')
                  ? ' · because you told us about a balance'
                  : ''}
              </p>
            </Card>
          </section>
        )}

        {/* ---- One question, no cadence pressure ---- */}
        <section className="today-section">
          <h2 className="today-section-title">One question</h2>
          <Card onClick={() => nav.push('drill', () => <DrillScreen />)}>
            <p className="today-drill-question">{drill.question}</p>
            <span className="today-drill-cta">
              {drillDone ? 'See the arithmetic again' : 'Three tries · about a minute'}
              <Icon name="chevron" size={14} stroke />
            </span>
          </Card>
        </section>

        <p className="today-footnote">
          Educational only — not investment, tax or legal advice. Every figure here is an
          illustration from assumptions you control, not a prediction. We take no affiliate or
          referral money from anyone.
        </p>
      </div>
    </Screen>
  )
}
