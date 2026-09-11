import { useMemo } from 'react'
import { motion } from 'motion/react'
import { Screen } from '../ui/Screen'
import { Card } from '../ui/Card'
import { Button } from '../ui/Button'
import { Icon } from '../ui/Icon'
import { useNav } from '../ui/NavStack'
import { NumberRoll } from '../ui/NumberRoll'
import { useStore } from '../state/store'
import { DRILLS } from '../data/drills'
import { drillIndex, drillNumber, glyph, MAX_ATTEMPTS } from '../lib/daily'
import { rankLessons } from '../data/lessons'
import { dayKey, duration, money, moneyCompact, percent } from '../lib/format'
import { debtFreeDate, independence } from '../lib/calculators'
import { futureValue } from '../lib/finance'
import { spring } from '../lib/motion'
import { DrillScreen } from './Drill'
import { LessonPlayer } from './LessonPlayer'
import './Today.css'

/**
 * Home.
 *
 * The ordering is the argument. The drill is first because it is the daily ritual
 * and the growth loop. The standing position — one computed number, not a score —
 * is second, because that is the reason to still be here in month six. Open
 * commitments come before new lessons, because confirming an action you actually
 * took is the only thing on this screen that reflects real-world change.
 *
 * There is deliberately no XP total, no level, and no leaderboard.
 */
export function TodayScreen() {
  const { state, confirmCommitment } = useStore()
  const nav = useNav()
  const today = dayKey()
  const number = drillNumber(today)
  const drill = DRILLS[drillIndex(number, DRILLS.length)]
  const record = state.drills[number]
  const finished = record ? record.solved || record.attempts.length >= MAX_ATTEMPTS : false

  const p = state.profile
  const openCommitments = state.commitments.filter((c) => !c.doneAt).slice(0, 2)

  const nextLesson = useMemo(
    () => rankLessons(p, state.lessons, state.jurisdiction ?? 'US')[0],
    [p, state.lessons, state.jurisdiction],
  )

  /**
   * The standing number.
   *
   * Which one depends on where the person actually is. Someone carrying expensive
   * debt does not need a retirement projection; they need a date. Showing the same
   * hero metric to everyone is how a dashboard becomes decoration.
   */
  const position = useMemo(() => {
    if ((p.debtBalance ?? 0) > 0) {
      const r = debtFreeDate(
        [
          {
            id: 'main',
            name: 'Highest-rate debt',
            balance: p.debtBalance!,
            apr: p.debtApr ?? 0.229,
            monthlyPayment: Math.max(25, p.debtBalance! * 0.02),
          },
        ],
        0,
      )
      return {
        label: 'Debt-free on minimums',
        value: r.neverClears ? 'Not on this payment' : duration(r.months),
        sub: r.neverClears
          ? 'At this payment the interest matches what you send. The Debt-free date tool shows what does clear it.'
          : `${money(r.totalInterest)} of interest along the way. Raising the payment changes both numbers sharply.`,
        tone: 'drag' as const,
        tool: 'debt',
      }
    }

    if ((p.monthly ?? 0) > 0 || (p.invested ?? 0) > 0) {
      const r = independence({
        age: p.age ?? 30,
        annualSpend: (p.income ?? 65_000) * 0.7,
        invested: p.invested ?? 0,
        monthly: p.monthly ?? 0,
        takeHomeAnnual: (p.income ?? 65_000) * 0.78,
        realReturn: p.assumedReturn - p.assumedInflation,
      })
      if (r.alreadyCoasting) {
        return {
          label: 'You are past coasting',
          value: 'Already there',
          sub: 'On these assumptions, what you have invested grows into your target by 65 with no further contributions. Everything you add now buys time, not security.',
          tone: 'growth' as const,
          tool: 'fi',
        }
      }
      return {
        label: 'Coast point',
        value: Number.isFinite(r.yearsToCoast) ? `${Math.round(r.yearsToCoast)} years` : '—',
        sub: `At ${moneyCompact(
          r.coastTarget,
        )} invested you could stop adding entirely and still reach your target by 65. That is the milestone worth aiming at, and it is much nearer than the full number.`,
        tone: 'growth' as const,
        tool: 'fi',
      }
    }

    const end = futureValue({
      principal: 0,
      monthly: 100,
      annualRate: p.assumedReturn,
      years: Math.max(10, (p.targetAge ?? 65) - (p.age ?? 30)),
    })
    return {
      label: 'What $100 a month becomes',
      value: moneyCompact(end.balance),
      sub: `On a ${percent(p.assumedReturn)} assumption, by ${p.targetAge ?? 65}. Add your own numbers in You to make this yours.`,
      tone: 'growth' as const,
      tool: 'growth',
    }
  }, [p])

  return (
    <Screen
      title="Today"
      eyebrow={new Date().toLocaleDateString('en-US', {
        weekday: 'long',
        month: 'long',
        day: 'numeric',
      })}
      right={
        state.streak.current > 0 ? (
          <span className="today-streak" aria-label={`${state.streak.current} day streak`}>
            <Icon name="flame" size={16} />
            <span className="num">{state.streak.current}</span>
          </span>
        ) : undefined
      }
    >
      <div className="today">
        {/* ---- The daily drill ---- */}
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={spring.smooth}>
          <Card
            tone={finished ? 'default' : 'growth'}
            onClick={() => nav.push('drill', () => <DrillScreen />)}
          >
            <div className="today-drill">
              <div className="today-drill-head">
                <span className="today-drill-label">Daily drill #{number}</span>
                {finished && (
                  <span className="today-drill-glyph" aria-hidden="true">
                    {glyph({
                      number,
                      attempts: record!.attempts,
                      solved: record!.solved,
                      streak: state.streak.current,
                    })}
                  </span>
                )}
              </div>

              <p className="today-drill-question">
                {finished ? drill.question : drill.question}
              </p>

              <span className="today-drill-cta">
                {finished ? 'See the arithmetic and share' : 'Three tries · about a minute'}
                <Icon name="chevron" size={14} stroke />
              </span>
            </div>
          </Card>
        </motion.div>

        {/* ---- Where you stand ---- */}
        <section className="today-section">
          <h2 className="today-section-title">Where you stand</h2>
          <Card tone={position.tone}>
            <p className="today-position-label">{position.label}</p>
            <p className="today-position-value num">{position.value}</p>
            <p className="today-position-sub">{position.sub}</p>
          </Card>
        </section>

        {/* ---- Things you said you'd do ---- */}
        {openCommitments.length > 0 && (
          <section className="today-section">
            <h2 className="today-section-title">You said you would</h2>
            {openCommitments.map((c) => (
              <Card key={c.id} className="today-commitment">
                <p className="today-commitment-text">
                  <span className="today-commitment-when">When {c.when},</span> {c.then}.
                </p>
                {c.worth != null && c.worth > 0 && (
                  <p className="today-commitment-worth num">
                    Worth about <NumberRoll value={c.worth} format={moneyCompact} duration={0.8} />
                  </p>
                )}
                <div className="today-commitment-actions">
                  <Button size="sm" variant="tinted" onClick={() => confirmCommitment(c.id)} feedback="success">
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

        {/* ---- One lesson, chosen by relevance ---- */}
        {nextLesson && (
          <section className="today-section">
            <h2 className="today-section-title">
              {state.lessons[nextLesson.id] ? 'Worth revisiting' : 'Next for you'}
            </h2>
            <Card onClick={() => nav.push(nextLesson.id, () => <LessonPlayer lessonId={nextLesson.id} />)}>
              <p className="today-lesson-title">{nextLesson.title}</p>
              {/* The competence reads as a sentence, not a label — at caption size in
                  caps it wrapped to two shouted lines and buried the actual title. */}
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

        <p className="today-footnote">
          Educational only — not investment, tax or legal advice. Every figure here is an
          illustration from assumptions you control, not a prediction. We take no affiliate or
          referral money from anyone.
        </p>
      </div>
    </Screen>
  )
}
