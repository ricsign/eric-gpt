import { useMemo } from 'react'
import { Histogram } from '../ui/Histogram'
import { COMPUTE } from '../calls/compute'
import { blendCrowd } from '../lib/crowd'
import {
  judge,
  referenceValue,
  resolveOptimal,
  stepCount,
  type BreakdownLine,
  type Call,
  type Profile,
  type Verdict,
} from '../calls/types'
import { money } from '../lib/format'
import { haptic } from '../lib/haptics'
import './Outcome.css'

/**
 * The reveal's own words, keyed by the compute function behind each question.
 *
 * The registry says what was asked. This says why the answer is the answer,
 * which is a different job and a different voice — and it is the only copy on
 * the screen a reader can check the figures against. Keyed by `compute` rather
 * than by id, so a sentence can never end up attached to a question whose
 * arithmetic it does not describe.
 *
 * `dollars` marks a dial whose number is money. Five of the ten dials carry an
 * empty unit, because the readout prints the unit immediately after the figure
 * and "3200$" is not a thing — but only four of those five are dollars, and the
 * fifth counts days. Nothing in the record tells them apart, so this does.
 */
const WHY: Record<string, { mechanism: string; dollars?: true }> = {
  employerMatch: {
    mechanism:
      'Your boss adds fifty cents for every dollar you put in, up to 6% of your pay, so below 6% part of that offer simply stays with them.',
  },
  debtSplit: {
    dollars: true,
    mechanism:
      'Both cards charge interest every month on whatever is left, so the same dollar clears more than twice as much interest on the 24.99% card as on the 11.99% one.',
  },
  emergencyFund: {
    mechanism:
      'Cash earns about 4% while investing earns about 7%, so every month of bills you hold costs you the difference — and holding too little means putting the next broken thing on a 24.99% card.',
  },
  promoDeadline: {
    mechanism:
      'Nothing is charged while the deal runs, but one month past the deadline and 26.99% is added back to day one on the whole $3,000.',
  },
  anchorOffer: {
    mechanism:
      'Every raise after this one is a percentage of the number you agree to now, so a few points follow you for the rest of your career, while the chance of the offer being pulled only bites once the ask gets big.',
  },
  withholding: {
    dollars: true,
    mechanism:
      'Your tax bill is the same either way, so a big refund only means you handed the money over early and got it back with nothing added — and you can owe up to a tenth of the bill before the IRS charges interest.',
  },
  repairOrReplace: {
    dollars: true,
    mechanism:
      'The first few hundred dollars buy most of the months a repair adds, so past about half of what the car is worth you are paying more than the newer car would cost over the same five years.',
  },
  feeDragCall: {
    dollars: true,
    mechanism:
      'The fee comes out every year on everything you hold, growth included, so a small difference in the yearly cost takes a visible slice of what you end up with.',
  },
  rentVsBuy: {
    mechanism:
      'Buying and then selling the house costs about 9% of the price, and it takes roughly five years of paying down the loan and prices creeping up to earn that back.',
  },
  timingMarket: {
    mechanism:
      'The best days land in the middle of the worst weeks, so stepping out to dodge the drop usually means missing the bounce that pays for it.',
  },
}

/**
 * The reveal.
 *
 * What shipped before opened with a signed six-figure number in red under the
 * words "money left behind". Every reviewer who read it said the same thing:
 * there is no way to tell where that figure came from, so it can only be
 * believed or ignored, and it punishes a good-faith answer with the largest,
 * reddest element on the page.
 *
 * The order here is the order a sceptical reader needs. What you said and what
 * the answer is, in a sentence. The mechanism in one line. Then the two or
 * three figures that mechanism turns on, yours beside the answer's, because a
 * reader told us the comparison is the only place they could check us. Then the
 * one errand that makes any of it real, which is the part that changes a life
 * rather than a screen.
 *
 * The forty-year projection is not here at all. See `isHorizon`.
 */
export function Outcome({
  call,
  questionNo,
  total,
  value,
  profile,
  localCrowd,
  onReceipt,
  onPersonalise,
  personalised,
}: {
  call: Call
  /** Position in the set of ten, one-based. */
  questionNo: number
  total: number
  value: number
  profile: Profile
  /** Answers this device has recorded here, same length as the seed. */
  localCrowd: number[] | undefined
  onReceipt: () => void
  onPersonalise: () => void
  /** False while every figure on the screen is still running on the default salary. */
  personalised: boolean
}) {
  const compute = COMPUTE[call.compute]
  const why = WHY[call.compute]
  const dollars = why.dollars === true
  const unit = call.variable.unit

  const verdict = judge(value, call.optimal, profile)
  const answer = resolveOptimal(call.optimal, profile)
  // The nearest right answer, never the middle of a band: someone who banked
  // three months of bills is measured against three months, not four and a half.
  const reference = referenceValue(value, call.optimal, profile)

  const mine = useMemo(() => compute(value, profile), [compute, value, profile])
  const theirs = useMemo(() => compute(reference, profile), [compute, reference, profile])

  const rows = useMemo(
    () => comparison(mine.breakdown, theirs.breakdown, value, reference),
    [mine, theirs, value, reference],
  )

  const steps = stepCount(call.variable)
  const crowd = useMemo(
    () => blendCrowd(call.crowd, localCrowd ?? new Array(steps).fill(0)),
    [call.crowd, localCrowd, steps],
  )

  // A right answer is its own reference, so the two columns would print the
  // same figures twice. One column, and the header says whose it is.
  const twoColumns = reference !== value

  // The distance to the nearest right answer. Every dial in the set steps by a
  // whole number today; the rounding is here so the first fractional one does
  // not put "under by 0.7500000000000002" at the top of the screen.
  const gap = Math.round(Math.abs(value - reference) * 100) / 100

  return (
    <div className="outcome scroll">
      <header className="outcome-head">
        {/* Under and over are facts about a dial, not a grade. The chip states
            which side and how far, and then gets out of the way. */}
        <span className="outcome-chip" data-v={verdict}>
          {verdict === 'optimal'
            ? 'Nailed it'
            : `${verdict === 'short' ? 'Under' : 'Over'} by ${reading(gap, unit, dollars)}`}
        </span>
        <span className="data-sm num">
          Question {questionNo} of {total}
        </span>
      </header>

      {/*
        A sentence, not a figure. The number someone needs out of this screen is
        the one they can repeat to a colleague on Monday, and "you said 3%, the
        answer is at least 6%" survives being repeated. A hero number does not:
        it has to be explained before it means anything, and by then it has
        already been screenshotted without the explanation.
      */}
      <h1 className="outcome-said">
        <span className="outcome-said-yours">You said {spoken(value, unit, dollars)}.</span>{' '}
        <span className="outcome-said-answer">
          {answerSentence(verdict, answer, reference, unit, dollars)}
        </span>
      </h1>
      <p className="outcome-said-note data-sm">{call.variable.label}</p>

      <section className="outcome-why">
        <p className="outcome-kicker data-sm">Why</p>
        <p className="outcome-mechanism">{why.mechanism}</p>

        <table className="outcome-table">
          <caption className="visually-hidden">
            Your answer beside the right one, on the figures the answer turns on.
          </caption>
          <thead>
            <tr>
              <td className="outcome-corner" />
              <th scope="col">
                <span className="outcome-col-k data-sm">You said</span>
                <span className="outcome-col-v num">{reading(value, unit, dollars)}</span>
              </th>
              {twoColumns && (
                <th scope="col" data-answer>
                  {/* Not "the answer": Courier at this tracking puts the
                      article on a line of its own and the header goes crooked. */}
                  <span className="outcome-col-k data-sm">Answer</span>
                  <span className="outcome-col-v num">{reading(reference, unit, dollars)}</span>
                </th>
              )}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.label}>
                <th scope="row" className="outcome-row-k">
                  {r.label}
                </th>
                <td className="num">{r.yours}</td>
                {twoColumns && (
                  <td className="num" data-answer>
                    {r.answer}
                  </td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      {/* The payoff. Everything above it is a lesson; this is the only part
          that changes what happens in someone's actual week. */}
      <section className="outcome-do">
        <p className="outcome-kicker data-sm">Do this week</p>
        <p className="outcome-do-text">{call.action}</p>
      </section>

      <section className="outcome-rule">
        <p className="outcome-kicker data-sm">The rule</p>
        <p className="outcome-rule-text">{call.rule}</p>
      </section>

      <section className="outcome-crowd">
        <p className="outcome-kicker data-sm">What most people pick</p>
        <Histogram
          distribution={crowd}
          min={call.variable.min}
          step={call.variable.step}
          unit={unit}
          value={value}
          // The nearest right answer, so the marker on the chart stands where
          // the sentence at the top of the screen says it does. Handing this
          // the midpoint of a band, which is what it used to get, drew a
          // "best" at 10.5% under a headline that said 6%.
          optimal={reference}
          verdict={verdict}
        />
        <p className="outcome-source data-sm">
          From published research on what people really do. Not other people using this app.
        </p>
      </section>

      {/* Asked here rather than at the door: the figures above are already on
          a salary, so there is something concrete to make theirs. */}
      {!personalised && (
        <button
          className="outcome-personalise press"
          onClick={() => {
            haptic('light')
            onPersonalise()
          }}
        >
          Those figures were on {money(profile.salary)} a year. Make them yours?
        </button>
      )}

      <footer className="outcome-foot">
        <p className="outcome-fine">{call.assumptions}</p>
        <p className="outcome-fine">Estimates, not advice.</p>
      </footer>

      <button
        className="outcome-print press"
        onClick={() => {
          haptic('medium')
          onReceipt()
        }}
      >
        Print my receipt
      </button>
    </div>
  )
}

/** One row of the comparison: the same figure at two positions on the dial. */
interface Row {
  label: string
  yours: string
  answer: string
  /** The figure is different at the two positions. */
  moved: boolean
  /** The figure is the dial position said back in other words. */
  echo: boolean
}

/**
 * The two or three figures worth putting side by side.
 *
 * Built out of the compute breakdown rather than written by hand, so the table
 * cannot drift away from the maths it is quoting — the receipt prints the same
 * lines from the same source. Three rules do the picking:
 *
 *  - Rows are paired by label. The refund question renames its own line when
 *    the sign flips ("back in April" becomes "you owe in April"), and those two
 *    are not the same row however neatly they would line up.
 *  - A row that only repeats the dial back is passed over. "You said 54, the
 *    answer is 3" is already the headline; spending a row of the most important
 *    block on the screen restating it teaches nothing.
 *  - So is a figure that does not move — but only while at least two others do.
 *
 * Both of those last two are preferences rather than rules, applied in order
 * and each abandoned the moment it would leave fewer than two rows. A table
 * with one row in it is not a comparison, and the case that produces one is
 * the reader who got the answer exactly right, who has earned the opposite of
 * an empty screen.
 */
function comparison(
  mine: BreakdownLine[],
  theirs: BreakdownLine[],
  value: number,
  reference: number,
): Row[] {
  const paired: Row[] = []

  for (const m of mine) {
    const t = theirs.find((x) => x.label === m.label)
    if (!t) continue
    if (isHorizon(m) || isHorizon(t)) continue
    paired.push({
      label: m.label,
      yours: m.value,
      answer: t.value,
      moved: m.value !== t.value,
      echo: echoes(m.value, value) && echoes(t.value, reference),
    })
  }

  const best = paired.filter((r) => r.moved && !r.echo)
  if (best.length >= 2) return best.slice(0, 3)

  const rest = paired.filter((r) => !r.echo)
  return (rest.length >= 2 ? rest : paired).slice(0, 3)
}

/**
 * Is this line a projection to 65?
 *
 * Those figures are all over the compute breakdowns and none of them belong on
 * this screen. Two reasons, and the second is the one that settles it. They are
 * nominal dollars forty years out, so "$398K" is not $398K of anything anybody
 * can picture, and stating it in today's money means discounting it by an
 * inflation assumption no part of this app models — a number invented in a
 * screen file to make another number look honest. And a six-figure figure next
 * to a $16 one flattens the $16, which is the figure that is actually true
 * today and the one the errand below acts on.
 *
 * Detected two ways because the breakdowns name them two ways: a label that
 * says 65, or a value in the compact form (`$1.34M`), which this app uses for
 * the far horizon and nowhere else.
 */
function isHorizon(line: BreakdownLine): boolean {
  return /\b65\b/.test(line.label) || /\d\s*[KMB]\b/.test(line.value)
}

/** Does this formatted figure just say the dial position back? `54 BP` at 54. */
function echoes(text: string, dial: number): boolean {
  const m = /-?\$?\d[\d,]*(?:\.\d+)?/.exec(text)
  if (!m) return false
  const n = Number(m[0].replace(/[$,]/g, ''))
  return Number.isFinite(n) && Math.abs(n - dial) < 1e-9
}

/** The dial position as a noun: `6%`, `$3,000`, `owe $500`, `4`. */
function reading(v: number, unit: string, dollars: boolean): string {
  if (!dollars) return `${v}${unit}`
  // Only the refund dial goes below zero, and there a negative is not a
  // negative dollar amount — it is a bill, and it reads as one.
  return v < 0 ? `owe ${money(-v)}` : money(v)
}

/** The same, inside a sentence: "you said to owe $500". */
function spoken(v: number, unit: string, dollars: boolean): string {
  return dollars && v < 0 ? `to ${reading(v, unit, dollars)}` : reading(v, unit, dollars)
}

/**
 * The half of the headline that carries the answer.
 *
 * A band does not have an answer, it has an edge: below it the reader needs to
 * know the floor, above it they need to know they are past it and nothing is
 * wrong. "At least 6%" and "12 months is plenty" are both true statements about
 * the same shape, and saying either one as a flat "the answer is" would be
 * false — which is exactly what the old copy did when it called a 12% saver
 * overcommitted.
 */
function answerSentence(
  verdict: Verdict,
  answer: number | { min: number; max: number },
  reference: number,
  unit: string,
  dollars: boolean,
): string {
  const say = (v: number) => reading(v, unit, dollars)

  if (typeof answer === 'number') {
    return verdict === 'optimal'
      ? 'That is the answer.'
      : `The answer is ${spoken(reference, unit, dollars)}.`
  }

  if (verdict === 'optimal') {
    return Number.isFinite(answer.max)
      ? `Anything from ${say(answer.min)} to ${say(answer.max)} works.`
      : `Anything from ${say(answer.min)} up works.`
  }

  return verdict === 'short'
    ? `The answer is at least ${say(reference)}.`
    : `${say(reference)} is plenty.`
}
