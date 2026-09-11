import {
  costOfWaiting,
  doublingTime,
  employerMatch,
  feeDrag,
  futureValue,
  growthSeries,
  payOffDebt,
  ruleOf72,
} from '../lib/finance'
import { duration, money, moneyCompact, percent } from '../lib/format'
import type { Profile } from '../state/store'
import { FACTS } from './facts'
import type { Beat, Lesson } from './lesson-types'

/**
 * The lessons.
 *
 * Each one exists to correct a *specific, documented* wrong belief — not to cover a
 * topic. That is the whole architectural bet: topic-organised curricula are the
 * intervention shape the evidence says explains almost none of the variance in real
 * financial behaviour, while single counter-intuitive concepts delivered near a
 * decision move it measurably.
 *
 * Two rules every lesson here obeys:
 *
 *  - **Compute, never conclude.** A lesson may show what $500/month becomes. It may
 *    never say "you should invest $500/month". The legal line in this category is
 *    personalisation, not topic, and no disclaimer cures an instruction.
 *  - **Forward framing only.** "Starting today puts you here", never "you already
 *    lost $47,000 by waiting". The regret framing is both worse pedagogy and, in
 *    2026, the thing that gets quote-tweeted as the joke rather than shared as the
 *    insight.
 */

/** Fallbacks so a lesson still works before onboarding has collected anything. */
function withDefaults(p: Profile) {
  return {
    age: p.age ?? 28,
    income: p.income ?? 65_000,
    invested: p.invested ?? 0,
    monthly: p.monthly ?? 300,
    debtBalance: p.debtBalance ?? 0,
    debtApr: p.debtApr ?? 0.229,
    targetAge: p.targetAge ?? 65,
    rate: p.assumedReturn,
    inflation: p.assumedInflation,
  }
}

/* ========================================================================== *
 * 1. Growth is not a line
 * ========================================================================== */

const growthIsNotALine: Lesson = {
  id: 'growth-is-not-a-line',
  title: 'Growth is not a line',
  competence: 'You can predict where compounding actually ends up',
  misconception:
    'People project savings by multiplying — "$300 a month for 30 years is about $108,000" — because human intuition for exponential processes is close to linear.',
  citation: {
    text: 'McKenzie & Liersch (2011) found people underestimate compound growth so severely that simply correcting the misperception raised retirement contributions.',
  },
  triggers: ['always'],
  jurisdiction: 'any',
  minutes: 4,
  concepts: ['exponential-growth', 'contributions-vs-growth'],
  build: (profile) => {
    const p = withDefaults(profile)
    const years = Math.max(10, p.targetAge - p.age)
    const end = futureValue({
      principal: p.invested,
      monthly: p.monthly,
      annualRate: p.rate,
      years,
    })
    const linearGuess = p.invested + p.monthly * 12 * years

    const beats: Beat[] = [
      {
        kind: 'anchor',
        body: `You put away ${money(p.monthly)} a month. You are ${p.age}. Left alone until ${p.targetAge}, that is ${years} years of contributions.`,
        note: `Everything here assumes ${percent(p.rate)} a year. You can change that assumption any time — it is yours, not ours.`,
      },
      {
        kind: 'probe',
        mode: 'estimate',
        question: `Before we show you: what do you think that account is worth at ${p.targetAge}?`,
        answer: end.balance,
        unit: 'usd',
        tolerance: 0.2,
        min: linearGuess * 0.5,
        max: Math.max(end.balance * 1.6, linearGuess * 4),
        // Starting at the sum-of-contributions is deliberate: it is the wrong answer
        // most people give, so the slider begins exactly where the misconception is.
        start: linearGuess,
        because: `Almost everyone lands near ${moneyCompact(
          linearGuess,
        )} — the contributions added up. That is the linear answer, and it is the one the mind reaches for.`,
      },
      {
        kind: 'reveal',
        headline: `${moneyCompact(end.balance)}`,
        body: `You will have paid in ${money(end.contributed)}. The other ${money(
          end.growth,
        )} was not paid in by anyone. Watch where the two bands separate — that year is the whole point.`,
        chart: () => ({
          points: growthSeries({
            principal: p.invested,
            monthly: p.monthly,
            annualRate: p.rate,
            years,
          }),
        }),
      },
      {
        kind: 'mechanism',
        sentence: 'Each year\'s growth is calculated on last year\'s growth, so the amount that grows is itself growing.',
        detail:
          'A line adds the same amount every year. Compounding adds a percentage of a bigger number every year. Early on the difference is invisible, which is exactly why it is so easy to underestimate.',
        visual: 'split',
      },
      {
        kind: 'worked',
        setup: `Take one dollar at ${percent(p.rate)}, and just track the doublings.`,
        steps: [
          { label: 'Time for money to double', value: `${doublingTime(p.rate).toFixed(1)} years` },
          { label: `Dollars after ${Math.round(doublingTime(p.rate))} years`, value: '$2' },
          {
            label: `After ${Math.round(doublingTime(p.rate) * 2)} years`,
            value: '$4',
            blankable: true,
            answer: 4,
            unit: 'usd',
          },
          {
            label: `After ${Math.round(doublingTime(p.rate) * 3)} years`,
            value: '$8',
            blankable: true,
            answer: 8,
            unit: 'usd',
          },
        ],
        conclusion:
          'Four doublings is sixteen times, not four times. Counting doublings is the one mental model that survives — it turns a multiplying process into something you can count on your fingers.',
      },
      {
        kind: 'practice',
        items: [
          {
            prompt: `At ${percent(p.rate)}, money roughly doubles every ${Math.round(
              doublingTime(p.rate),
            )} years. A dollar invested at 25 is worth about how much at 65?`,
            options: [
              { label: '4 dollars', correct: false, why: 'That is four doublings added, not multiplied.' },
              {
                label: '16 dollars',
                correct: true,
                why: `Forty years is about four doublings: 2, 4, 8, 16.`,
              },
              { label: '8 dollars', correct: false, why: 'That is three doublings — you have four.' },
              { label: '40 dollars', correct: false, why: 'Doubling grows fast, but not that fast.' },
            ],
          },
          {
            prompt:
              'Two people each save the same amount each month. One starts at 25 and stops at 35. The other starts at 35 and never stops. At 65, who has more?',
            options: [
              {
                label: 'The one who started at 25, often',
                correct: true,
                why: 'Ten early years get thirty years of doubling on top. Thirty later years get very few. Time in the market does more work than the amount, which is the counter-intuitive part.',
              },
              {
                label: 'The one who saved for thirty years, always',
                correct: false,
                why: 'More contributions, but each one has less time to compound. Depending on the rate, the early saver can still win outright.',
              },
            ],
            transfer: true,
          },
          {
            prompt: 'Which change makes the biggest difference to the final number?',
            options: [
              { label: 'Adding ten more years', correct: true, why: 'Time enters the calculation as an exponent. Everything else is multiplication.' },
              { label: 'Doubling the monthly amount', correct: false, why: 'It doubles the result. Ten more years can more than double it.' },
              { label: 'Finding an extra 1% of return', correct: false, why: 'Powerful, but not as powerful as another decade, and far less reliable.' },
            ],
            transfer: true,
          },
        ],
      },
      {
        kind: 'rule',
        name: 'The Rule of 72',
        statement: '72 divided by your return, as a percentage, is roughly how many years your money takes to double.',
        example: `At ${percent(p.rate)}: 72 ÷ ${(p.rate * 100).toFixed(
          0,
        )} ≈ ${ruleOf72(p.rate).toFixed(0)} years. The exact answer is ${doublingTime(p.rate).toFixed(
          1,
        )}. The shortcut is closest around 8% and drifts a little either side.`,
      },
      {
        kind: 'action',
        when: 'the next time you get a raise',
        then: 'move half of it into the same account, before you have lived a month at the new salary',
        options: [
          { label: "I'll do that", commits: true },
          { label: 'Remind me at my next review', commits: true },
          { label: 'Not now', commits: false },
        ],
        worth: (prof) => {
          const q = withDefaults(prof)
          const yrs = Math.max(10, q.targetAge - q.age)
          const bump = (q.income * 0.03) / 12 / 2
          return futureValue({ principal: 0, monthly: bump, annualRate: q.rate, years: yrs }).balance
        },
      },
    ]
    return beats
  },
}

/* ========================================================================== *
 * 2. What starting today is worth
 * ========================================================================== */

const startingToday: Lesson = {
  id: 'starting-today',
  title: 'What starting today is worth',
  competence: 'You can price ten years of delay in dollars',
  misconception:
    'Delay feels like it costs the contributions you skipped. It costs several times that, because the skipped years are the ones with the most doubling left.',
  citation: {
    text: 'Goda et al. showed that presenting the consequence of delay in dollars — rather than in percentages — measurably changed contribution behaviour.',
  },
  triggers: ['always', 'new-job', 'first-paycheck'],
  jurisdiction: 'any',
  minutes: 3,
  concepts: ['time-in-market', 'cost-of-delay'],
  build: (profile) => {
    const p = withDefaults(profile)
    const years = Math.max(15, p.targetAge - p.age)
    const input = {
      principal: p.invested,
      monthly: p.monthly,
      annualRate: p.rate,
      years,
    }
    const waiting = costOfWaiting(input, 10)

    return [
      {
        kind: 'anchor',
        body: `Two versions of you, both saving ${money(p.monthly)} a month until ${p.targetAge}. One starts this month. The other starts in ten years.`,
        note: 'Same monthly amount. Same return. The only difference is when they begin.',
      },
      {
        kind: 'probe',
        mode: 'estimate',
        question:
          'The later starter skips ten years of contributions. How much smaller is their final balance?',
        answer: waiting.gap,
        unit: 'usd',
        tolerance: 0.25,
        min: waiting.contributionsSkipped * 0.5,
        max: waiting.gap * 1.8,
        // Anchored at "just the skipped contributions" — the intuitive wrong answer.
        start: waiting.contributionsSkipped,
        because: `The instinct is ${moneyCompact(
          waiting.contributionsSkipped,
        )} — the payments that never happened. That is the floor, not the answer.`,
      },
      {
        kind: 'reveal',
        headline: `Starting now is worth ${moneyCompact(waiting.gap)} more`,
        body: `Only ${money(
          waiting.contributionsSkipped,
        )} of that is money the later starter never paid in. The remaining ${money(
          waiting.growthForfeited,
        )} is growth that never got the chance to happen. Every dollar of delayed contribution costs about ${waiting.multiple.toFixed(
          1,
        )} dollars at the finish.`,
        chart: () => ({
          points: growthSeries(input),
          comparison: growthSeries({ ...input, years: Math.max(0, years - 10) }),
          comparisonLabel: 'Starting in ten years',
        }),
      },
      {
        kind: 'mechanism',
        sentence:
          'The years you skip are not the small early ones — they are the ones whose growth would have had the longest to compound.',
        detail:
          'A dollar invested today gets every doubling between now and the end. A dollar invested in ten years misses the first one. Missing the first doubling halves what that dollar becomes.',
        visual: 'doubling',
      },
      {
        kind: 'worked',
        setup: 'One dollar, doubling every ten years, over forty years.',
        steps: [
          { label: 'Invested at year 0', value: '$16 at the end' },
          { label: 'Invested at year 10', value: '$8 at the end', blankable: true, answer: 8, unit: 'usd' },
          { label: 'Invested at year 20', value: '$4 at the end' },
          { label: 'Invested at year 30', value: '$2 at the end' },
        ],
        conclusion:
          'Each decade of delay halves the result for that dollar. Not reduces — halves. That is why the gap is so much bigger than the contributions.',
      },
      {
        kind: 'practice',
        items: [
          {
            prompt: 'Which does more for a final balance?',
            options: [
              { label: 'Starting five years earlier', correct: true, why: 'Those five years get the longest runway, so they carry the most doubling.' },
              { label: 'Saving 20% more each month for the whole time', correct: false, why: 'It helps, and it is under your control — but early years are worth more per dollar.' },
            ],
          },
          {
            prompt:
              'You can only do one this year: open the account with a small amount, or spend the year researching the perfect fund and open it next January.',
            options: [
              {
                label: 'Open it now with the small amount',
                correct: true,
                why: 'A year of compounding on a small amount beats a perfect choice made a year late. The gap between a good fund and the best fund is far smaller than the gap between starting and waiting.',
              },
              {
                label: 'Research first — getting it right matters',
                correct: false,
                why: 'Getting it roughly right now beats getting it exactly right later. This is the most common and most expensive form of delay.',
              },
            ],
            transfer: true,
          },
          {
            prompt: 'Someone is 45 and has not started. What does this lesson say?',
            options: [
              {
                label: 'The same thing: today is the earliest remaining start',
                correct: true,
                why: 'The maths compares today against later, never against a past you cannot change. Twenty years still contains two doublings.',
              },
              {
                label: 'It is too late for compounding to matter',
                correct: false,
                why: 'This is the belief that turns a late start into no start. Two doublings is four times the money.',
              },
            ],
            transfer: true,
          },
        ],
      },
      {
        kind: 'rule',
        name: 'Today is the earliest remaining start',
        statement:
          'The only comparison that changes anything is today against later. Never today against a past you cannot go back to.',
        example: `For you, ten years from now instead of now is a ${moneyCompact(
          waiting.gap,
        )} difference. Next month instead of this month is small. The decision is which of those you are making.`,
      },
      {
        kind: 'action',
        when: 'your next payday arrives',
        then: 'set up one automatic transfer, for any amount you would not notice, on the day the money lands',
        options: [
          { label: 'Setting it up today', commits: true },
          { label: 'On my next payday', commits: true },
          { label: 'Not now', commits: false },
        ],
        worth: () => waiting.gap,
      },
    ]
  },
}

/* ========================================================================== *
 * 3. Compounding runs backwards too
 * ========================================================================== */

const compoundingInReverse: Lesson = {
  id: 'compounding-in-reverse',
  title: 'Compounding runs backwards too',
  competence: 'You can work out what a minimum payment actually costs',
  misconception:
    'The minimum payment reads as the amount the card is asking for. It is the amount that keeps the balance alive for as long as possible.',
  citation: {
    text: 'Stango & Zinman (2009) showed that people who underestimate exponential growth systematically borrow more and save less — the same bias, running in the other direction.',
  },
  triggers: ['has-debt', 'statement-day'],
  jurisdiction: 'any',
  minutes: 4,
  concepts: ['debt-compounding', 'minimum-payment'],
  build: (profile) => {
    const p = withDefaults(profile)
    const balance = p.debtBalance > 0 ? p.debtBalance : 4_200
    const apr = p.debtApr

    // The standard consumer-card formula: 1% of principal plus the month's interest,
    // floored at $25. It is designed to amortise extremely slowly.
    const minimum = Math.max(25, balance * 0.01 + (balance * apr) / 12)
    const doubled = minimum * 2

    const minPlan = payOffDebt({ balance, apr, monthlyPayment: minimum })
    const doublePlan = payOffDebt({ balance, apr, monthlyPayment: doubled })

    return [
      {
        kind: 'anchor',
        body: `${money(balance)} on a card at ${percent(apr)}. The statement asks for ${money(
          minimum,
        )} this month.`,
        note: 'A typical minimum is 1% of the balance plus that month\'s interest.',
      },
      {
        kind: 'probe',
        mode: 'estimate',
        question: `Paying exactly the minimum every month, how long until the balance reaches zero?`,
        answer: minPlan.months / 12,
        unit: 'years',
        tolerance: 0.25,
        min: 1,
        max: Math.max(25, (minPlan.months / 12) * 1.5),
        start: 3,
        because:
          'Most people guess three to five years. The minimum is calculated to make that impossible.',
      },
      {
        kind: 'reveal',
        headline: minPlan.neverPaysOff
          ? 'It never reaches zero'
          : `${duration(minPlan.months)} — and ${money(minPlan.totalInterest)} in interest`,
        body: minPlan.neverPaysOff
          ? 'At this payment the interest charged each month is as large as the payment, so the balance does not fall. This is not a cautionary tale; it is arithmetic.'
          : `You would pay ${money(minPlan.totalPaid)} to clear ${money(
              balance,
            )}. Doubling the payment to ${money(doubled)} clears it in ${duration(
              doublePlan.months,
            )} and costs ${money(doublePlan.totalInterest)} — saving ${money(
              minPlan.totalInterest - doublePlan.totalInterest,
            )}.`,
        contrast: [
          {
            label: `Minimum, ${money(minimum)}/mo`,
            value: minPlan.neverPaysOff ? 'never' : duration(minPlan.months),
            tone: 'drag',
          },
          {
            label: `Double it, ${money(doubled)}/mo`,
            value: duration(doublePlan.months),
            tone: 'growth',
          },
        ],
      },
      {
        kind: 'mechanism',
        sentence:
          'The minimum is recalculated as a percentage of a shrinking balance, so it shrinks too — always leaving a little more to charge interest on.',
        detail:
          'It is the same curve as your investment account, pointed the other way. On a card at this rate, compounding is working, just not for you.',
        visual: 'reverse',
      },
      {
        kind: 'worked',
        setup: `The first month on ${money(balance)} at ${percent(apr)}.`,
        steps: [
          { label: 'Interest charged this month', value: money((balance * apr) / 12) },
          { label: 'Minimum payment', value: money(minimum) },
          {
            label: 'How much came off the balance',
            value: money(minimum - (balance * apr) / 12),
            blankable: true,
            answer: Math.round(minimum - (balance * apr) / 12),
            unit: 'usd',
          },
        ],
        conclusion: `Of the ${money(minimum)} you sent, ${money(
          (balance * apr) / 12,
        )} never touched the debt. Every dollar above the minimum goes entirely to principal — which is why the second dollar is worth so much more than the first.`,
      },
      {
        kind: 'practice',
        items: [
          {
            prompt: 'You have $100 spare. Where does it do the most good on this card?',
            options: [
              {
                label: 'Added to this month\'s payment',
                correct: true,
                why: `Above the minimum, every dollar reduces principal directly — and removes all the future interest that dollar would have generated at ${percent(apr)}.`,
              },
              {
                label: 'Saved for a bigger payment in six months',
                correct: false,
                why: 'Six months of interest gets charged in the meantime, on a balance that stayed high.',
              },
            ],
          },
          {
            prompt: `Paying off a balance at ${percent(apr)} is equivalent to earning what, guaranteed?`,
            options: [
              {
                label: `About ${percent(apr)}, risk-free`,
                correct: true,
                why: 'A dollar that stops being charged 22.9% is worth exactly as much as a dollar earning 22.9% — with no market risk and no tax.',
              },
              { label: 'Nothing — it is not an investment', correct: false, why: 'Avoided interest and earned interest are the same arithmetic.' },
              { label: 'About 7%, like the market', correct: false, why: 'The rate you avoid is the card\'s rate, not the market\'s.' },
            ],
            transfer: true,
          },
          {
            prompt: 'A 0% balance transfer offer runs for 18 months with a 3% fee. When is that worth taking?',
            options: [
              {
                label: 'When you will clear most of it inside the 18 months',
                correct: true,
                why: 'The 3% fee buys 18 months without interest. That is a good trade only if the balance is actually gone or nearly gone before the rate reverts.',
              },
              {
                label: 'Always — 0% is lower than 22.9%',
                correct: false,
                why: 'The reverted rate applies to whatever is left, and the fee is paid up front regardless.',
              },
            ],
            transfer: true,
          },
        ],
      },
      {
        kind: 'rule',
        name: 'The minimum is the lender\'s plan, not yours',
        statement:
          'Treat the minimum as the number that keeps the account open, and decide your own payment separately.',
        example: `Here, ${money(minimum)} means ${
          minPlan.neverPaysOff ? 'never' : duration(minPlan.months)
        }. ${money(doubled)} means ${duration(doublePlan.months)}. Nothing else changed.`,
      },
      {
        kind: 'action',
        when: 'your next statement arrives',
        then: `set the autopay amount to a fixed figure you choose, rather than to "minimum due" — a fixed payment keeps shrinking the balance instead of shrinking itself`,
        options: [
          { label: 'Changing it this week', commits: true },
          { label: 'When the statement lands', commits: true },
          { label: 'Not now', commits: false },
        ],
        worth: () => minPlan.totalInterest - doublePlan.totalInterest,
      },
    ]
  },
}

/* ========================================================================== *
 * 4. The match comes first
 * ========================================================================== */

const matchComesFirst: Lesson = {
  id: 'match-comes-first',
  title: 'The match comes first',
  competence: 'You can rank a match against a credit card and get it right',
  misconception:
    'Kill the highest rate first is a good rule that has exactly one exception, and it is the one most people get wrong.',
  citation: {
    text: 'Vanguard, How America Saves 2026: the most common formula is 50% of contributions up to 6% of pay, and average employer contributions have reached about 4.7% of pay.',
    url: 'https://workplace.vanguard.com/insights-and-research/report/how-america-saves-2026.html',
  },
  triggers: ['has-employer-plan', 'new-job', 'open-enrollment'],
  jurisdiction: 'US',
  minutes: 3,
  concepts: ['employer-match', 'ordering-decisions'],
  build: (profile) => {
    const p = withDefaults(profile)
    const match = employerMatch({
      salary: p.income,
      employeeRate: 0.06,
      matchRate: 0.5,
      matchLimit: 0.06,
    })
    const unclaimed = employerMatch({
      salary: p.income,
      employeeRate: 0,
      matchRate: 0.5,
      matchLimit: 0.06,
    })

    return [
      {
        kind: 'anchor',
        body: `On ${money(p.income)}, a typical match — 50 cents per dollar, up to 6% of pay — is worth ${money(
          match.employerContribution,
        )} a year. You also have a card at ${percent(p.debtApr)}.`,
        note: 'Assume you can only fund one of them this month.',
      },
      {
        kind: 'probe',
        mode: 'choice',
        question: 'Which dollar should go first?',
        options: [
          {
            label: 'The credit card — 22.9% is the highest rate in my life',
            correct: false,
            misconception:
              'The highest-rate rule is right almost everywhere. The match is the exception because its return is not a rate at all.',
          },
          { label: 'The 401(k), up to the match', correct: true },
          { label: 'Split it evenly to make progress on both', correct: false, misconception: 'Splitting feels balanced and costs you the match.' },
        ],
        because:
          'Both answers are defensible in most situations. This is the one place where the intuitive ordering is mathematically wrong.',
      },
      {
        kind: 'reveal',
        headline: '50% the moment it lands',
        body: `A 50% match returns 50 cents per dollar immediately — before any market return, in the first second. No consumer debt in existence charges 50% a year. Skipping it to pay down a ${percent(
          p.debtApr,
        )} card trades a 50% return for a ${percent(p.debtApr)} one.`,
        contrast: [
          { label: 'Match, on the first dollar', value: '+50%', tone: 'growth' },
          { label: 'Paying the card instead', value: `+${percent(p.debtApr)}`, tone: 'neutral' },
        ],
      },
      {
        kind: 'mechanism',
        sentence:
          'A match is not a rate of return over a year — it is a one-time multiplication of the dollar at the instant you contribute it.',
        detail:
          'Interest rates compete on time. A match does not: the 50% arrives whole, on day one, and then compounds on top. That is why it outranks even very expensive debt.',
        visual: 'none',
      },
      {
        kind: 'worked',
        setup: `Contributing 6% of ${money(p.income)} for one year.`,
        steps: [
          { label: 'You contribute', value: money(match.employeeContribution) },
          { label: 'Employer adds', value: money(match.employerContribution) },
          {
            label: 'Total in the account',
            value: money(match.employeeContribution + match.employerContribution),
            blankable: true,
            answer: Math.round(match.employeeContribution + match.employerContribution),
            unit: 'usd',
          },
          { label: 'Return, before the market does anything', value: percent(match.instantReturn) },
        ],
        conclusion: `Contributing nothing leaves ${money(
          unclaimed.unclaimed,
        )} of your own compensation unclaimed each year. It is part of your pay that requires an action to collect.`,
      },
      {
        kind: 'practice',
        items: [
          {
            prompt: 'Your employer matches dollar-for-dollar up to 3%. Where do the first dollars go?',
            options: [
              { label: 'Up to 3%, into the plan', correct: true, why: 'Dollar-for-dollar is a 100% return on contact. Nothing beats it.' },
              { label: 'The credit card, then the plan', correct: false, why: 'A 100% instant return outranks every consumer rate there is.' },
            ],
          },
          {
            prompt: 'Your employer offers no match at all. Now where do the dollars go?',
            options: [
              {
                label: 'The high-interest debt',
                correct: true,
                why: 'With no match, the exception disappears and the highest-rate rule takes over. This is why "always max the 401(k)" is bad advice — the match is what made it right.',
              },
              { label: 'Still the 401(k) — retirement accounts always win', correct: false, why: 'Without a match there is no instant return to beat the card rate.' },
            ],
            transfer: true,
          },
          {
            prompt: 'You are matched, but the match vests after three years and you expect to leave in one.',
            options: [
              {
                label: 'The match is not yet yours — weigh it accordingly',
                correct: true,
                why: 'An unvested match can be clawed back when you leave. Vesting schedules are the one thing that legitimately changes this answer.',
              },
              { label: 'Vesting does not affect the maths', correct: false, why: 'A 50% return you forfeit on departure is not a 50% return.' },
            ],
            transfer: true,
          },
        ],
      },
      {
        kind: 'rule',
        name: 'Match, then rate',
        statement:
          'Take the full employer match first. After that, and only after that, work strictly down from the highest interest rate.',
        example: `For you: the first ${money(
          match.employeeContribution,
        )} a year goes to the plan to collect ${money(
          match.employerContribution,
        )}. Every dollar after that goes to the ${percent(p.debtApr)} card until it is gone.`,
      },
      {
        kind: 'action',
        when: 'you next open your payroll or benefits portal',
        then: 'check one number — the percentage you contribute — against one other number, the percentage your employer matches up to',
        options: [
          { label: "I'll check this week", commits: true },
          { label: 'At open enrollment', commits: true },
          { label: "I don't have an employer plan", commits: false },
        ],
        worth: () => unclaimed.unclaimed,
      },
    ]
  },
}

/* ========================================================================== *
 * 5. Where your cash is sitting
 * ========================================================================== */

const whereCashSits: Lesson = {
  id: 'where-cash-sits',
  title: 'Where your cash is sitting',
  competence: 'You can tell whether your savings account is doing anything',
  misconception:
    '"Savings account" sounds like a category. It is a product, and the difference between two of them is about ten-fold.',
  triggers: ['always', 'cash-heavy'],
  jurisdiction: 'US',
  minutes: 3,
  concepts: ['cash-yield', 'apy'],
  build: (profile) => {
    const p = withDefaults(profile)
    const cash = p.invested > 0 ? Math.min(p.invested, 20_000) : 8_000
    const national = FACTS.hysaNationalAverage
    const competitive = FACTS.hysaCompetitive

    const atNational = cash * national.value
    const atCompetitive = cash * competitive.value

    return [
      {
        kind: 'anchor',
        body: `Say you have ${money(cash)} sitting in cash — an emergency fund, or money waiting for a decision.`,
        note: `The FDIC national average savings rate was ${percent(
          national.value,
          2,
        )} as of ${national.asOf}. Competitive accounts were paying around ${percent(
          competitive.value,
        )}.`,
      },
      {
        kind: 'probe',
        mode: 'estimate',
        question: `How much more interest does that ${money(
          cash,
        )} earn in a year at a competitive rate versus the national average?`,
        answer: atCompetitive - atNational,
        unit: 'usd',
        tolerance: 0.2,
        min: 0,
        max: (atCompetitive - atNational) * 2.5,
        start: (atCompetitive - atNational) * 0.25,
        because: 'The gap is not a few percent. It is roughly ten times the rate.',
      },
      {
        kind: 'reveal',
        headline: `${money(atCompetitive - atNational)} a year`,
        body: `${money(atNational)} at the national average, ${money(
          atCompetitive,
        )} at a competitive rate. Same money, same access, same federal insurance. The only difference is which institution is holding it.`,
        contrast: [
          { label: `National average, ${percent(national.value, 2)}`, value: money(atNational), tone: 'drag' },
          { label: `Competitive, around ${percent(competitive.value)}`, value: money(atCompetitive), tone: 'growth' },
        ],
      },
      {
        kind: 'mechanism',
        sentence:
          'Banks set the rate on existing accounts, not the market, and most large banks have no reason to raise yours.',
        detail:
          'Your deposit is their cheapest funding. A rate change requires nothing from you and costs them directly, so it does not happen on its own.',
        visual: 'none',
      },
      {
        kind: 'worked',
        setup: `${money(cash)}, over five years, with nothing added.`,
        steps: [
          { label: `At ${percent(national.value, 2)}`, value: money(cash * Math.pow(1 + national.value, 5)) },
          {
            label: `At ${percent(competitive.value)}`,
            value: money(cash * Math.pow(1 + competitive.value, 5)),
            blankable: true,
            answer: Math.round(cash * Math.pow(1 + competitive.value, 5)),
            unit: 'usd',
          },
          {
            label: 'Difference',
            value: money(
              cash * Math.pow(1 + competitive.value, 5) - cash * Math.pow(1 + national.value, 5),
            ),
          },
        ],
        conclusion:
          'This is the rare case with no trade-off to weigh: no extra risk, no lock-up, no market exposure. It is an afternoon of paperwork.',
      },
      {
        kind: 'practice',
        items: [
          {
            prompt: 'Why is this not the same decision as "should I invest this money"?',
            options: [
              {
                label: 'Because both options are insured cash — the risk is identical',
                correct: true,
                why: 'Moving from one insured savings account to another changes nothing about risk or access. Investing changes both.',
              },
              { label: 'Because a higher rate always means higher risk', correct: false, why: 'True across asset classes, not across identical insured deposit accounts.' },
            ],
          },
          {
            prompt: 'An account advertises 4.5% APY "for the first three months, then 0.40%".',
            options: [
              {
                label: 'The number that matters is the rate after the promo',
                correct: true,
                why: 'Teaser rates are priced on the assumption you will not move again. Blended over a year, this is barely above the national average.',
              },
              { label: '4.5% is 4.5% — take it', correct: false, why: 'Three months of 4.5% and nine of 0.40% averages near 1.4%.' },
            ],
            transfer: true,
          },
          {
            prompt: 'Where does this NOT apply?',
            options: [
              {
                label: 'Money you need this week for rent',
                correct: true,
                why: 'A transfer takes days to settle. Chasing yield on money with a deadline attached is how you miss the deadline.',
              },
              { label: 'An emergency fund you might need in six months', correct: false, why: 'Six months is plenty of runway; savings accounts stay liquid.' },
            ],
            transfer: true,
          },
        ],
      },
      {
        kind: 'rule',
        name: 'Know your rate, or you do not have one',
        statement:
          'If you cannot name the interest rate on your savings account, it is almost certainly near zero. Check it once a year.',
        example: `The national average was ${percent(national.value, 2)} on ${national.asOf}. If that is roughly what you are getting, the gap in this lesson is yours to collect.`,
      },
      {
        kind: 'action',
        when: 'you next open your banking app',
        then: 'find the interest rate on your main savings account and write the number down — deciding what to do about it is a separate question for a separate day',
        options: [
          { label: 'Looking it up now', commits: true },
          { label: 'Remind me tomorrow', commits: true },
          { label: 'I already know it', commits: false },
        ],
        worth: () => atCompetitive - atNational,
      },
    ]
  },
}

/* ========================================================================== *
 * 6. A third of a percent
 * ========================================================================== */

const smallFeesCompound: Lesson = {
  id: 'small-fees-compound',
  title: 'A third of a percent',
  competence: 'You can convert a fee into what it costs you at the end',
  misconception:
    'A 0.4% fee sounds like it costs 0.4%. Over a career it costs a visible fraction of the whole outcome, because the fee compounds exactly as the returns do.',
  citation: {
    text: 'ICI, Trends in the Expenses and Fees of Funds 2025: asset-weighted average expense ratios were 0.05% for index equity mutual funds and 0.40% for equity mutual funds overall.',
    url: 'https://www.ici.org/files/2026/per32-01.pdf',
  },
  triggers: ['always', 'open-enrollment'],
  jurisdiction: 'any',
  minutes: 3,
  concepts: ['expense-ratio', 'fee-drag'],
  build: (profile) => {
    const p = withDefaults(profile)
    const years = Math.max(20, p.targetAge - p.age)
    const base = { principal: p.invested, monthly: p.monthly, annualRate: p.rate, years }

    const cheap = FACTS.expenseRatioIndexEquity.value
    const average = FACTS.expenseRatioAllEquity.value
    const drag = feeDrag(base, average - cheap)

    return [
      {
        kind: 'anchor',
        body: `Two funds holding the same thing. One charges ${percent(
          cheap,
          2,
        )} a year, the other ${percent(average, 2)}. A difference of ${percent(average - cheap, 2)}.`,
        note: `Those are the real 2025 asset-weighted averages for index equity funds and for equity funds overall, not a strawman comparison.`,
      },
      {
        kind: 'probe',
        mode: 'estimate',
        question: `Over ${years} years on your contributions, what share of your final balance does that ${percent(
          average - cheap,
          2,
        )} difference take?`,
        answer: drag.shareOfOutcome * 100,
        unit: 'percent',
        tolerance: 0.3,
        min: 0,
        max: 30,
        start: 1,
        because:
          'Almost everyone answers with roughly the fee itself. The fee is charged annually, on a balance that keeps growing.',
      },
      {
        kind: 'reveal',
        headline: `${percent(drag.shareOfOutcome)} of the final balance`,
        body: `${money(drag.lost)} over ${years} years. The fee is small; the thing it is charged against is not, and it is charged again every year on a larger number.`,
        contrast: [
          { label: `At ${percent(cheap, 2)}`, value: moneyCompact(drag.withoutFee), tone: 'growth' },
          { label: `At ${percent(average, 2)}`, value: moneyCompact(drag.withFee), tone: 'drag' },
        ],
      },
      {
        kind: 'mechanism',
        sentence:
          'A fee is subtracted from the return before it compounds, so you lose the fee and every future dollar the fee would have earned.',
        detail:
          'Paying 0.4% instead of 0.05% does not cost 0.35% of your money. It costs 0.35% of a growing balance, every year, for the whole period — and all the growth those amounts would have produced.',
        visual: 'split',
      },
      {
        kind: 'worked',
        setup: 'The same portfolio, run twice.',
        steps: [
          { label: 'Assumed return', value: percent(p.rate) },
          { label: 'Return after the cheap fund\'s fee', value: percent(p.rate - cheap) },
          {
            label: 'Return after the average fund\'s fee',
            value: percent(p.rate - average),
            blankable: true,
            answer: Number(((p.rate - average) * 100).toFixed(2)),
            unit: 'percent',
          },
          { label: `Gap after ${years} years`, value: money(drag.lost) },
        ],
        conclusion:
          'The expense ratio is the only variable in investing that is known in advance, guaranteed, and entirely under your control. Returns are none of those things.',
      },
      {
        kind: 'practice',
        items: [
          {
            prompt: 'A fund returned 9% last year and charges 0.85%. Another returned 7% and charges 0.04%. Which tells you more about the future?',
            options: [
              {
                label: 'The fee',
                correct: true,
                why: 'Last year\'s return has almost no predictive power. The fee will be charged again next year with certainty.',
              },
              { label: 'The return — 9% beat 7%', correct: false, why: 'Past returns do not persist. Fees do.' },
            ],
          },
          {
            prompt: 'An advisor charges 1% of assets annually and picks funds averaging 0.5%.',
            options: [
              {
                label: 'The total drag is 1.5%, not 1%',
                correct: true,
                why: 'Layered fees add. This is the most common place people underestimate what they are paying, because the two are disclosed in different documents.',
              },
              { label: 'The 1% includes the fund fees', correct: false, why: 'An advisory fee is charged on top of whatever the underlying funds charge.' },
            ],
            transfer: true,
          },
          {
            prompt: 'Where is a higher fee sometimes defensible?',
            options: [
              {
                label: 'When it buys something the cheap option genuinely does not do',
                correct: true,
                why: 'Fees are a price, not a sin. The question is always what you get. Most of the time, for a broad index, the answer is nothing.',
              },
              { label: 'Never — always pick the lowest fee', correct: false, why: 'A slightly pricier fund that you will actually hold beats a cheaper one you abandon.' },
            ],
            transfer: true,
          },
        ],
      },
      {
        kind: 'rule',
        name: 'Six-tenths of your years',
        statement:
          'Multiply the fee by six-tenths of the years you will be invested. That is roughly the share of your final balance it takes.',
        // The popular "multiply by 25" version is only right at a full 40-year
        // horizon — it overstates the cost by more than double at 20 years. The
        // 0.6-per-year form tracks the real figure across every horizon that matters.
        example: `${percent(average - cheap, 2)} × (0.6 × ${years}) ≈ ${percent(
          (average - cheap) * 0.6 * years,
        )}. The exact figure for your numbers is ${percent(
          drag.shareOfOutcome,
        )} — close enough to do in your head.`,
      },
      {
        kind: 'action',
        when: 'you next log into your retirement account',
        then: 'find the expense ratio of your largest holding — it is usually one tap into the fund detail, listed as "gross expense ratio"',
        options: [
          { label: 'Checking this week', commits: true },
          { label: 'Remind me at open enrollment', commits: true },
          { label: 'Not now', commits: false },
        ],
        worth: () => drag.lost,
      },
    ]
  },
}

/* ========================================================================== *
 * Registry
 * ========================================================================== */

export const LESSONS: Lesson[] = [
  growthIsNotALine,
  startingToday,
  compoundingInReverse,
  matchComesFirst,
  whereCashSits,
  smallFeesCompound,
]

export function lessonById(id: string): Lesson | undefined {
  return LESSONS.find((l) => l.id === id)
}

/**
 * Orders lessons by how relevant they are *right now*.
 *
 * Relevance, not sequence. A lesson whose trigger has fired outranks the next one
 * in a nominal order, because the trigger is the only thing the evidence says
 * makes education stick.
 */
export function rankLessons(
  profile: Profile,
  completed: Record<string, unknown>,
  jurisdiction: 'US' | 'other' = 'US',
): Lesson[] {
  const hasDebt = (profile.debtBalance ?? 0) > 0
  const hasIncome = (profile.income ?? 0) > 0

  const score = (l: Lesson): number => {
    let s = 0
    if (l.triggers.includes('has-debt') && hasDebt) s += 10
    if (l.triggers.includes('has-employer-plan') && hasIncome) s += 8
    if (l.triggers.includes('always')) s += 3
    // A completed lesson drops to the bottom but stays reachable.
    if (completed[l.id]) s -= 100
    return s
  }

  return LESSONS.filter((l) => l.jurisdiction === 'any' || l.jurisdiction === jurisdiction).sort(
    (a, b) => score(b) - score(a),
  )
}
