import type { Call } from './types'

/**
 * The ten calls, in play order.
 *
 * Authoring rules that are not obvious from the type:
 *
 *  - `title` is a SITUATION, never a question, and never contains the answer. It
 *    renders as three lines of 34px Archivo Black, which is why twelve words is a
 *    hard ceiling rather than a style note.
 *  - `variable.start` is deliberately the *wrong* answer, and usually the answer
 *    the real world defaults to (3% auto-enrolment, $0 counter-offer, the whole
 *    repair quote). If the control started on the optimum the gesture would teach
 *    nothing, because the player would never have to move it.
 *  - `crowd` is a real-world prior, not a bell curve. Its shape is the second half
 *    of the lesson: the gap between where the crowd piles up and where the optimum
 *    sits is the thing worth screenshotting. Entries are weights, not percentages
 *    — only their relative size matters to the histogram.
 *  - Money variables carry an empty `unit`. The unit renders immediately after the
 *    figure in the 64px readout, so a dollar amount would come out as "3200$";
 *    the caption underneath names the currency instead.
 */
export const CALLS: Call[] = [
  {
    id: 1,
    title: 'Your boss pays 50c for every dollar you save.',
    domain: 'retirement',
    variable: {
      key: 'contribution',
      min: 0,
      max: 15,
      step: 1,
      unit: '%',
      label: 'of every paycheck',
      start: 3,
    },
    fixed: [
      { k: 'PAY', v: '{{salary}}' },
      { k: 'MATCH', v: '50% UP TO 6% OF PAY' },
    ],
    compute: 'employerMatch',
    optimal: 6,
    rule: 'Take the match before anything else.',
    // Vanguard's How America Saves shows deferral rates are not a distribution at
    // all — they are a set of spikes at whatever number a form put in front of
    // someone. The tallest is the auto-enrolment default (3%), which is also the
    // single most expensive number in this file: it stops half a step short of the
    // match. Second spike at 0% (the ~15% who never enrolled), third at the 6% cap
    // (people who were actually told), plus the usual round-number pull at 5% and
    // 10% and a small max-out tail at 15%.
    crowd: [14, 2, 3, 22, 6, 9, 13, 2, 5, 1, 12, 1, 3, 1, 1, 5],
    tomorrow: 'You have $500 spare. Both cards want it.',
    assumptions:
      '50% of the first 6% of pay, the most common formula, grown at 7% a year to 65 — the match is certain, the 7% is not.',
  },

  {
    id: 2,
    title: 'You have $500 spare. Both cards want it.',
    domain: 'debt',
    variable: {
      key: 'toLowerRate',
      min: 0,
      max: 500,
      step: 25,
      unit: '',
      label: 'dollars onto the 17.99% card',
      start: 250,
    },
    fixed: [
      { k: 'SPARE CASH', v: '$500, ONE TIME' },
      { k: 'THE CARDS', v: '$2,400 AT 24.99% · $900 AT 17.99%' },
    ],
    compute: 'debtSplit',
    optimal: 0,
    rule: 'Kill the highest rate first. Math beats momentum.',
    // Two instincts fight here and neither is the optimum. The snowball instinct
    // sends the whole $500 at the smaller, lower-rate card to feel a balance hit
    // zero, so there is a wall at $500. The fairness instinct splits it down the
    // middle, so there is an equal wall at $250. Everything between is round-number
    // noise at the hundreds. Only the 21% at $0 are doing arithmetic — which is why
    // the optimum sits at the very end of the track the player is least likely to
    // drag toward.
    crowd: [21, 1, 2, 1, 5, 1, 2, 1, 4, 1, 24, 1, 4, 1, 2, 1, 4, 1, 2, 1, 20],
    tomorrow: 'The car dies on a Tuesday. Rent is due Friday.',
    assumptions:
      'APRs billed monthly at APR/12 the way lenders quote them, minimums continuing on both cards — the interest avoided is arithmetic, not a projection.',
  },

  {
    id: 3,
    title: 'The car dies on a Tuesday. Rent is due Friday.',
    domain: 'savings',
    variable: {
      key: 'months',
      min: 0,
      max: 12,
      step: 1,
      unit: 'mo',
      label: 'of costs held in cash',
      start: 1,
    },
    fixed: [
      { k: 'PAY', v: '{{salary}}' },
      { k: 'MONTHLY COSTS', v: '55% OF NET PAY' },
    ],
    compute: 'emergencyFund',
    optimal: { min: 3, max: 6 },
    rule: 'Bank three months of costs, not income.',
    // The Fed's SHED survey is blunt: a large minority cannot cover a $400 shock at
    // all, so the mass is jammed against zero and decays fast. The two bumps at 3
    // and 6 are pure folklore transmission — those are the only two numbers anyone
    // has ever been told — and the small lift at 12 is the over-corrector who keeps
    // a year of spending in a 0.38% savings account and calls it safe. Both tails
    // are wrong; only one of them feels wrong.
    crowd: [26, 15, 10, 12, 5, 4, 10, 3, 3, 2, 3, 1, 6],
    tomorrow: 'The sofa is zero percent. The paperwork is not.',
    assumptions:
      'Cash earning a 4.0% high-yield APY against 7% a year invested, over the months you choose — neither rate is guaranteed.',
  },

  {
    id: 4,
    title: 'The sofa is zero percent. The paperwork is not.',
    domain: 'credit',
    variable: {
      key: 'payoffMonths',
      min: 1,
      max: 24,
      step: 1,
      unit: 'mo',
      label: 'to clear the balance',
      start: 18,
    },
    fixed: [
      { k: 'SOFA', v: '$2,400 ON STORE CREDIT' },
      { k: 'IF UNPAID', v: '29.99% BACK TO DAY ONE' },
    ],
    compute: 'promoDeadline',
    optimal: { min: 1, max: 12 },
    rule: 'Treat 0% as a deadline, not a discount.',
    // Store cards set the minimum payment so that paying it exactly runs past the
    // promo — that is the product, not an accident — so the biggest spike by far is
    // at 24 months, the people who will simply pay what the statement asks. The
    // spikes at 12 and 18 are the players who divide the price by a term they
    // half-remember. Almost nobody clusters at the fast end, which is the whole
    // reason deferred interest is profitable.
    crowd: [3, 1, 1, 2, 1, 3, 2, 1, 1, 2, 1, 14, 1, 1, 1, 2, 1, 12, 1, 1, 2, 1, 1, 44],
    tomorrow: 'The recruiter says the number out loud. You say nothing.',
    assumptions:
      '0% for twelve months and then 29.99% charged back to day one, the standard deferred-interest structure — a contract term, not a forecast.',
  },

  {
    id: 5,
    title: 'The recruiter says the number out loud. You say nothing.',
    domain: 'income',
    variable: {
      key: 'counter',
      min: 0,
      max: 20,
      step: 1,
      unit: '%',
      label: 'above their offer',
      start: 0,
    },
    fixed: [
      { k: 'THE OFFER', v: '{{salary}}' },
      { k: 'FUTURE RAISES', v: '3%/YR ON THE NEW BASE' },
    ],
    compute: 'anchorOffer',
    optimal: { min: 6, max: 10 },
    rule: 'Counter. The first number anchors every number after.',
    // Roughly half of people accept the first offer without countering, so the bar
    // at 0 dwarfs everything and is the single most expensive habit in the file:
    // every future raise is a percentage of the number they did not argue with.
    // Everyone who does counter reaches for a round number — 5, 10, 15, 20 — and
    // the spaces between them are nearly empty, which is exactly what an anchoring
    // effect looks like when you plot it.
    crowd: [45, 2, 2, 2, 3, 10, 2, 2, 3, 1, 11, 1, 1, 1, 1, 5, 1, 1, 1, 1, 4],
    tomorrow: 'Your plan will tax you once. You choose when.',
    assumptions:
      '3% raises a year on whatever base you agree to, compounded to 65 with savings at 7% — the raises are an assumption, not a promise.',
  },

  {
    id: 6,
    title: 'Your plan will tax you once. You choose when.',
    domain: 'tax',
    variable: {
      key: 'toRoth',
      min: 0,
      max: 100,
      step: 5,
      unit: '%',
      label: 'of each contribution to Roth',
      start: 0,
    },
    fixed: [
      { k: 'PAY', v: '{{salary}}' },
      { k: 'AT RETIREMENT', v: 'ASSUMED 22% BRACKET' },
    ],
    compute: 'rothSplit',
    optimal: { min: 80, max: 100 },
    rule: 'Go pre-tax when high, Roth when low.',
    // Pre-tax is the default election in most plans and defaults win, so the tallest
    // bar is at 0% Roth. The second tallest is at 100%, because "Roth is for young
    // people" is the one piece of tax folklore that actually circulates. The bump at
    // 50% is the hedge — people splitting because they cannot forecast a bracket
    // forty years out, which is a more defensible answer than its popularity
    // suggests and the reason the optimum here is a band, not a point.
    crowd: [38, 2, 2, 1, 2, 6, 1, 1, 1, 1, 14, 1, 1, 1, 1, 4, 1, 1, 1, 1, 19],
    tomorrow: 'Your mechanic leaves a voicemail with a number in it.',
    assumptions:
      '2026 single-filer brackets on your pay today against an assumed 22% in retirement, growing at 7% until 65 — that future bracket is the assumption doing all the work.',
  },

  {
    id: 7,
    title: 'Your mechanic leaves a voicemail with a number in it.',
    domain: 'vehicles',
    variable: {
      key: 'repairSpend',
      min: 0,
      max: 4000,
      step: 100,
      unit: '',
      label: 'dollars into the old car',
      start: 3200,
    },
    fixed: [
      { k: 'CAR WORTH', v: '$5,000 PRIVATE PARTY' },
      { k: 'OR REPLACE', v: '$520/MO FOR 60 MONTHS' },
    ],
    compute: 'repairOrReplace',
    optimal: { min: 0, max: 2500 },
    rule: 'Compare the repair to a year of payments.',
    // Anchoring again, in its purest form: the quote is $3,200 and that is where the
    // tallest spike sits, because the number a professional says out loud becomes
    // the question. The second cluster is at $0 — walk away, buy new — and the rest
    // is round-number bargaining at each $500. Very few people land in the band
    // between half the car's value and the quote, which is where the answer lives.
    crowd: [
      10, 1, 1, 2, 1, 6, 1, 1, 1, 1, 9, 1, 1, 1, 1, 5, 1, 1, 1, 1, 7, 1, 1, 1, 1, 4, 1, 1, 1, 1,
      5, 1, 16, 1, 1, 3, 1, 1, 1, 1, 3,
    ],
    tomorrow: 'Two funds own the same stocks and end up different.',
    assumptions:
      'The repair assumed to buy two more years against $520 a month for 60 months at 9.1%, the difference invested at 7% — the two years is an estimate.',
  },

  {
    id: 8,
    title: 'Two funds own the same stocks and end up different.',
    domain: 'investing',
    variable: {
      key: 'expenseRatio',
      min: 3,
      max: 150,
      step: 3,
      unit: 'bps',
      label: 'a year in fees',
      start: 54,
    },
    fixed: [
      { k: 'CONTRIBUTION', v: '10% OF {{salary}}' },
      { k: 'RETURN', v: '7% A YEAR BEFORE FEES' },
    ],
    compute: 'feeDragCall',
    optimal: 3,
    rule: 'Pay basis points, not percent. Fees compound too.',
    // Counted by person rather than by dollar, which is the distinction that makes
    // this call work. Asset-weighted, index money dominates and the average index
    // equity fund is about 5bps (ICI 2025) — but most individual savers are in a
    // plan lineup, so the mode is the broad 30-70bps shoulder around the 40bps
    // all-equity average, with a bump just under 100bps where retail active funds
    // price themselves. The small block at the left edge is the index crowd. The
    // player is almost certainly standing on the shoulder; the optimum is off the
    // far end of it, and the distance between those two points is the lesson.
    crowd: [
      6, 5, 4, 3, 3, 3, 3, 4, 4, 6, 5, 6, 6, 7, 7, 6, 6, 7, 5, 6, 4, 4, 3, 3, 3, 2, 2, 2, 2, 3,
      2, 2, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
    ],
    tomorrow: 'The mortgage payment matches your rent. Everyone says buy.',
    assumptions:
      '10% of pay invested to 65 at 7% a year before fees — the fee is contractual, the 7% is not.',
  },

  {
    id: 9,
    title: 'The mortgage payment matches your rent. Everyone says buy.',
    domain: 'housing',
    variable: {
      key: 'years',
      min: 0,
      max: 15,
      step: 1,
      unit: 'yr',
      label: 'before you move again',
      start: 2,
    },
    fixed: [
      { k: 'THE HOUSE', v: '$420,000 AT 6.4%' },
      { k: 'ROUND TRIP', v: '9% IN CLOSING AND SELLING' },
    ],
    compute: 'rentVsBuy',
    optimal: { min: 5, max: 15 },
    rule: 'Rent under five years. Buying costs more.',
    // Buyers systematically over-forecast how long they will stay, so the mass sits
    // high: spikes at 5 and 10 (the two numbers anyone says out loud) and a wall at
    // 15 meaning "forever". The honest answer for most people is the sparse left
    // half — jobs move, relationships change, and the transaction costs do not care
    // about intentions. This is the one call where the crowd and the optimum mostly
    // agree, and they agree for the wrong reason.
    crowd: [2, 2, 4, 9, 5, 18, 5, 4, 6, 3, 17, 1, 2, 2, 2, 18],
    tomorrow: 'You went to cash for a month. Stocks did not wait.',
    assumptions:
      '$420,000 at 6.4% fixed, 9% round-trip transaction costs, rent rising 3% a year and the down payment otherwise invested at 7% — prices assumed flat in real terms.',
  },

  {
    id: 10,
    title: 'You went to cash for a month. Stocks did not wait.',
    domain: 'investing',
    variable: {
      key: 'daysMissed',
      min: 0,
      max: 30,
      step: 1,
      unit: '',
      label: 'of the best days, missed',
      start: 4,
    },
    fixed: [
      { k: 'INVESTED', v: '$10,000 ONCE' },
      { k: 'HELD', v: '20 YEARS AT 7% A YEAR' },
    ],
    compute: 'timingMarket',
    optimal: 0,
    rule: 'Stay in. Missing ten days halves your return.',
    // Almost nobody believes a handful of days can matter, so the hump sits at three
    // to six — "I was only out for a couple of weeks". The spike at 10 is the famous
    // statistic leaking back in, and the block at 0 is the minority who have already
    // been burned once. The long flat tail past 15 is people who sat out an entire
    // cycle and have stopped counting.
    crowd: [
      16, 4, 7, 9, 8, 9, 6, 4, 4, 2, 9, 1, 1, 1, 1, 3, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      3,
    ],
    tomorrow: 'Your boss pays 50c for every dollar you save.',
    assumptions:
      '$10,000 held 20 years at 7% a year, each missed day priced at a top-30 single-session gain — an illustration of a real pattern, not a forecast.',
  },
]

export const callCount = CALLS.length

const BY_ID = new Map(CALLS.map((c) => [c.id, c]))

/** Undefined rather than throwing: a deep link to /99 is a 404 screen, not a crash. */
export function callById(id: number): Call | undefined {
  return BY_ID.get(id)
}
