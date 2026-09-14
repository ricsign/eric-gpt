import type { Call } from './types'

/**
 * The ten calls, in play order.
 *
 * Authoring rules that are not obvious from the type:
 *
 *  - `title` is a SITUATION, never a question, and never contains the answer.
 *    It sets in Archivo Black at --t-display and must wrap to THREE LINES at
 *    402px. Word count is not the real constraint — line packing is: "The
 *    payment matches rent. Everyone says buy." wraps to four lines and
 *    "Everyone says buy. The payment matches rent." wraps to three, at the same
 *    44 characters. Every title here was measured in the browser, and a new one
 *    has to be (`node scripts/shoot.mjs`), because no unit test can do it.
 *  - `fixed` values set at 22px Courier Prime in a half-width tile: eleven
 *    characters to the line. Every value here fits on two, which is what keeps
 *    all ten call screens exactly the same height — 194px of slack under the
 *    hero at 402x874 — so the control and the lock button land under the same
 *    thumb every day. A third line eats that slack, and .call clips rather than
 *    scrolls. Long scene-setting belongs in `assumptions`, which is 11px fine
 *    print on the outcome screen.
 *  - `variable.start` is deliberately the *wrong* answer, and usually the answer
 *    the real world defaults to (3% auto-enrolment, $0 counter-offer, the whole
 *    repair quote). If the control started on the optimum the gesture would
 *    teach nothing, because the player would never have to move it.
 *  - `optimal` is a band only where several answers really are equally right.
 *    Every value inside it has to beat the position the player started from, or
 *    the app is congratulating someone for dragging the wrong way — registry
 *    .test.ts enforces exactly that, and it is the reason call 7's band starts
 *    at $2,000 rather than at $0.
 *  - `crowd` is a real-world prior, not a bell curve. Its shape is the second
 *    half of the lesson: the gap between where the crowd piles up and where the
 *    optimum sits is the thing worth screenshotting. Entries are weights, not
 *    percentages — only their relative size matters to the histogram.
 *  - Every figure in `assumptions` and in `variable.label` must be a number
 *    compute.ts actually models. The prose is the one place where a scenario
 *    can rot unnoticed, because nothing renders it next to the maths.
 *
 * Money variables carry an empty `unit`: the unit renders immediately after the
 * figure in the 64px readout, so a dollar amount would come out as "3200$", and
 * the caption underneath names the currency instead.
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
    // 6% is where the employer's money stops, not where saving stops being
    // wise. A player who drags to 12% has more at 65, not less, so calling
    // that "overshot" would be the app telling a plain lie to the one person
    // who did the best thing on the screen. The lesson lives entirely on the
    // left edge: below 6 you are handing back pay you were offered.
    optimal: { min: 6, max: 15 },
    rule: 'Take the match before anything else.',
    // Vanguard's How America Saves shows deferral rates are not a distribution at
    // all — they are a set of spikes at whatever number a form put in front of
    // someone. The tallest is the auto-enrolment default (3%), which is also the
    // single most expensive number in this file: it stops half a step short of the
    // match. Second spike at 0% (the ~15% who never enrolled), third at the 6% cap
    // (people who were actually told), plus the usual round-number pull at 5% and
    // 10% and a small max-out tail at 15%.
    crowd: [14, 2, 3, 22, 6, 9, 13, 2, 5, 1, 12, 1, 3, 1, 1, 5],
    tomorrow: '$500 a month. Both cards want it.',
    assumptions:
      '50% of the first 6% of pay, the most common formula, with both streams invested at 7% a year to 65 — the match is certain, the 7% is not.',
  },

  {
    id: 2,
    title: '$500 a month. Both cards want it.',
    domain: 'debt',
    variable: {
      key: 'toLowerRate',
      min: 0,
      max: 500,
      step: 25,
      unit: '',
      label: 'dollars onto the 11.99% card',
      start: 250,
    },
    // The $500 is in the headline, so both tiles can carry a whole card each
    // rather than cramming two balances and two rates into one.
    fixed: [
      { k: 'BIG CARD', v: '$4,200 AT 24.99%' },
      { k: 'SMALL CARD', v: '$2,800 AT 11.99%' },
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
      '$500 every month until both cards clear, no minimum payments so the split is the only variable, and interest billed at APR/12 the way lenders quote it.',
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
      { k: 'MONTHLY COSTS', v: '50% OF TAKE-HOME' },
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
    tomorrow: 'The sofa is zero percent. Until it is not.',
    // The shock is a model, not a statistic, and it is the half of this call that
    // decides the answer — so it is stated rather than buried.
    assumptions:
      'Costs are half of take-home after federal tax and FICA. Cash earns 4% against 7% invested, and a shock — one year in three, four months of costs on average — is borrowed at 24.99%.',
  },

  {
    id: 4,
    title: 'The sofa is zero percent. Until it is not.',
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
    // The length of the promo window is deliberately NOT on a tile. It is the
    // answer, and the player is meant to find it by dragging until the deferred
    // interest detonates — which is exactly how the offer works in a store.
    fixed: [
      { k: 'STORE CREDIT', v: '$3,000' },
      { k: 'IF UNPAID', v: '26.99% FROM DAY ONE' },
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
    tomorrow: 'The recruiter says a number. You say nothing.',
    assumptions:
      '0% until the promotional window closes, then 26.99% charged back to day one on the whole purchase — a contract term, not a forecast. Cash not yet paid earns 4% meanwhile.',
  },

  {
    id: 5,
    title: 'The recruiter says a number. You say nothing.',
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
      { k: 'RAISES ON BASE', v: '3% A YEAR' },
    ],
    compute: 'anchorOffer',
    // Not the literal 8%: a one-in-twenty-one exact match on a salary
    // negotiation would be false precision, and the model's own peak drifts
    // with age. Every value in the band is within a few percent of the best.
    optimal: { min: 6, max: 10 },
    rule: 'Counter. The first number anchors every number after.',
    // Roughly half of people accept the first offer without countering, so the bar
    // at 0 dwarfs everything and is the single most expensive habit in the file:
    // every future raise is a percentage of the number they did not argue with.
    // Everyone who does counter reaches for a round number — 5, 10, 15, 20 — and
    // the spaces between them are nearly empty, which is exactly what an anchoring
    // effect looks like when you plot it.
    crowd: [45, 2, 2, 2, 3, 10, 2, 2, 3, 1, 11, 1, 1, 1, 1, 5, 1, 1, 1, 1, 4],
    tomorrow: 'The refund felt like a win. It was your own money.',
    assumptions:
      'A point of base compounds through 3% raises and 7% returns to 65. The chance of the offer being pulled is a stylised curve, not a measurement, and it is what stops the dial short of the top.',
  },

  {
    id: 6,
    title: 'The refund felt like a win. It was your own money.',
    domain: 'tax',
    variable: {
      key: 'withheld',
      min: 60,
      max: 180,
      step: 5,
      unit: '%',
      label: 'of the tax you actually owe',
      start: 160,
    },
    fixed: [
      { k: 'PAY', v: '{{salary}}' },
      { k: 'SAFE HARBOR', v: '90% OF THE TAX' },
    ],
    compute: 'withholding',
    optimal: 90,
    rule: 'Owe a little. A refund is a 0% loan.',
    // The model's own bill for a $62,000 single filer is about $5,300, and the
    // IRS's average refund is a little over $3,000 — most of that bill again — so
    // the mass sits well right of par and the dial opens at 160%, not at 100%. The
    // spike at 100% is the small group who actually tune a W-4 and aim to break
    // even; the hump at 110-130 is an ordinary one- or two-thousand-dollar refund;
    // the lump at 150 is the people who treat withholding as a savings account.
    // The thin left tail is those who owe in April, nearly all by accident. Every
    // bar left of 90 is a penalty the crowd does not know it is paying.
    crowd: [1, 1, 1, 2, 2, 3, 3, 4, 8, 5, 7, 9, 10, 8, 7, 6, 5, 4, 5, 3, 3, 2, 2, 1, 3],
    tomorrow: 'The mechanic calls with a number.',
    assumptions:
      'Federal income tax only, single filer, no credits and no other income. Below 90% the safe harbor is gone and the IRS charges interest; above it, the extra withheld simply earns nothing.',
  },

  {
    id: 7,
    title: 'The mechanic calls with a number.',
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
      { k: 'CAR WORTH', v: '$5,000' },
      { k: 'OR REPLACE', v: '$22,000 AT $441/MO' },
    ],
    compute: 'repairOrReplace',
    // $0 is NOT optimal here, and an earlier band that started there said it was:
    // walking away from a $5,000 car to start a $441 payment is the worst play on
    // the whole dial. The band is the plateau around half the car's value, where
    // the repair still buys more months than the payments it defers.
    optimal: { min: 2000, max: 3000 },
    rule: 'Fix it up to half the car, then stop.',
    // Anchoring in its purest form: the quote is the $3,200 the dial opens on, and
    // that is where the tallest spike sits, because the number a professional says
    // out loud becomes the question. The second cluster is at $0 — walk away, buy
    // new — and the rest is round-number bargaining at each $500. Very few people
    // land in the band just under half the car's value, which is where the answer
    // lives.
    crowd: [
      10, 1, 1, 2, 1, 6, 1, 1, 1, 1, 9, 1, 1, 1, 1, 5, 1, 1, 1, 1, 7, 1, 1, 1, 1, 4, 1, 1, 1, 1,
      5, 1, 16, 1, 1, 3, 1, 1, 1, 1, 3,
    ],
    tomorrow: 'Two funds own the same stocks. You pick one.',
    assumptions:
      '$22,000 financed over 60 months is $441 a month. The repair is assumed to buy about two more years, with the first few hundred dollars buying most of them — a judgement, not a measurement.',
  },

  {
    id: 8,
    title: 'Two funds own the same stocks. You pick one.',
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
      { k: 'INVESTED', v: '{{salary}}, THEN 10%/YR' },
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
    tomorrow: 'Everyone says buy. The payment matches rent.',
    assumptions:
      'A year of pay invested now and a tenth of pay a year after that, held to 65 at 7% a year before fees — the fee is contractual, the 7% is not.',
  },

  {
    id: 9,
    title: 'Everyone says buy. The payment matches rent.',
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
      { k: 'THE HOUSE', v: '$360,000 AT 6.5%' },
      { k: 'OR RENT', v: '$2,160/MO' },
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
    tomorrow: 'You sat out a month. Stocks did not wait.',
    assumptions:
      '$360,000 with a fifth down, 30 years at 6.5%, 1.5% a year in tax and upkeep, 3% appreciation, 9% round trip to buy and sell, and the down payment otherwise invested at 7%.',
  },

  {
    id: 10,
    title: 'You sat out a month. Stocks did not wait.',
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
      { k: 'INVESTED', v: '{{salary}}, THEN 10%/YR' },
      { k: 'HELD', v: 'UNTIL 65 AT 7% A YEAR' },
    ],
    compute: 'timingMarket',
    optimal: 0,
    // Not "missing ten days halves your return": the receipt for ten days says
    // you keep 58%, because the model keeps contributing while you sit out, and
    // a rule the player's own screen contradicts is worse than no rule.
    rule: 'Stay in. The best days hide inside the worst.',
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
      'A year of pay invested now and a tenth of pay a year after that, held to 65 at 7%. Missing the best days is modelled as halving the growth at ten of them — the finding, not a forecast.',
  },
]

export const callCount = CALLS.length

const BY_ID = new Map(CALLS.map((c) => [c.id, c]))

/** Undefined rather than throwing: a deep link to /99 is a 404 screen, not a crash. */
export function callById(id: number): Call | undefined {
  return BY_ID.get(id)
}
