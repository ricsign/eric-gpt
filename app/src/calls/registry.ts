import type { Call } from './types'
import { bestRefund } from './compute'

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
    title: 'Your boss adds 50 cents to every dollar you save.',
    question: 'How much of your paycheck should you put in?',
    domain: 'retirement',
    variable: {
      key: 'contribution',
      min: 0,
      max: 15,
      step: 1,
      unit: '%',
      label: 'of your pay, into retirement',
      start: 3,
    },
    fixed: [
      { k: 'YOU EARN A YEAR', v: '{{salary}}' },
      { k: 'YOUR BOSS ADDS', v: 'HALF, UP TO 6% OF PAY' },
    ],
    compute: 'employerMatch',
    // 6% is where the employer's money stops, not where saving stops being
    // wise. A player who drags to 12% has more at 65, not less, so calling
    // that "overshot" would be the app telling a plain lie to the one person
    // who did the best thing on the screen. The lesson lives entirely on the
    // left edge: below 6 you are handing back pay you were offered.
    optimal: { min: 6, max: 15 },
    rule: 'Take the match before anything else.',
    action:
      'Log in to your retirement plan at work and set your contribution to at least the full match. Five minutes.',
    // Vanguard's How America Saves shows deferral rates are not a distribution at
    // all — they are a set of spikes at whatever number a form put in front of
    // someone. The tallest is the auto-enrolment default (3%), which is also the
    // single most expensive number in this file: it stops half a step short of the
    // match. Second spike at 0% (the ~15% who never enrolled), third at the 6% cap
    // (people who were actually told), plus the usual round-number pull at 5% and
    // 10% and a small max-out tail at 15%.
    crowd: [14, 2, 3, 22, 6, 9, 13, 2, 5, 1, 12, 1, 3, 1, 1, 5],
    tomorrow: '$500 to spare. Two credit cards to pay.',
    assumptions:
      'Your boss pays half of what you put in, up to 6% of your pay. That is the most common deal at work. Money grows at 7% a year to 65, which nobody can promise.',
  },

  {
    id: 2,
    title: '$500 to spare. Two credit cards to pay.',
    question: 'How much of the $500 should go to the smaller card?',
    domain: 'debt',
    variable: {
      key: 'toLowerRate',
      min: 0,
      max: 500,
      step: 25,
      unit: '',
      label: 'dollars to the smaller card',
      start: 250,
    },
    // The $500 is in the headline, so both tiles can carry a whole card each
    // rather than cramming two balances and two rates into one.
    fixed: [
      { k: 'THE BIGGER CARD', v: '$4,200 AT 24.99%' },
      { k: 'THE SMALLER CARD', v: '$2,800 AT 11.99%' },
    ],
    compute: 'debtSplit',
    optimal: 0,
    rule: 'Kill the highest rate first. Math beats momentum.',
    action:
      'Find the rate on every card you owe on. Put every spare dollar on the highest one and pay the minimum on the rest.',
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
      'You pay $500 every month until both cards are clear. Minimum payments are left out, so the split is all that changes. Those rates are yearly; interest is charged monthly, the way card issuers do it.',
  },

  {
    id: 3,
    title: 'The car dies on a Tuesday. Rent is due Friday.',
    question: 'How many months of bills should you keep in cash?',
    domain: 'savings',
    variable: {
      key: 'months',
      min: 0,
      max: 12,
      step: 1,
      unit: 'mo',
      label: 'of bills, kept in the bank',
      start: 1,
    },
    fixed: [
      { k: 'YOU EARN A YEAR', v: '{{salary}}' },
      { k: 'BILLS EACH MONTH', v: 'HALF YOUR TAKE-HOME' },
    ],
    compute: 'emergencyFund',
    optimal: { min: 3, max: 6 },
    rule: 'Bank three months of costs, not income.',
    action:
      'Open a savings account paying over 4% and set up a transfer for payday, even if it is $25.',
    // The Fed's SHED survey is blunt: a large minority cannot cover a $400 shock at
    // all, so the mass is jammed against zero and decays fast. The two bumps at 3
    // and 6 are pure folklore transmission — those are the only two numbers anyone
    // has ever been told — and the small lift at 12 is the over-corrector who keeps
    // a year of spending in a 0.38% savings account and calls it safe. Both tails
    // are wrong; only one of them feels wrong.
    crowd: [26, 15, 10, 12, 5, 4, 10, 3, 3, 2, 3, 1, 6],
    tomorrow: 'The sofa is 0% interest. Until it is not.',
    // The shock is a model, not a statistic, and it is the half of this call that
    // decides the answer — so it is stated rather than buried.
    assumptions:
      'Bills are half your take-home after tax, Social Security and Medicare. Cash earns 4%, investing earns 7%. One year in three something breaks, costing four months of bills, put on a 24.99% card.',
  },

  {
    id: 4,
    title: 'The sofa is 0% interest. Until it is not.',
    question: 'How fast do you have to clear it?',
    domain: 'credit',
    variable: {
      key: 'payoffMonths',
      min: 1,
      max: 24,
      step: 1,
      unit: 'mo',
      label: 'to pay it off in full',
      start: 18,
    },
    // The length of the promo window is deliberately NOT on a tile. It is the
    // answer, and the player is meant to find it by dragging until the deferred
    // interest detonates — which is exactly how the offer works in a store.
    fixed: [
      { k: 'ON THE STORE CARD', v: '$3,000' },
      { k: 'MISS THE DEADLINE', v: '26.99% FROM DAY ONE' },
    ],
    compute: 'promoDeadline',
    optimal: { min: 1, max: 12 },
    rule: 'Treat 0% as a deadline, not a discount.',
    action:
      'Find the end date of any 0% deal you have. Divide what you owe by the months left, and set that as a standing payment.',
    // Store cards set the minimum payment so that paying it exactly runs past the
    // promo — that is the product, not an accident — so the biggest spike by far is
    // at 24 months, the people who will simply pay what the statement asks. The
    // spikes at 12 and 18 are the players who divide the price by a term they
    // half-remember. Almost nobody clusters at the fast end, which is the whole
    // reason deferred interest is profitable.
    crowd: [3, 1, 1, 2, 1, 3, 2, 1, 1, 2, 1, 14, 1, 1, 1, 2, 1, 12, 1, 1, 2, 1, 1, 44],
    tomorrow: 'The offer comes in. Now you say a number.',
    assumptions:
      'You pay the same amount every month. Miss the deadline by one month and 26.99% is charged back to day one on the whole $3,000 — that is in the contract, not a guess. Money not yet paid earns 4%.',
  },

  {
    id: 5,
    title: 'The offer comes in. Now you say a number.',
    question: 'How much above their number should you ask for?',
    domain: 'income',
    variable: {
      key: 'counter',
      min: 0,
      max: 20,
      step: 1,
      unit: '%',
      label: 'above their number',
      start: 0,
    },
    fixed: [
      { k: 'THE OFFER', v: '{{salary}}' },
      { k: 'RAISES AFTER THAT', v: '3% A YEAR' },
    ],
    compute: 'anchorOffer',
    // Not the literal 8%: a one-in-twenty-one exact match on a salary
    // negotiation would be false precision, and the model's own peak drifts
    // with age. Every value in the band is within a few percent of the best.
    optimal: { min: 6, max: 10 },
    rule: 'Counter. The first number anchors every number after.',
    action:
      'Write down one number above the offer and the sentence you will say. Practise it out loud once.',
    // Roughly half of people accept the first offer without countering, so the bar
    // at 0 dwarfs everything and is the single most expensive habit in the file:
    // every future raise is a percentage of the number they did not argue with.
    // Everyone who does counter reaches for a round number — 5, 10, 15, 20 — and
    // the spaces between them are nearly empty, which is exactly what an anchoring
    // effect looks like when you plot it.
    crowd: [45, 2, 2, 2, 3, 10, 2, 2, 3, 1, 11, 1, 1, 1, 1, 5, 1, 1, 1, 1, 4],
    tomorrow: 'The refund lands in April. It feels like a bonus.',
    assumptions:
      'Every extra dollar of pay grows with 3% raises for the rest of your career, invested at 7% to 65. The risk of an offer being pulled is our estimate, not a measurement, and it is what caps the dial.',
  },

  {
    id: 6,
    title: 'The refund lands in April. It feels like a bonus.',
    question: 'How big should your April refund be?',
    domain: 'tax',
    variable: {
      key: 'refund',
      min: -2_000,
      max: 8_000,
      step: 250,
      unit: '',
      label: 'back at tax time',
      start: 3_000,
    },
    fixed: [
      { k: 'YOU EARN A YEAR', v: '{{salary}}' },
      { k: 'YOUR TAX BILL', v: 'THE SAME EITHER WAY' },
    ],
    compute: 'withholding',
    // The only optimum in the set that is a rule of law rather than a
    // projection: you may owe up to a tenth of your bill before the IRS
    // charges interest, so the best answer is to owe exactly that much and
    // not a dollar more. It scales with the bill, which is why it is a
    // function — on $28,000 the room is under $250 and the right answer is to
    // break even, while a high earner should be owing four figures every
    // April. Derived from the model so the two cannot drift.
    optimal: (profile) => bestRefund(profile),
    rule: 'Owe a little. A refund is a 0% loan.',
    action:
      'Check last April. If you got more than a few hundred back, file a new W-4 with your employer to move it into your paycheck.',
    // The IRS reports refunds on roughly two thirds of individual returns,
    // averaging a little over $3,000, so the mass sits well right of zero and
    // the dial opens there. The bump at $0 is the small group who tune a W-4
    // deliberately; the long right tail is people using withholding as a
    // savings account. Everything left of zero is the minority who owe, nearly
    // all of them by accident rather than by plan.
    // 41 positions: -$2,000 to $8,000 in $250 steps.
    crowd: [
      1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 3, 2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21,
      22, 22, 21, 19, 17, 15, 13, 11, 9, 8, 7, 6, 5, 4, 4, 3, 3, 2, 2,
    ],
    tomorrow: 'The mechanic calls with a number.',
    assumptions:
      'Federal income tax only, filing single, no credits and no other income. Money the IRS holds for you earns you nothing. Aim too low and you can owe interest on top of the bill.',
  },
  {
    id: 7,
    title: 'The mechanic calls with a number.',
    question: 'How much should you spend fixing it?',
    domain: 'vehicles',
    variable: {
      key: 'repairSpend',
      min: 0,
      max: 4000,
      step: 100,
      unit: '',
      label: 'dollars to fix the old car',
      start: 3200,
    },
    fixed: [
      { k: 'THE CAR IS WORTH', v: '$5,000' },
      { k: 'OR BUY A NEWER ONE', v: '$22,000 AT $441/MO' },
    ],
    compute: 'repairOrReplace',
    // $0 is NOT optimal here, and an earlier band that started there said it was:
    // walking away from a $5,000 car to start a $441 payment is the worst play on
    // the whole dial. The band is the plateau around half the car's value, where
    // the repair still buys more months than the payments it defers.
    optimal: { min: 2000, max: 3000 },
    rule: 'Fix it up to half the car, then stop.',
    action:
      'Look up what your car is worth privately. Write the number down and keep it next to the repair quote.',
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
    tomorrow: 'Two funds hold the same stocks. One costs more.',
    assumptions:
      'A $22,000 replacement paid off over 60 months is $441 a month. We assume the repair buys two more years, with the first few hundred dollars buying most of that — our estimate, not a measurement.',
  },

  {
    id: 8,
    title: 'Two funds hold the same stocks. One costs more.',
    question: 'What is the most you should pay a fund each year?',
    domain: 'investing',
    variable: {
      key: 'expenseRatio',
      min: 3,
      max: 150,
      step: 3,
      unit: '',
      label: 'dollars a year per $10,000 invested',
      start: 54,
    },
    fixed: [
      { k: 'YOU HAVE INVESTED', v: '{{salary}}' },
      { k: 'IT GROWS', v: '7% A YEAR BEFORE FEES' },
    ],
    compute: 'feeDragCall',
    optimal: 3,
    rule: 'Check the yearly fee. Same stocks, cheaper.',
    action:
      'Find the yearly fee on every fund you hold. Anything over $50 a year per $10,000 has a cheaper twin.',
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
      'A year of your pay invested now, plus a tenth of your pay every year after, growing at 7% a year before fees until you are 65. The fee is in the contract. The 7% is not.',
  },

  {
    id: 9,
    title: 'Everyone says buy. The payment matches rent.',
    question: 'How long do you have to stay for buying to beat renting?',
    domain: 'housing',
    variable: {
      key: 'years',
      min: 0,
      max: 15,
      step: 1,
      unit: 'yr',
      label: 'in the house before you sell',
      start: 2,
    },
    fixed: [
      { k: 'THE HOUSE', v: '$360,000, 6.5% LOAN' },
      { k: 'OR KEEP RENTING', v: '$2,160 A MONTH' },
    ],
    compute: 'rentVsBuy',
    optimal: { min: 5, max: 15 },
    rule: 'Rent under five years. Buying costs more.',
    action:
      'Before you make an offer, write down how long you honestly expect to stay. If it is under five years, run the numbers again.',
    // Buyers systematically over-forecast how long they will stay, so the mass sits
    // high: spikes at 5 and 10 (the two numbers anyone says out loud) and a wall at
    // 15 meaning "forever". The honest answer for most people is the sparse left
    // half — jobs move, relationships change, and the transaction costs do not care
    // about intentions. This is the one call where the crowd and the optimum mostly
    // agree, and they agree for the wrong reason.
    crowd: [2, 2, 4, 9, 5, 18, 5, 4, 6, 3, 17, 1, 2, 2, 2, 18],
    tomorrow: 'The market drops. Everyone says get out.',
    assumptions:
      '$360,000, a fifth paid up front, 30 years at 6.5%. Tax and upkeep run 1.5% a year, house and rent both rise 3%, buying and selling cost 9%, and the cash you put down would otherwise earn 7%.',
  },

  {
    id: 10,
    title: 'The market drops. Everyone says get out.',
    question: 'How many of the market\'s best days can you miss?',
    domain: 'investing',
    variable: {
      key: 'daysMissed',
      min: 0,
      max: 30,
      step: 1,
      unit: '',
      label: 'of the market\'s best days, missed',
      start: 4,
    },
    fixed: [
      { k: 'YOU HAVE INVESTED', v: '{{salary}}' },
      { k: 'LEFT ALONE', v: 'TO 65 AT 7% A YEAR' },
    ],
    compute: 'timingMarket',
    optimal: 0,
    // Not "missing ten days halves your return": the receipt for ten days says
    // you keep 58%, because the model keeps contributing while you sit out, and
    // a rule the player's own screen contradicts is worse than no rule.
    rule: 'Stay in. The best days hide inside the worst.',
    action:
      'Set your investing to happen automatically on payday, so the decision is already made when the news is bad.',
    // Almost nobody believes a handful of days can matter, so the hump sits at three
    // to six — "I was only out for a couple of weeks". The spike at 10 is the famous
    // statistic leaking back in, and the block at 0 is the minority who have already
    // been burned once. The long flat tail past 15 is people who sat out an entire
    // cycle and have stopped counting.
    crowd: [
      16, 4, 7, 9, 8, 9, 6, 4, 4, 2, 9, 1, 1, 1, 1, 3, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      3,
    ],
    tomorrow: 'Your boss adds 50 cents to every dollar you save.',
    assumptions:
      'A year of your pay invested now, plus a tenth of your pay every year after, growing at 7% to 65. Missing the ten best days roughly halves what you end up with. That is the finding, not a forecast.',
  },
]

export const callCount = CALLS.length

const BY_ID = new Map(CALLS.map((c) => [c.id, c]))

/** Undefined rather than throwing: a deep link to /99 is a 404 screen, not a crash. */
export function callById(id: number): Call | undefined {
  return BY_ID.get(id)
}
