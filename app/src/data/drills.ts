/**
 * The Daily Drill bank.
 *
 * Constraints every question here obeys:
 *
 *  - **No personal data.** The question is identical for every player worldwide, so
 *    a group chat can argue about it, and so the shared result discloses nothing.
 *  - **Counter-intuitive, not obscure.** A good drill is one where the confident
 *    answer is wrong. Trivia about contribution limits is not a drill; it is a
 *    quiz, and quizzes do not get forwarded.
 *  - **Every wrong option is a real misconception**, not filler. The distractor is
 *    where the teaching happens — for most players the wrong answer they nearly
 *    picked is the thing they remember.
 *  - **The reveal shows the arithmetic.** Losing should still be worth the sixty
 *    seconds, or people stop coming back.
 */

export interface Drill {
  id: string
  question: string
  options: {
    label: string
    correct?: boolean
    /** Why this option attracts people. Shown after the attempt. */
    why: string
  }[]
  /** The arithmetic, shown after the drill resolves either way. */
  reveal: string
  /** Concept id, so a missed drill can feed the review queue. */
  concept: string
}

export const DRILLS: Drill[] = [
  {
    id: 'penny-doubling',
    question:
      'A penny that doubles every day for 30 days, or $1,000,000 today. Which is worth more on day 30?',
    options: [
      { label: 'The million, easily', why: 'The doubling penny is worth almost nothing for the first three weeks, which is exactly why this feels obvious and is wrong.' },
      { label: 'The penny', correct: true, why: 'Doubling is far more violent than intuition allows.' },
      { label: 'About the same', why: 'They are not close in either direction.' },
      { label: 'The penny, but only after 60 days', why: 'It passes the million well before day 30.' },
    ],
    reveal:
      'The penny reaches $5,368,709 on day 30. It only passes the million on day 28, at $1,342,177 — the day before that it was $671,089, and on day 20 it was still $5,243. Almost all of it arrives in the last three days.',
    concept: 'exponential-growth',
  },
  {
    id: 'which-costs-most',
    question:
      'Over 10 years at a 7% return, which of these costs you the most in forgone growth?',
    options: [
      { label: 'A $6/day coffee habit', why: 'The famous one, and the smallest of the four. About $2,190 a year.' },
      { label: 'A $450/month car payment', correct: true, why: 'Large recurring costs dwarf small ones, however often the small one gets moralised about.' },
      { label: 'A $60/month subscription stack', why: 'Real money, but a seventh of the car payment.' },
      { label: 'One $4,000 holiday', why: 'A single expense, not a recurring one — it compounds once, not 120 times.' },
    ],
    reveal:
      'At 7% over 10 years: the car payment forgoes about $77,000, the coffee about $31,200, subscriptions about $10,300, the one-off holiday about $7,900. Recurrence matters more than size, and size matters more than guilt.',
    concept: 'opportunity-cost',
  },
  {
    id: 'apr-vs-apy',
    question: 'A card quotes 24% APR, compounded monthly. What do you actually pay over a year?',
    options: [
      { label: '24%', why: 'APR is the quoted nominal rate, not what a carried balance costs.' },
      { label: '26.8%', correct: true, why: 'Monthly compounding turns 24% nominal into 26.8% effective.' },
      { label: '22%', why: 'Compounding never makes a debt cheaper.' },
      { label: '48%', why: 'Compounding adds, but not that much at monthly frequency.' },
    ],
    reveal: '(1 + 0.24/12)^12 − 1 = 26.82%. The gap between APR and APY widens as the rate rises — at 6% it is only 0.17 points.',
    concept: 'apr-vs-apy',
  },
  {
    id: 'fifty-percent-loss',
    question: 'An investment falls 50%. What return does it need to get back to even?',
    options: [
      { label: '50%', why: 'The instinct to reverse a percentage with the same percentage. Percentages do not compose that way.' },
      { label: '100%', correct: true, why: 'You are now growing a halved number.' },
      { label: '75%', why: 'A compromise between the two intuitions, and not how the arithmetic works.' },
      { label: '150%', why: 'Doubling the remainder is enough.' },
    ],
    reveal:
      '$100 → $50. To get from $50 back to $100 you must double it: +100%. This asymmetry is why avoiding large losses matters more than capturing large gains.',
    concept: 'percentage-composition',
  },
  {
    id: 'match-vs-card',
    question:
      'One spare dollar. A 401(k) matched 50 cents on the dollar, or a credit card at 24%. Which first?',
    options: [
      { label: 'The card — it is the highest rate you have', why: 'The right rule almost everywhere, and wrong here. A match is not a rate.' },
      { label: 'The 401(k), up to the match', correct: true, why: 'A 50% match is a 50% return the instant it lands.' },
      { label: 'Split it', why: 'Splitting forfeits half the match to make slower progress on the card.' },
      { label: 'Depends on your tax bracket', why: 'The bracket changes the size of the win, never which one wins.' },
    ],
    reveal:
      'A 50% match returns 50% immediately, before any market movement. No consumer debt charges 50% a year. The exception: if the match has not vested and you are leaving soon, it is not yet yours.',
    concept: 'employer-match',
  },
  {
    id: 'utilization-myth',
    question: 'Which of these actually helps your credit score?',
    options: [
      { label: 'Carrying a small balance month to month', why: 'The most persistent myth in consumer credit. It costs interest and does nothing a paid-in-full statement would not.' },
      { label: 'Paying the balance before the statement closes', correct: true, why: 'Utilisation is measured from the reported statement balance.' },
      { label: 'Closing cards you no longer use', why: 'Closing a card removes its limit, which raises your utilisation ratio.' },
      { label: 'Keeping utilisation at exactly 30%', why: 'There is no cliff at 30%. It is a gradient, and lower is better.' },
    ],
    reveal:
      'FICO weights payment history 35% and amounts owed 30%. myFICO states plainly that the 30% figure is not a threshold and that utilisation "has no memory" — pay it down and the score responds. Source: myfico.com.',
    concept: 'credit-utilization',
  },
  {
    id: 'start-early-vs-more',
    question:
      'A saves $200/month from 25 to 35, then stops forever. B saves $200/month from 35 to 65. At 7%, who has more at 65?',
    options: [
      { label: 'B — they saved three times as much', why: 'True on contributions, false on outcome. B put in $72,000 against A\'s $24,000.' },
      { label: 'A', correct: true, why: 'A\'s ten years get thirty more years of doubling on top.' },
      { label: 'They tie almost exactly', why: 'Closer than the famous version of this example suggests — but A still wins.' },
      { label: 'Impossible to say without the fees', why: 'Fees matter, but not enough to reverse this.' },
    ],
    reveal:
      'At 7%: A ends at about $260,400 having contributed $24,000. B ends at about $233,900 having contributed $72,000. A put in a third as much and still finished ahead by roughly $26,500. Versions of this example circulating online quote far bigger gaps by quietly assuming 10-12% returns — at an honest 7% the effect is smaller, and still remarkable.',
    concept: 'time-in-market',
  },
  {
    id: 'inflation-cash',
    question: 'Cash earning 0.4% while inflation runs 2.5%. What happens to it over ten years?',
    options: [
      { label: 'It grows slowly but it grows', why: 'Nominally yes. That is not the question worth asking about money you intend to spend.' },
      { label: 'It loses about a fifth of its purchasing power', correct: true, why: 'A 2.1-point real loss, compounded for a decade.' },
      { label: 'It loses about 2% in total', why: 'That is one year, not ten, and it compounds.' },
      { label: 'It breaks even', why: 'Only if inflation fell to 0.4%.' },
    ],
    reveal:
      '1.004^10 / 1.025^10 = 0.81. $10,000 still reads as $10,408 but buys what $8,100 buys today. Nominal safety and real safety are different things.',
    concept: 'real-vs-nominal',
  },
  {
    id: 'rule-of-72',
    question: 'At 6%, roughly how long does money take to double?',
    options: [
      { label: 'About 12 years', correct: true, why: '72 ÷ 6 = 12. The exact answer is 11.9.' },
      { label: 'About 17 years', why: 'That is 100 ÷ 6 — a common but wrong shortcut.' },
      { label: 'About 6 years', why: 'That would need roughly 12%.' },
      { label: 'About 20 years', why: 'That is closer to 3.5%.' },
    ],
    reveal:
      'The Rule of 72 is near-exact at 8% (it says 9.0 years; the true figure is 9.006). It slightly overestimates below 8% and underestimates above it.',
    concept: 'rule-of-72',
  },
  {
    id: 'minimum-payment',
    question: '$5,000 on a card at 22%, paying a fixed $200 a month. How long to clear it?',
    options: [
      { label: 'About 25 months', why: 'That is $5,000 ÷ $200 — the answer if the rate were zero.' },
      { label: 'About 34 months', correct: true, why: 'Interest adds nine months and about $1,750.' },
      { label: 'About 60 months', why: 'Too pessimistic at this payment level.' },
      { label: 'It never clears', why: 'It would not clear at $95/month. At $200 it does.' },
    ],
    reveal:
      '34 months, with $1,749.88 of interest. The first month\'s $200 payment includes $91.67 of interest, so only $108 comes off the balance.',
    concept: 'debt-compounding',
  },
  {
    id: 'fee-drag',
    question:
      'Two identical index funds, one charging 0.03% and one charging 1%. Over 40 years, what does the pricier one cost you?',
    options: [
      { label: 'About 1% of the final balance', why: 'The intuitive answer: read the fee, assume it is the cost. The fee is charged every year.' },
      { label: 'Over a fifth of the final balance', correct: true, why: 'The fee compounds against you exactly as returns compound for you.' },
      { label: 'About 5%', why: 'Still thinking additively rather than multiplicatively.' },
      { label: 'About 40%', why: 'Too high for this gap over this period.' },
    ],
    reveal:
      'A 0.97-point difference over 40 years costs about 22% of the terminal balance. Field rule: multiply the fee by roughly 0.6 times the years you will be invested — 40 years gives about 24x, 20 years only about 11x. The popular "multiply by 25" version is right only at a full career length.',
    concept: 'expense-ratio',
  },
  {
    id: 'avalanche-vs-snowball',
    question:
      'Four debts: $500 at 8%, $2,000 at 24%, $8,000 at 6%, $300 at 19%. Which do you attack first to pay the least interest?',
    options: [
      { label: 'The $300 — quickest win', why: 'The snowball. Motivationally strong, mathematically second-best.' },
      { label: 'The $2,000 at 24%', correct: true, why: 'Highest rate first always minimises total interest.' },
      { label: 'The $8,000 — biggest balance', why: 'Balance size does not determine the cost of carrying it; the rate does.' },
      { label: 'Spread it evenly', why: 'Even spreading is strictly worse than any prioritised order.' },
    ],
    reveal:
      'Highest rate first ("avalanche") always costs the least. The honest caveat: the smallest-balance order ("snowball") has real evidence behind it for people who need visible progress to keep going — a plan you follow beats an optimal plan you abandon.',
    concept: 'debt-ordering',
  },
  {
    id: 'diversification',
    question: 'Which portfolio is better diversified?',
    options: [
      { label: '30 tech stocks', why: 'Count is not diversification. Thirty names that all fall together behave like one name.' },
      { label: '3 funds: US stocks, international stocks, bonds', correct: true, why: 'Diversification is about correlation, not count.' },
      { label: '10 stocks across 10 industries', why: 'Better than 30 tech names, still exposed to single-company risk in each slot.' },
      { label: '50 stocks picked at random', why: 'Reduces company risk, leaves market and geography risk untouched.' },
    ],
    reveal:
      'Diversification is measured by how little the parts move together. Three low-correlation funds beat thirty high-correlation stocks.',
    concept: 'diversification',
  },
  {
    id: 'sunk-cost',
    question:
      'You have paid $3,000 into a course you now dislike. $1,500 is left to pay. A better course costs $1,200 total. What matters?',
    options: [
      { label: 'The $3,000 already spent — do not waste it', why: 'The sunk cost. The $3,000 is gone in both futures, so it cannot distinguish between them.' },
      { label: 'Only the $1,500 against the $1,200', correct: true, why: 'Compare the remaining costs and the remaining benefits. Nothing else is a live variable.' },
      { label: 'The total: $4,500 against $1,200', why: 'The $3,000 is spent whichever you choose, so including it on one side only is the error.' },
      { label: 'Finish what you start', why: 'A good rule for habits, a costly one for investments.' },
    ],
    reveal:
      'Arkes & Blumer (1985) named this. The test: if you had never paid the $3,000, which would you choose today? That is the answer, because the $3,000 is not coming back either way.',
    concept: 'sunk-cost',
  },
  {
    id: 'marginal-rate',
    question:
      'You are in the 22% bracket and a raise pushes $1,000 into the 24% bracket. What happens to tax on your existing income?',
    options: [
      { label: 'All of it is now taxed at 24%', why: 'The single most expensive tax misconception — it makes people turn down raises.' },
      { label: 'Nothing — only the $1,000 is taxed at 24%', correct: true, why: 'Brackets are marginal. Each layer is taxed at its own rate.' },
      { label: 'It depends on your deductions', why: 'Deductions change which layers you reach, not how layers work.' },
      { label: 'You could take home less overall', why: 'Not possible from a bracket change alone. Benefit cliffs can do it; tax brackets cannot.' },
    ],
    reveal:
      'A bracket applies only to the dollars inside it. Moving into a higher bracket never reduces your take-home pay. Your effective rate is always well below your marginal rate.',
    concept: 'marginal-vs-effective',
  },
  {
    id: 'emergency-fund-location',
    question: 'Where does a six-month emergency fund belong?',
    options: [
      { label: 'Invested in index funds for growth', why: 'Emergencies correlate with market falls — job losses cluster in downturns. You would sell at the worst time.' },
      { label: 'A high-yield savings account', correct: true, why: 'Insured, liquid, and currently paying a real rate.' },
      { label: 'In your checking account', why: 'Safe and liquid, but typically earns nothing.' },
      { label: 'Split between crypto and cash', why: 'An emergency fund with a volatile half is not an emergency fund.' },
    ],
    reveal:
      'The job of this money is to be there on a bad day, not to grow. The FDIC national average savings rate was 0.38% in August 2026 while competitive accounts paid around 4% — same insurance, same access.',
    concept: 'emergency-fund',
  },
  {
    id: 'lifestyle-creep',
    question: 'A $10,000 raise. Which choice makes the biggest difference at retirement?',
    options: [
      { label: 'Save all $10,000 for one year, then spend it after', why: 'One year of contributions. Real, but one-off.' },
      { label: 'Save half of it, permanently', correct: true, why: 'A permanent change to the monthly amount compounds every year that follows.' },
      { label: 'Pay down the mortgage by $10,000 once', why: 'Helpful, and a single event at a typically low rate.' },
      { label: 'Spend it — one raise will not change the outcome', why: 'A permanent $5,000/year is transformative over decades.' },
    ],
    reveal:
      '$5,000 a year for 30 years at 7% becomes about $487,000. The same $10,000 saved once and left alone becomes about $76,000. Recurring beats one-off, by a lot.',
    concept: 'lifestyle-creep',
  },
  {
    id: 'checking-frequency',
    question: 'Two investors hold the identical portfolio for 20 years. One checks daily, one yearly. Who tends to end up with more?',
    options: [
      { label: 'The daily checker — more informed', why: 'More information about a random process is not more insight; it is more noise.' },
      { label: 'The yearly checker', correct: true, why: 'Seeing more short-term losses provokes more selling at the wrong moments.' },
      { label: 'Identical — same portfolio', why: 'Only if neither ever acted on what they saw. They do.' },
      { label: 'Depends on the market', why: 'The effect shows up in rising and falling markets alike.' },
    ],
    reveal:
      'Myopic loss aversion (Benartzi & Thaler). Morningstar\'s Mind the Gap 2026 put the investor-versus-fund gap at 1.2 points a year over the decade to 2025 — and found volatility, not fees, drove it. The methodology is disputed, but the direction is not.',
    concept: 'myopic-loss-aversion',
  },
  {
    id: 'mental-accounting',
    question:
      'You have $5,000 in a savings account earning 4% and $5,000 on a card at 22%. What is the savings account really earning you?',
    options: [
      { label: '4%', why: 'What the statement says, and not what the money is doing.' },
      { label: '−18%', correct: true, why: 'Holding both simultaneously costs the difference.' },
      { label: '26%', why: 'Adding the two rates rather than netting them.' },
      { label: '0%', why: 'They do not cancel; the debt rate is far higher.' },
    ],
    reveal:
      'Earning 4% while paying 22% on the same $5,000 is a net 18% loss. The caveat that keeps this honest: an emergency fund buys you the ability not to borrow more at 22% next month, which is worth something real.',
    concept: 'mental-accounting',
  },
  {
    id: 'car-loan-true-cost',
    question: 'A $35,000 car on a 72-month loan at 7.5%. What do you actually pay?',
    options: [
      { label: 'About $37,000', why: 'Underestimates six years of interest.' },
      { label: 'About $43,600', correct: true, why: 'About $8,570 of interest on top.' },
      { label: 'About $39,000', why: 'Closer to a 48-month term.' },
      { label: 'About $52,000', why: 'Too high for this rate.' },
    ],
    reveal:
      '$605.15/month x 72 = $43,571, of which $8,571 is interest. The longer term lowers the payment and raises the total — and for much of those 72 months the loan balance exceeds the car\'s value.',
    concept: 'loan-term',
  },
  {
    id: 'roth-vs-traditional',
    question: 'What actually decides between Roth and traditional contributions?',
    options: [
      { label: 'Your age — Roth when young', why: 'A decent heuristic precisely because income usually rises. But age is the proxy, not the reason.' },
      { label: 'Your tax rate now versus in retirement', correct: true, why: 'The whole decision is which rate you would rather pay.' },
      { label: 'Which one grows faster', why: 'Identical growth. Only the timing of the tax differs.' },
      { label: 'Whichever your employer offers', why: 'Availability constrains the choice; it does not decide it.' },
    ],
    reveal:
      'Traditional deducts now and taxes later; Roth taxes now and is free later. Pay the lower rate. If the two rates are equal, the outcomes are mathematically identical — and since nobody knows their future rate, holding some of each is the usual answer.',
    concept: 'roth-vs-traditional',
  },
  {
    id: 'stock-picking',
    question: 'Over 20 years, roughly what share of actively managed US large-cap funds beat their index?',
    options: [
      { label: 'About half', why: 'The intuition that skill should be evenly distributed. Fees make it a losing game before skill enters.' },
      { label: 'Under 10%', correct: true, why: 'Persistently, across every long measurement period.' },
      { label: 'About 70%', why: 'This is what fund marketing implies and what the data contradicts.' },
      { label: 'About 30%', why: 'Roughly right over one year; far too high over twenty.' },
    ],
    reveal:
      'SPIVA has found under 10% of active US large-cap funds beat the S&P 500 over 20-year periods. Note the survivorship point: many of the losers closed, so the surviving record flatters the category.',
    concept: 'active-vs-passive',
  },
  {
    id: 'raise-vs-return',
    question: 'For someone 25 with $5,000 invested, which is worth more by 65?',
    options: [
      { label: 'One extra percentage point of annual return', why: 'Powerful, and almost entirely outside your control.' },
      { label: 'Adding $200 a month, every month', correct: true, why: 'The contribution is larger, and it is the part you can actually decide.' },
      { label: 'A $20,000 lump sum today', why: 'Substantial, but a single event against 480 recurring ones.' },
      { label: 'Retiring two years later', why: 'Meaningful, and much smaller than the other three.' },
    ],
    reveal:
      'At 7% over 40 years: +$200/month adds about $494,000. The $20,000 lump sum adds about $299,000. An extra point of return on the original $5,000 adds about $34,000. Savings rate is the lever you control; returns are the one you hope for.',
    concept: 'savings-rate',
  },
  {
    id: 'dollar-cost-averaging',
    question:
      'You inherit $50,000. Invest it all at once, or spread it over 12 months? Which wins more often historically?',
    options: [
      { label: 'Spreading it out — lower risk', why: 'Lower regret, genuinely. But markets rise more often than they fall, so waiting usually costs.' },
      { label: 'All at once', correct: true, why: 'It wins roughly two times in three, because time in the market is the dominant factor.' },
      { label: 'Exactly even odds', why: 'The upward drift of markets tilts it.' },
      { label: 'Wait for a dip', why: 'The dip may come from a higher level than today.' },
    ],
    reveal:
      'Lump-sum investing beats averaging in roughly two-thirds of historical periods. The honest counterpoint: averaging is not about returns, it is about not abandoning the plan after a bad first month — and that is a real risk with a real cost.',
    concept: 'lump-sum-vs-dca',
  },
]

export const DRILL_COUNT = DRILLS.length
