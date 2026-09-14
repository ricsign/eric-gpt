/**
 * Dated, sourced constants.
 *
 * Three rules, learned the hard way by everyone who has shipped a finance product:
 *
 *  1. **No number lives in prose.** Every figure the app states is a row here, with
 *     the primary source and the date it was read. Lesson copy interpolates the row.
 *     When the IRS publishes 2027 limits, one file changes.
 *  2. **Every number has an expiry.** A contribution limit is correct for a tax
 *     year and wrong the moment the next notice lands. `asOf` and `expires` make a
 *     stale figure detectable rather than silently wrong.
 *  3. **Confidence is recorded, not assumed.** Anything not read from the primary
 *     source is marked, and the UI can decline to state it flatly.
 *
 * Verified 2026-09-11 against the primary sources cited on each row.
 */

export type Jurisdiction = 'US' | 'other'

export interface Fact {
  id: string
  /** Human label, used in tables and tooltips. */
  label: string
  value: number
  /** How to render it. */
  unit: 'usd' | 'rate' | 'count'
  /** Tax year or effective year the figure applies to. */
  year: number
  /** Date the primary source was read. */
  asOf: string
  /** Primary source URL. Shown to the user, not just kept in a comment. */
  source: string
  sourceName: string
  /**
   * `primary` means read from the issuing authority's own document.
   * `secondary` means taken from a reputable aggregator and not independently confirmed.
   */
  confidence: 'primary' | 'secondary'
  jurisdiction: Jurisdiction
  note?: string
}

const F = (f: Fact) => f

/* ---- Retirement and tax-advantaged accounts (IRS Notice 2025-67) ----------- */

export const FACTS: Record<string, Fact> = {
  contrib401k: F({
    id: 'contrib401k',
    label: '401(k) elective deferral limit',
    value: 24_500,
    unit: 'usd',
    year: 2026,
    asOf: '2026-09-11',
    source: 'https://www.irs.gov/pub/irs-drop/n-25-67.pdf',
    sourceName: 'IRS Notice 2025-67',
    confidence: 'primary',
    jurisdiction: 'US',
  }),

  catchUp50: F({
    id: 'catchUp50',
    label: '401(k) catch-up, age 50+',
    value: 8_000,
    unit: 'usd',
    year: 2026,
    asOf: '2026-09-11',
    source: 'https://www.irs.gov/pub/irs-drop/n-25-67.pdf',
    sourceName: 'IRS Notice 2025-67',
    confidence: 'primary',
    jurisdiction: 'US',
  }),

  catchUp60to63: F({
    id: 'catchUp60to63',
    label: 'SECURE 2.0 "super" catch-up, ages 60–63',
    value: 11_250,
    unit: 'usd',
    year: 2026,
    asOf: '2026-09-11',
    source: 'https://www.irs.gov/pub/irs-drop/n-25-67.pdf',
    sourceName: 'IRS Notice 2025-67',
    confidence: 'primary',
    jurisdiction: 'US',
    note: 'Unchanged from 2025 — this one did not index up.',
  }),

  rothCatchUpWageThreshold: F({
    id: 'rothCatchUpWageThreshold',
    label: 'Wages above which catch-up must be Roth',
    value: 150_000,
    unit: 'usd',
    year: 2026,
    asOf: '2026-09-11',
    source: 'https://www.irs.gov/pub/irs-drop/n-25-67.pdf',
    sourceName: 'IRS Notice 2025-67',
    confidence: 'primary',
    jurisdiction: 'US',
    note:
      'Prior-year FICA wages from the same employer. A job-changer may escape it; the self-employed with no FICA wages are not subject; and if the plan has no Roth option, catch-up eligibility is lost entirely.',
  }),

  contribIra: F({
    id: 'contribIra',
    label: 'IRA contribution limit',
    value: 7_500,
    unit: 'usd',
    year: 2026,
    asOf: '2026-09-11',
    source: 'https://www.irs.gov/pub/irs-drop/n-25-67.pdf',
    sourceName: 'IRS Notice 2025-67',
    confidence: 'primary',
    jurisdiction: 'US',
  }),

  iraCatchUp: F({
    id: 'iraCatchUp',
    label: 'IRA catch-up, age 50+',
    value: 1_100,
    unit: 'usd',
    year: 2026,
    asOf: '2026-09-11',
    source: 'https://www.irs.gov/pub/irs-drop/n-25-67.pdf',
    sourceName: 'IRS Notice 2025-67',
    confidence: 'primary',
    jurisdiction: 'US',
    note:
      'The first increase ever — SECURE 2.0 made this indexed. Most third-party content still says $1,000.',
  }),

  hsaSelf: F({
    id: 'hsaSelf',
    label: 'HSA contribution limit, self-only',
    value: 4_400,
    unit: 'usd',
    year: 2026,
    asOf: '2026-09-11',
    source: 'https://www.irs.gov/pub/irs-drop/rp-25-19.pdf',
    sourceName: 'IRS Rev. Proc. 2025-19',
    confidence: 'primary',
    jurisdiction: 'US',
  }),

  hsaFamily: F({
    id: 'hsaFamily',
    label: 'HSA contribution limit, family',
    value: 8_750,
    unit: 'usd',
    year: 2026,
    asOf: '2026-09-11',
    source: 'https://www.irs.gov/pub/irs-drop/rp-25-19.pdf',
    sourceName: 'IRS Rev. Proc. 2025-19',
    confidence: 'primary',
    jurisdiction: 'US',
  }),

  standardDeductionSingle: F({
    id: 'standardDeductionSingle',
    label: 'Standard deduction, single',
    value: 16_100,
    unit: 'usd',
    year: 2026,
    asOf: '2026-09-11',
    source:
      'https://www.irs.gov/newsroom/irs-releases-tax-inflation-adjustments-for-tax-year-2026-including-amendments-from-the-one-big-beautiful-bill',
    sourceName: 'IRS Rev. Proc. 2025-32',
    confidence: 'primary',
    jurisdiction: 'US',
  }),

  standardDeductionMfj: F({
    id: 'standardDeductionMfj',
    label: 'Standard deduction, married filing jointly',
    value: 32_200,
    unit: 'usd',
    year: 2026,
    asOf: '2026-09-11',
    source:
      'https://www.irs.gov/newsroom/irs-releases-tax-inflation-adjustments-for-tax-year-2026-including-amendments-from-the-one-big-beautiful-bill',
    sourceName: 'IRS Rev. Proc. 2025-32',
    confidence: 'primary',
    jurisdiction: 'US',
  }),

  socialSecurityWageBase: F({
    id: 'socialSecurityWageBase',
    label: 'Social Security taxable wage base',
    value: 184_500,
    unit: 'usd',
    year: 2026,
    asOf: '2026-09-11',
    source:
      'https://www.federalregister.gov/documents/2025/11/03/2025-19763/cost-of-living-increase-and-other-determinations-for-2026',
    sourceName: 'SSA, 90 FR (2025-11-03)',
    confidence: 'primary',
    jurisdiction: 'US',
    note:
      'Earnings above this stop paying the 6.2% OASDI tax, so take-home pay rises mid-year for high earners.',
  }),

  /* ---- Rates and yields, September 2026 ----------------------------------- */

  hysaNationalAverage: F({
    id: 'hysaNationalAverage',
    label: 'FDIC national average savings rate',
    value: 0.0038,
    unit: 'rate',
    year: 2026,
    asOf: '2026-08-17',
    source: 'https://www.fdic.gov/national-rates-and-rate-caps',
    sourceName: 'FDIC National Rates and Rate Caps',
    confidence: 'primary',
    jurisdiction: 'US',
  }),

  hysaCompetitive: F({
    id: 'hysaCompetitive',
    label: 'Competitive high-yield savings APY',
    value: 0.04,
    unit: 'rate',
    year: 2026,
    asOf: '2026-09-11',
    source: 'https://www.fdic.gov/national-rates-and-rate-caps',
    sourceName: 'Market range, Sept 2026 (3.75%–4.21%)',
    confidence: 'secondary',
    jurisdiction: 'US',
    note: 'A range, not a rate. Shown as "around 4%" and never as a specific product.',
  }),

  irsUnderpayment: F({
    id: 'irsUnderpayment',
    label: 'IRS underpayment interest rate, individuals',
    value: 0.07,
    unit: 'rate',
    year: 2026,
    asOf: '2026-09-11',
    source: 'https://www.irs.gov/payments/quarterly-interest-rates',
    sourceName: 'IRS quarterly interest rates',
    confidence: 'primary',
    jurisdiction: 'US',
    note: 'Federal short-term rate plus 3 points, reset every quarter. Held flat here for a full year, which is the only way to state it as one number.',
  }),

  treasury10y: F({
    id: 'treasury10y',
    label: '10-year Treasury par yield',
    value: 0.0495,
    unit: 'rate',
    year: 2026,
    asOf: '2026-09-10',
    source:
      'https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value_month=202609',
    sourceName: 'US Treasury daily par yield curve',
    confidence: 'primary',
    jurisdiction: 'US',
  }),

  iBondComposite: F({
    id: 'iBondComposite',
    label: 'I bond composite rate (May–Oct 2026)',
    value: 0.0426,
    unit: 'rate',
    year: 2026,
    asOf: '2026-09-11',
    source: 'https://www.treasurydirect.gov/savings-bonds/i-bonds/i-bonds-interest-rates/',
    sourceName: 'TreasuryDirect',
    confidence: 'primary',
    jurisdiction: 'US',
    note: 'Fixed 0.90% plus semiannual inflation 1.67%.',
  }),

  studentLoanUndergrad: F({
    id: 'studentLoanUndergrad',
    label: 'Federal Direct undergraduate loan rate',
    value: 0.0652,
    unit: 'rate',
    year: 2026,
    asOf: '2026-09-11',
    source:
      'https://fsapartners.ed.gov/knowledge-center/library/electronic-announcements/2026-06-04/interest-rates-federal-direct-loans-first-disbursed-between-july-1-2026-and-june-30-2027',
    sourceName: 'US Dept. of Education, FSA',
    confidence: 'primary',
    jurisdiction: 'US',
    note: 'Disbursed 2026-07-01 to 2027-06-30. Set from the May 10-year Treasury auction plus 2.05 points.',
  }),

  /* ---- Costs -------------------------------------------------------------- */

  expenseRatioIndexEquity: F({
    id: 'expenseRatioIndexEquity',
    label: 'Average index equity mutual fund expense ratio',
    value: 0.0005,
    unit: 'rate',
    year: 2025,
    asOf: '2026-09-11',
    source: 'https://www.ici.org/files/2026/per32-01.pdf',
    sourceName: 'ICI, Trends in the Expenses and Fees of Funds, 2025',
    confidence: 'secondary',
    jurisdiction: 'US',
    note: 'Asset-weighted average.',
  }),

  expenseRatioAllEquity: F({
    id: 'expenseRatioAllEquity',
    label: 'Average expense ratio, all equity mutual funds',
    value: 0.004,
    unit: 'rate',
    year: 2025,
    asOf: '2026-09-11',
    source: 'https://www.ici.org/files/2026/per32-01.pdf',
    sourceName: 'ICI, Trends in the Expenses and Fees of Funds, 2025',
    confidence: 'secondary',
    jurisdiction: 'US',
    note:
      'The honest comparison is 0.05% vs 0.40%, not a strawman 1.5% fund. It is still a large number over thirty years.',
  }),

  behaviorGap: F({
    id: 'behaviorGap',
    label: 'Gap between investor returns and fund returns',
    value: 0.012,
    unit: 'rate',
    year: 2026,
    asOf: '2026-09-11',
    source: 'https://www.morningstar.com/funds/investors-still-need-mind-gap-their-funds-returns',
    sourceName: 'Morningstar, Mind the Gap 2026',
    confidence: 'secondary',
    jurisdiction: 'US',
    note:
      'Decade to 2025-12-31: 8.7% for the average dollar vs 9.9% for the funds. The methodology is disputed in the Financial Analysts Journal (2026). Volatility, not fees, drove the gap: −0.4% for the least volatile funds, −2.1% for the most.',
  }),

  /* ---- Credit -------------------------------------------------------------- */

  ficoPaymentHistory: F({
    id: 'ficoPaymentHistory',
    label: 'FICO weight: payment history',
    value: 0.35,
    unit: 'rate',
    year: 2026,
    asOf: '2026-09-11',
    source: 'https://www.myfico.com/credit-education/whats-in-your-credit-score',
    sourceName: 'myFICO',
    confidence: 'primary',
    jurisdiction: 'US',
  }),

  ficoAmountsOwed: F({
    id: 'ficoAmountsOwed',
    label: 'FICO weight: amounts owed',
    value: 0.3,
    unit: 'rate',
    year: 2026,
    asOf: '2026-09-11',
    source: 'https://www.myfico.com/credit-education/whats-in-your-credit-score',
    sourceName: 'myFICO',
    confidence: 'primary',
    jurisdiction: 'US',
  }),

  /* ---- Employer plans ------------------------------------------------------ */

  averageEmployerContribution: F({
    id: 'averageEmployerContribution',
    label: 'Average employer retirement contribution',
    value: 0.047,
    unit: 'rate',
    year: 2026,
    asOf: '2026-09-11',
    source: 'https://workplace.vanguard.com/insights-and-research/report/how-america-saves-2026.html',
    sourceName: 'Vanguard, How America Saves 2026',
    confidence: 'secondary',
    jurisdiction: 'US',
    note: 'Share of pay. The most common formula is 50% of contributions up to 6% of pay.',
  }),
}

/** FICO's five factors, for the credit lesson. */
export const FICO_FACTORS = [
  { label: 'Payment history', weight: 0.35 },
  { label: 'Amounts owed', weight: 0.3 },
  { label: 'Length of credit history', weight: 0.15 },
  { label: 'Credit mix', weight: 0.1 },
  { label: 'New credit', weight: 0.1 },
] as const

/** 2026 federal brackets, single filer. Used by the marginal-vs-effective widget. */
export const BRACKETS_2026_SINGLE = [
  { rate: 0.1, upTo: 12_400 },
  { rate: 0.12, upTo: 50_400 },
  { rate: 0.22, upTo: 105_700 },
  { rate: 0.24, upTo: 201_775 },
  { rate: 0.32, upTo: 256_225 },
  { rate: 0.35, upTo: 640_600 },
  { rate: 0.37, upTo: Infinity },
] as const

export const BRACKETS_SOURCE = {
  name: 'IRS Rev. Proc. 2025-32',
  url: 'https://www.irs.gov/newsroom/irs-releases-tax-inflation-adjustments-for-tax-year-2026-including-amendments-from-the-one-big-beautiful-bill',
  asOf: '2026-09-11',
}

export function fact(id: keyof typeof FACTS): Fact {
  return FACTS[id]
}

/**
 * How stale a figure is, in days. The UI uses this to soften a claim rather than
 * state it flatly once it ages past a threshold.
 */
export function staleness(f: Fact, today: string): number {
  const toUtc = (s: string) => {
    const [y, m, d] = s.split('-').map(Number)
    return Date.UTC(y, m - 1, d)
  }
  return Math.max(0, Math.round((toUtc(today) - toUtc(f.asOf)) / 86_400_000))
}
