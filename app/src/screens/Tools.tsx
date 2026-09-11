import { useMemo, useState } from 'react'
import { Screen } from '../ui/Screen'
import { Card } from '../ui/Card'
import { Icon } from '../ui/Icon'
import { BackButton, useNav } from '../ui/NavStack'
import { ValueSlider } from '../ui/ValueSlider'
import { GrowthChart } from '../ui/GrowthChart'
import { NumberRoll } from '../ui/NumberRoll'
import { useStore } from '../state/store'
import { growthSeries } from '../lib/finance'
import { debtFreeDate, federalTax, independence, matchGap, rentVsBuy } from '../lib/calculators'
import { duration, money, moneyCompact, percent } from '../lib/format'
import { BRACKETS_SOURCE, FACTS } from '../data/facts'
import './Tools.css'

/**
 * The tools.
 *
 * These are the reason to open the app in month six. Every one of them obeys the
 * same contract, which is both a product rule and a legal one: the output is a
 * date, a dollar amount, a threshold or a percentile — never a verdict. A tool may
 * compute. It may not conclude.
 *
 * All of them are free and unmetered. The calculators are the trust artifact; the
 * moment you meter them they stop being a neutral utility and become a tease.
 */

interface ToolDef {
  id: string
  name: string
  /** The question in the user's words, not the tool's. */
  question: string
  /** The shape of the answer, stated up front so nobody expects advice. */
  answers: string
  usOnly?: boolean
  render: () => React.ReactNode
}

export function ToolsScreen() {
  const { state } = useStore()
  const nav = useNav()
  const isUS = (state.jurisdiction ?? 'US') === 'US'

  const tools: ToolDef[] = [
    {
      id: 'growth',
      name: 'What it becomes',
      question: 'If I keep saving this much, where does it end up?',
      answers: 'A balance, split into what you put in and what growth added',
      render: () => <GrowthTool />,
    },
    {
      id: 'debt',
      name: 'Debt-free date',
      question: 'When is this actually gone, and what does waiting cost?',
      answers: 'A date and a total interest figure',
      render: () => <DebtTool />,
    },
    {
      id: 'coast',
      name: 'Coast point',
      question: 'When could I stop adding and still be fine?',
      answers: 'A balance and the number of years to reach it',
      render: () => <CoastTool />,
    },
    {
      id: 'tax',
      name: 'Marginal vs effective',
      question: 'What does a raise actually cost me in tax?',
      answers: 'Two rates and a bracket-by-bracket breakdown',
      usOnly: true,
      render: () => <TaxTool />,
    },
    {
      id: 'match',
      name: 'Match gap',
      question: 'How much employer money am I leaving behind?',
      answers: 'An annual figure and what it compounds to',
      usOnly: true,
      render: () => <MatchTool />,
    },
    {
      id: 'rentbuy',
      name: 'Rent vs buy',
      question: 'How long would I need to stay for buying to win?',
      answers: 'A break-even year — never a recommendation',
      render: () => <RentBuyTool />,
    },
  ]

  const available = tools.filter((t) => !t.usOnly || isUS)

  return (
    <Screen title="Tools" eyebrow="Free, unmetered, no sign-up">
      <div className="tools">
        {available.map((t) => (
          <Card key={t.id} onClick={() => nav.push(t.id, t.render)}>
            <div className="tool-row">
              <div className="tool-row-text">
                <p className="tool-row-name">{t.name}</p>
                <p className="tool-row-question">{t.question}</p>
                <p className="tool-row-answers">Gives you: {t.answers}</p>
              </div>
              <Icon name="chevron" size={15} stroke className="tool-row-chevron" />
            </div>
          </Card>
        ))}

        <p className="tools-note">
          Each tool computes a number from assumptions you control. None of them recommends a
          product, names a fund, or tells you what to do — and there is nowhere in this app a
          company could pay to appear.
        </p>
      </div>
    </Screen>
  )
}

/* ---- Shell ----------------------------------------------------------------- */

function Tool({
  title,
  headline,
  headlineNote,
  children,
  disclaimer,
}: {
  title: string
  headline: React.ReactNode
  headlineNote: string
  children: React.ReactNode
  disclaimer: string
}) {
  return (
    <Screen title={title} inlineTitle left={<BackButton label="Tools" />} noTabBar>
      <div className="tool">
        <div className="tool-headline">
          <div className="tool-headline-value">{headline}</div>
          <p className="tool-headline-note">{headlineNote}</p>
        </div>

        <div className="tool-body">{children}</div>

        {/* Point-of-use disclosure. A footer-only disclaimer is weak evidence of
            impersonality when the interaction itself is personalised. */}
        <p className="tool-disclaimer">{disclaimer}</p>
      </div>
    </Screen>
  )
}

/* ---- What it becomes -------------------------------------------------------- */

function GrowthTool() {
  const { state } = useStore()
  const p = state.profile
  const [monthly, setMonthly] = useState(p.monthly ?? 300)
  const [start, setStart] = useState(p.invested ?? 0)
  const [years, setYears] = useState(Math.max(10, (p.targetAge ?? 65) - (p.age ?? 30)))
  const [rate, setRate] = useState(p.assumedReturn * 100)

  const points = useMemo(
    () => growthSeries({ principal: start, monthly, annualRate: rate / 100, years }),
    [start, monthly, rate, years],
  )
  const end = points[points.length - 1]

  return (
    <Tool
      title="What it becomes"
      headline={
        <NumberRoll value={end.balance} format={moneyCompact} duration={1.2} className="tool-big" />
      }
      headlineNote={`after ${years} years — ${money(end.contributed)} put in, ${money(
        end.growth,
      )} added by growth`}
      disclaimer={`An illustration at a constant ${percent(
        rate / 100,
      )}, which no real market delivers. Returns vary year to year and can be negative. Not advice.`}
    >
      <GrowthChart points={points} height={220} showReadout={false} />

      <ValueSlider label="Added each month" value={monthly} onChange={setMonthly} min={0} max={5_000} step={25} format={money} />
      <ValueSlider label="Starting balance" value={start} onChange={setStart} min={0} max={1_000_000} step={500} curve="log" format={money} />
      <ValueSlider label="Years" value={years} onChange={setYears} min={1} max={50} format={(v) => `${v}`} />
      <ValueSlider
        label="Assumed annual return"
        value={rate}
        onChange={setRate}
        min={0}
        max={15}
        step={0.25}
        format={(v) => `${v.toFixed(2)}%`}
        hint="Drag this. Watching the answer move is the most honest thing a projection can show you."
      />
    </Tool>
  )
}

/* ---- Debt-free date ---------------------------------------------------------- */

function DebtTool() {
  const { state } = useStore()
  const p = state.profile
  const [balance, setBalance] = useState(p.debtBalance || 4_200)
  const [apr, setApr] = useState((p.debtApr ?? 0.229) * 100)
  const [payment, setPayment] = useState(Math.max(50, Math.round((p.debtBalance || 4200) * 0.03)))

  const result = useMemo(
    () =>
      debtFreeDate(
        [{ id: 'd', name: 'Debt', balance, apr: apr / 100, monthlyPayment: payment }],
        0,
      ),
    [balance, apr, payment],
  )

  const minimum = Math.max(25, balance * 0.01 + (balance * (apr / 100)) / 12)

  return (
    <Tool
      title="Debt-free date"
      headline={
        <span className="tool-big num">
          {result.neverClears ? 'Never' : duration(result.months)}
        </span>
      }
      headlineNote={
        result.neverClears
          ? 'At this payment the interest charged matches what you send, so the balance does not fall.'
          : `${money(result.totalInterest)} of interest along the way`
      }
      disclaimer="Assumes a fixed payment, a fixed rate, and no new spending on the account. Real cards vary all three. Not advice."
    >
      <ValueSlider label="Balance" value={balance} onChange={setBalance} min={100} max={60_000} step={100} curve="log" format={money} />
      <ValueSlider label="Interest rate" value={apr} onChange={setApr} min={0} max={36} step={0.1} format={(v) => `${v.toFixed(1)}%`} />
      <ValueSlider
        label="Monthly payment"
        value={payment}
        onChange={setPayment}
        min={10}
        max={Math.max(500, Math.round(balance / 6))}
        step={10}
        format={money}
        hint={`A typical minimum on this balance would be about ${money(minimum)}.`}
      />

      <div className="tool-compare">
        {[1, 1.5, 2].map((mult) => {
          const r = debtFreeDate(
            [{ id: 'd', name: 'Debt', balance, apr: apr / 100, monthlyPayment: payment * mult }],
            0,
          )
          return (
            <div key={mult} className="tool-compare-item" data-highlight={mult === 1 || undefined}>
              <p className="tool-compare-label">{money(payment * mult)}/mo</p>
              <p className="tool-compare-value num">
                {r.neverClears ? 'never' : duration(r.months)}
              </p>
              <p className="tool-compare-sub num">
                {r.neverClears ? '—' : `${money(r.totalInterest)} interest`}
              </p>
            </div>
          )
        })}
      </div>
    </Tool>
  )
}

/* ---- Coast point ------------------------------------------------------------- */

function CoastTool() {
  const { state } = useStore()
  const p = state.profile
  const [spend, setSpend] = useState(Math.round(((p.income ?? 65_000) * 0.7) / 1000) * 1000)
  const [invested, setInvested] = useState(p.invested ?? 0)
  const [monthly, setMonthly] = useState(p.monthly ?? 300)
  const [withdrawal, setWithdrawal] = useState(4)

  const r = useMemo(
    () =>
      independence({
        age: p.age ?? 30,
        annualSpend: spend,
        invested,
        monthly,
        takeHomeAnnual: (p.income ?? 65_000) * 0.78,
        realReturn: p.assumedReturn - p.assumedInflation,
        withdrawalRate: withdrawal / 100,
        retireAge: p.targetAge ?? 65,
      }),
    [p, spend, invested, monthly, withdrawal],
  )

  return (
    <Tool
      title="Coast point"
      headline={
        <span className="tool-big num">
          {r.alreadyCoasting
            ? 'Reached'
            : Number.isFinite(r.yearsToCoast)
              ? `${Math.round(r.yearsToCoast)} years`
              : '—'}
        </span>
      }
      headlineNote={
        r.alreadyCoasting
          ? `What you have already grows into ${moneyCompact(r.target)} by ${p.targetAge ?? 65} with nothing added.`
          : `until ${moneyCompact(r.coastTarget)} invested — the point where you could stop adding entirely and still reach ${moneyCompact(r.target)} by ${p.targetAge ?? 65}.`
      }
      disclaimer={`Uses a real return of ${percent(
        p.assumedReturn - p.assumedInflation,
      )} (your return assumption minus inflation). The withdrawal rate is a convention from historical US data, not a guarantee — and it says nothing about the order returns arrive in, which matters enormously near retirement.`}
    >
      <div className="tool-stats">
        <Stat label="Full target" value={moneyCompact(r.target)} />
        <Stat label="Coast target" value={moneyCompact(r.coastTarget)} tone="growth" />
        <Stat
          label="Full target in"
          value={Number.isFinite(r.yearsToTarget) ? `${Math.round(r.yearsToTarget)}y` : '—'}
        />
        <Stat label="Savings rate" value={percent(r.savingsRate, 0)} />
      </div>

      <ValueSlider label="Annual spending in retirement" value={spend} onChange={setSpend} min={10_000} max={250_000} step={1_000} curve="log" format={money} />
      <ValueSlider label="Already invested" value={invested} onChange={setInvested} min={0} max={2_000_000} step={1_000} curve="log" format={money} />
      <ValueSlider label="Added each month" value={monthly} onChange={setMonthly} min={0} max={10_000} step={50} format={money} />
      <ValueSlider
        label="Withdrawal rate"
        value={withdrawal}
        onChange={setWithdrawal}
        min={2}
        max={6}
        step={0.1}
        format={(v) => `${v.toFixed(1)}%`}
        hint="4% is the convention. Lower is more cautious and needs a bigger portfolio."
      />
    </Tool>
  )
}

/* ---- Tax --------------------------------------------------------------------- */

function TaxTool() {
  const { state } = useStore()
  const [income, setIncome] = useState(state.profile.income ?? 80_000)
  const std = FACTS.standardDeductionSingle.value
  const t = useMemo(() => federalTax(income, std), [income, std])
  const raise = useMemo(() => federalTax(income + 5_000, std), [income, std])

  return (
    <Tool
      title="Marginal vs effective"
      headline={
        <span className="tool-big num">
          {percent(t.effectiveRate)} <span className="tool-big-vs">vs</span> {percent(t.marginalRate)}
        </span>
      }
      headlineNote="your effective rate, against the rate on your next dollar"
      disclaimer={`Federal income tax only, tax year 2026, single filer taking the standard deduction. Excludes payroll tax, state tax, credits and every other real-world complication. Source: ${BRACKETS_SOURCE.name}, read ${BRACKETS_SOURCE.asOf}.`}
    >
      <div className="tool-brackets">
        {t.fill.map((f) => (
          <div key={f.rate} className="tool-bracket">
            <div
              className="tool-bracket-bar"
              style={{ width: `${(f.amountInBracket / Math.max(1, t.taxableIncome)) * 100}%` }}
            />
            <span className="tool-bracket-rate num">{percent(f.rate, 0)}</span>
            <span className="tool-bracket-amount num">{money(f.amountInBracket)}</span>
            <span className="tool-bracket-tax num">{money(f.taxFromBracket)}</span>
          </div>
        ))}
      </div>

      <div className="tool-stats">
        <Stat label="Taxable income" value={money(t.taxableIncome)} />
        <Stat label="Federal tax" value={money(t.totalTax)} />
        <Stat label="Take-home" value={money(t.takeHome)} tone="growth" />
        <Stat label="A $5,000 raise keeps" value={money(raise.takeHome - t.takeHome)} tone="growth" />
      </div>

      <p className="tool-insight">
        A $5,000 raise leaves you {money(raise.takeHome - t.takeHome)} better off, not worse. Moving
        into a higher bracket only changes the rate on the dollars above the line — never on the
        ones below it.
      </p>

      <ValueSlider label="Gross annual income" value={income} onChange={setIncome} min={0} max={600_000} step={1_000} curve="log" format={money} />
    </Tool>
  )
}

/* ---- Match gap ---------------------------------------------------------------- */

function MatchTool() {
  const { state } = useStore()
  const p = state.profile
  const [salary, setSalary] = useState(p.income ?? 65_000)
  const [current, setCurrent] = useState(3)
  const [limit, setLimit] = useState(6)
  const [matchRate, setMatchRate] = useState(50)

  const r = useMemo(
    () =>
      matchGap({
        salary,
        currentRate: current / 100,
        matchLimit: limit / 100,
        matchRate: matchRate / 100,
        age: p.age ?? 30,
        retireAge: p.targetAge ?? 65,
        annualReturn: p.assumedReturn,
      }),
    [salary, current, limit, matchRate, p],
  )

  return (
    <Tool
      title="Match gap"
      headline={
        <NumberRoll
          value={r.annualUnclaimed}
          format={money}
          duration={1}
          className="tool-big"
        />
      }
      headlineNote={
        r.annualUnclaimed > 0
          ? `of employer money unclaimed each year — about ${moneyCompact(
              r.byRetirement,
            )} by ${p.targetAge ?? 65} if it were invested`
          : 'You are collecting the whole match.'
      }
      disclaimer="A match is only yours once it vests — check your plan's schedule. Contribution limits also cap what you can put in. Not advice."
    >
      {r.annualUnclaimed > 0 && (
        <p className="tool-insight">
          Contributing {money(r.extraMonthlyNeeded)} more a month would collect all of it. Within
          the matched band, a {matchRate}% match is a {matchRate}% return the instant it lands —
          before the market does anything at all.
        </p>
      )}

      <ValueSlider label="Salary" value={salary} onChange={setSalary} min={0} max={400_000} step={1_000} curve="log" format={money} />
      <ValueSlider label="You contribute" value={current} onChange={setCurrent} min={0} max={20} step={0.5} format={(v) => `${v}% of pay`} />
      <ValueSlider label="Employer matches up to" value={limit} onChange={setLimit} min={0} max={15} step={0.5} format={(v) => `${v}% of pay`} />
      <ValueSlider
        label="Employer pays"
        value={matchRate}
        onChange={setMatchRate}
        min={0}
        max={100}
        step={5}
        format={(v) => `${v}¢ per $1`}
        hint="The most common formula is 50 cents on the dollar up to 6% of pay."
      />
    </Tool>
  )
}

/* ---- Rent vs buy --------------------------------------------------------------- */

function RentBuyTool() {
  const [price, setPrice] = useState(450_000)
  const [down, setDown] = useState(90_000)
  const [rate, setRate] = useState(6.5)
  const [rent, setRent] = useState(2_200)
  const [appreciation, setAppreciation] = useState(3)

  const r = useMemo(
    () =>
      rentVsBuy({
        homePrice: price,
        downPayment: Math.min(down, price),
        mortgageRate: rate / 100,
        termYears: 30,
        monthlyRent: rent,
        rentGrowth: 0.03,
        homeAppreciation: appreciation / 100,
        ownershipCostRate: 0.02,
        investmentReturn: 0.07,
        sellingCostRate: 0.06,
      }),
    [price, down, rate, rent, appreciation],
  )

  return (
    <Tool
      title="Rent vs buy"
      headline={
        <span className="tool-big num">
          {r.breakEvenYear ? `Year ${r.breakEvenYear}` : 'Never, here'}
        </span>
      }
      headlineNote={
        r.breakEvenYear
          ? `is when buying overtakes renting on these numbers. Staying less than that and renting costs less.`
          : 'On these inputs renting stays cheaper for the whole 30 years. Change the rent or the appreciation assumption and that flips.'
      }
      disclaimer="Counts the mortgage, 2% a year in tax/insurance/maintenance, 6% selling costs, 3% rent growth, and the return the down payment would have earned if invested instead at 7%. Excludes the mortgage interest deduction and every local specific. This is a break-even year, not a recommendation."
    >
      <ValueSlider label="Home price" value={price} onChange={setPrice} min={80_000} max={2_000_000} step={5_000} curve="log" format={money} />
      <ValueSlider label="Down payment" value={down} onChange={setDown} min={0} max={price} step={5_000} format={money} />
      <ValueSlider label="Mortgage rate" value={rate} onChange={setRate} min={0} max={12} step={0.125} format={(v) => `${v.toFixed(3)}%`} />
      <ValueSlider label="Rent you would pay instead" value={rent} onChange={setRent} min={400} max={12_000} step={50} format={money} />
      <ValueSlider
        label="Home appreciation"
        value={appreciation}
        onChange={setAppreciation}
        min={-3}
        max={8}
        step={0.25}
        format={(v) => `${v.toFixed(2)}%`}
        hint="The single assumption this answer is most sensitive to. Drag it and watch."
      />
    </Tool>
  )
}

/* ---- Bits ---------------------------------------------------------------------- */

function Stat({ label, value, tone }: { label: string; value: string; tone?: 'growth' }) {
  return (
    <div className="tool-stat" data-tone={tone}>
      <p className="tool-stat-label">{label}</p>
      <p className="tool-stat-value num">{value}</p>
    </div>
  )
}
