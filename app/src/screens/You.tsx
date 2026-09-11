import { Screen } from '../ui/Screen'
import { ListGroup, Row } from '../ui/List'
import { ValueSlider } from '../ui/ValueSlider'
import { useNav } from '../ui/NavStack'
import { BackButton } from '../ui/NavStack'
import { calibration, useStore } from '../state/store'
import { money, percent } from '../lib/format'
import { FACTS } from '../data/facts'
import { isPersistent } from '../lib/storage'
import './You.css'

/**
 * Profile, assumptions, and the honest bits.
 *
 * Two things here are load-bearing rather than housekeeping:
 *
 *  - **The assumptions are editable and visible.** Every projection in the app is
 *    driven by a return and an inflation number the user can see and move. A model
 *    that hides its assumptions is asking to be trusted; one that exposes them can
 *    be checked.
 *  - **The sources page is a real page.** Every figure the app states is listed
 *    with the primary document it came from and the date it was read.
 */
export function YouScreen() {
  const { state, setSettings, dispatch } = useStore()
  const nav = useNav()
  const p = state.profile

  const confirmed = state.commitments.filter((c) => c.doneAt)
  const totalCommitted = confirmed.reduce((s, c) => s + (c.worth ?? 0), 0)

  return (
    <Screen title="You">
      <div className="you">
        <Calibration />

        {confirmed.length > 0 && (
          <div className="you-hero">
            <p className="you-hero-label">Actions you have taken</p>
            <p className="you-hero-value num">{confirmed.length}</p>
            {totalCommitted > 0 && (
              <p className="you-hero-sub">
                Worth roughly <strong className="num">{money(totalCommitted)}</strong> on your
                own numbers and assumptions. Not a guarantee — an estimate from the same maths
                you can inspect in each lesson.
              </p>
            )}
          </div>
        )}

        <ListGroup
          header="Your numbers"
          footer="Stored in this browser only. Never uploaded, never attached to an account."
        >
          <Row label="Age" value={p.age ?? '—'} onClick={() => nav.push('numbers', () => <NumbersScreen />)} accessory="chevron" />
          <Row label="Income" value={p.income ? money(p.income) : '—'} onClick={() => nav.push('numbers', () => <NumbersScreen />)} accessory="chevron" />
          <Row label="Invested" value={p.invested != null ? money(p.invested) : '—'} onClick={() => nav.push('numbers', () => <NumbersScreen />)} accessory="chevron" />
          <Row label="Saved monthly" value={p.monthly != null ? money(p.monthly) : '—'} onClick={() => nav.push('numbers', () => <NumbersScreen />)} accessory="chevron" />
        </ListGroup>

        <ListGroup
          header="Assumptions"
          footer="Every projection in the app uses these. Move them and every number moves with them — that is the point of showing them."
        >
          <Row
            label="Expected return"
            value={percent(p.assumedReturn)}
            onClick={() => nav.push('assumptions', () => <AssumptionsScreen />)}
            accessory="chevron"
          />
          <Row
            label="Expected inflation"
            value={percent(p.assumedInflation)}
            onClick={() => nav.push('assumptions', () => <AssumptionsScreen />)}
            accessory="chevron"
          />
          <Row
            label="Show values in today's money"
            value={state.settings.realTerms ? 'On' : 'Off'}
            onClick={() => setSettings({ realTerms: !state.settings.realTerms })}
          />
        </ListGroup>

        <ListGroup header="App">
          <Row
            label="Appearance"
            value={
              state.settings.theme === 'system'
                ? 'System'
                : state.settings.theme === 'dark'
                  ? 'Dark'
                  : 'Light'
            }
            onClick={() => {
              const order = ['system', 'light', 'dark'] as const
              const next = order[(order.indexOf(state.settings.theme) + 1) % order.length]
              setSettings({ theme: next })
            }}
          />
          <Row
            label="Haptics"
            value={state.settings.haptics ? 'On' : 'Off'}
            onClick={() => setSettings({ haptics: !state.settings.haptics })}
          />
          <Row
            label="Country"
            value={state.jurisdiction === 'US' ? 'United States' : 'Other'}
            onClick={() =>
              dispatch({
                type: 'setJurisdiction',
                value: state.jurisdiction === 'US' ? 'other' : 'US',
              })
            }
          />
        </ListGroup>

        <ListGroup header="Where the numbers come from">
          <Row
            label="Sources"
            detail="Every figure, with the document it came from"
            onClick={() => nav.push('sources', () => <SourcesScreen />)}
            accessory="chevron"
          />
          <Row
            label="What this app is not"
            detail="The honest limits"
            onClick={() => nav.push('limits', () => <LimitsScreen />)}
            accessory="chevron"
          />
        </ListGroup>

        {!isPersistent() && (
          <p className="you-warning">
            This browser is blocking storage, so your progress will not survive a reload. Private
            browsing usually causes this.
          </p>
        )}

        <ListGroup>
          <Row
            label="Erase everything"
            destructive
            onClick={() => {
              if (confirm('Erase all your numbers, progress and history from this device?')) {
                dispatch({ type: 'reset' })
              }
            }}
          />
        </ListGroup>
      </div>
    </Screen>
  )
}

/**
 * Calibration — the only metric-like object in the app.
 *
 * It is the median error across every curve you have drawn, and it is here rather
 * than a streak or an XP total for reasons that each matter on their own:
 *
 *  - It measures the actual learning objective (how wrong your intuition about
 *    compounding is) rather than a proxy for attendance.
 *  - It is falsifiable, unlike a composite "financial health score".
 *  - It goes *down*, so it is not a brag ladder.
 *  - Missing a day cannot break it, so it carries no loss frame and needs no
 *    freezes, repairs, or notifications to defend it.
 */
function Calibration() {
  const { state } = useStore()
  const c = calibration(state.predictions)

  if (c.count === 0) {
    return (
      <div className="you-calibration you-calibration--empty">
        <p className="you-calibration-label">Calibration</p>
        <p className="you-calibration-empty">
          Draw a curve and we will start tracking how far off your intuition is. It is the only
          number this app keeps about you, and the goal is for it to fall.
        </p>
      </div>
    )
  }

  const pct = Math.round((c.recent ?? c.median ?? 0) * 100)
  const sparkline = state.predictions.slice(-16)
  const worst = Math.max(0.05, ...sparkline.map((p) => p.error))

  return (
    <div className="you-calibration">
      <p className="you-calibration-label">Calibration</p>
      <p className="you-calibration-value num">{pct}%</p>
      <p className="you-calibration-sub">
        your typical error across {c.count} {c.count === 1 ? 'curve' : 'curves'}
        {c.improving ? ' — and falling' : ''}
      </p>

      {sparkline.length > 2 && (
        <svg
          className="you-spark"
          viewBox={`0 0 ${sparkline.length - 1} 10`}
          preserveAspectRatio="none"
          role="img"
          aria-label={`Prediction error over your last ${sparkline.length} curves`}
        >
          <path
            d={sparkline
              .map((p, i) => `${i === 0 ? 'M' : 'L'}${i},${(p.error / worst) * 10}`)
              .join(' ')}
            fill="none"
            stroke="currentColor"
            strokeWidth="0.5"
            vectorEffect="non-scaling-stroke"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      )}
    </div>
  )
}

/* ------------------------------------------------------------------------- */

function NumbersScreen() {
  const { state, setProfile } = useStore()
  const p = state.profile

  return (
    <Screen title="Your numbers" inlineTitle left={<BackButton label="You" />} noTabBar>
      <div className="you-form">
        <ValueSlider label="Age" value={p.age ?? 28} onChange={(v) => setProfile({ age: v })} min={16} max={80} format={(v) => `${v}`} />
        <ValueSlider
          label="Annual income, before tax"
          value={p.income ?? 65_000}
          onChange={(v) => setProfile({ income: v })}
          min={0}
          max={500_000}
          step={1_000}
          curve="log"
          format={money}
        />
        <ValueSlider
          label="Already invested"
          value={p.invested ?? 0}
          onChange={(v) => setProfile({ invested: v })}
          min={0}
          max={2_000_000}
          step={500}
          curve="log"
          format={money}
        />
        <ValueSlider
          label="Added each month"
          value={p.monthly ?? 0}
          onChange={(v) => setProfile({ monthly: v })}
          min={0}
          max={10_000}
          step={25}
          format={money}
        />
        <ValueSlider
          label="Highest-rate debt balance"
          value={p.debtBalance ?? 0}
          onChange={(v) => setProfile({ debtBalance: v })}
          min={0}
          max={100_000}
          step={100}
          curve="log"
          format={money}
        />
        <ValueSlider
          label="Rate on that debt"
          value={(p.debtApr ?? 0.229) * 100}
          onChange={(v) => setProfile({ debtApr: v / 100 })}
          min={0}
          max={36}
          step={0.1}
          format={(v) => `${v.toFixed(1)}%`}
        />
        <ValueSlider
          label="Age you would like to stop needing a salary"
          value={p.targetAge ?? 65}
          onChange={(v) => setProfile({ targetAge: v })}
          min={40}
          max={80}
          format={(v) => `${v}`}
        />
      </div>
    </Screen>
  )
}

function AssumptionsScreen() {
  const { state, setProfile } = useStore()
  const p = state.profile

  return (
    <Screen title="Assumptions" inlineTitle left={<BackButton label="You" />} noTabBar>
      <div className="you-form">
        <ValueSlider
          label="Expected annual return"
          value={p.assumedReturn * 100}
          onChange={(v) => setProfile({ assumedReturn: v / 100 })}
          min={0}
          max={15}
          step={0.25}
          format={(v) => `${v.toFixed(2)}%`}
          hint="Nominal, before inflation. The long-run US equity average is often quoted near 10% nominal; 7% is a common, more conservative planning figure. Nobody knows the future one."
        />
        <ValueSlider
          label="Expected annual inflation"
          value={p.assumedInflation * 100}
          onChange={(v) => setProfile({ assumedInflation: v / 100 })}
          min={0}
          max={8}
          step={0.25}
          format={(v) => `${v.toFixed(2)}%`}
          hint="Used to show what a future balance would buy in today's money."
        />

        <div className="you-callout">
          <p>
            Your real return — what actually grows your purchasing power — is roughly{' '}
            <strong className="num">{percent(p.assumedReturn - p.assumedInflation)}</strong>. That
            is the number that decides how much a projection is really worth, and it is always
            smaller than the one people quote.
          </p>
        </div>
      </div>
    </Screen>
  )
}

function SourcesScreen() {
  const facts = Object.values(FACTS)

  return (
    <Screen title="Sources" inlineTitle left={<BackButton label="You" />} noTabBar>
      <div className="you-sources">
        <p className="you-sources-intro">
          Every figure this app states, the document it came from, and the date we read it. If a
          number here is out of date, it is wrong — tax limits and rates change, and a finance
          app that hides its vintage is asking you to trust it blindly.
        </p>

        {facts.map((f) => (
          <div key={f.id} className="you-source">
            <div className="you-source-head">
              <span className="you-source-label">{f.label}</span>
              <span className="you-source-value num">
                {f.unit === 'usd' ? money(f.value) : percent(f.value, 2)}
              </span>
            </div>
            <p className="you-source-meta">
              {f.sourceName} · read {f.asOf}
              {f.confidence === 'secondary' && ' · not read from the issuing authority'}
            </p>
            {f.note && <p className="you-source-note">{f.note}</p>}
            <a className="you-source-link" href={f.source} target="_blank" rel="noreferrer">
              {new URL(f.source).hostname}
            </a>
          </div>
        ))}
      </div>
    </Screen>
  )
}

function LimitsScreen() {
  return (
    <Screen title="What this is not" inlineTitle left={<BackButton label="You" />} noTabBar>
      <div className="you-limits selectable">
        <h3>It is not advice</h3>
        <p>
          Everything here is educational. The tools compute — they show what a number becomes
          under assumptions you supply. None of them concludes, and none of them will tell you
          what you should do, because a recommendation tailored to your circumstances is a
          regulated activity and no disclaimer changes that.
        </p>

        <h3>Every projection is an illustration</h3>
        <p>
          Returns are not a straight line and can be negative for years at a time. A chart that
          curves smoothly upward is a model, not a forecast. The order in which returns arrive
          matters enormously near retirement, and none of these tools model that.
        </p>

        <h3>The numbers have a shelf life</h3>
        <p>
          Contribution limits, tax brackets, and yields change. Each one is listed in Sources
          with the date it was read. Anything from a previous tax year is wrong, not merely old.
        </p>

        <h3>It assumes US federal rules</h3>
        <p>
          Where a lesson depends on US accounts or tax law it is hidden if you told us you are
          elsewhere. State and local taxes are not modelled anywhere.
        </p>

        <h3>We are not paid by anyone you might buy from</h3>
        <p>
          No affiliate links, no referral fees, no sponsored placements, no product comparison
          tables. There is no surface in this app where a company could pay to appear, which is
          the only version of that promise that is actually verifiable.
        </p>

        <h3>Nothing leaves your device</h3>
        <p>
          Your age, income, balances and answers are stored in this browser's local storage.
          There is no account, no server, and no analytics on your figures. Clearing your browser
          data erases all of it, including your streak.
        </p>
      </div>
    </Screen>
  )
}
