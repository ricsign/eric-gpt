import './Mechanism.css'

/**
 * The mechanism diagrams.
 *
 * One image per beat, drawn inline rather than loaded, so they inherit the theme
 * tokens and stay crisp at any size. Each one shows a *mechanism*, not a
 * decoration: if you removed the caption, the picture should still say something.
 */
export function Mechanism({ visual }: { visual: 'doubling' | 'split' | 'reverse' | 'bracket' | 'none' }) {
  if (visual === 'none') return null

  return (
    <div className="mech">
      {visual === 'doubling' && <Doubling />}
      {visual === 'split' && <Split />}
      {visual === 'reverse' && <Reverse />}
      {visual === 'bracket' && <Bracket />}
    </div>
  )
}

/** Four doublings, as countable blocks. Turns a multiplicative process into a countable one. */
function Doubling() {
  const steps = [1, 2, 4, 8, 16]
  return (
    <figure className="mech-fig">
      <div className="mech-doubling">
        {steps.map((n, i) => (
          <div key={n} className="mech-doubling-col">
            <div
              className="mech-doubling-bar"
              style={{ height: `${(n / 16) * 100}%` }}
              aria-hidden="true"
            />
            <span className="mech-doubling-label num">${n}</span>
            <span className="mech-doubling-year">{i === 0 ? 'now' : `${i * 10}y`}</span>
          </div>
        ))}
      </div>
      <figcaption>
        Each bar is one doubling. Four of them is sixteen times the money, not four times —
        which is why the last decade always looks like it did all the work.
      </figcaption>
    </figure>
  )
}

/** Contributions against growth, and the year the two swap places. */
function Split() {
  return (
    <figure className="mech-fig">
      <svg viewBox="0 0 300 120" className="mech-svg" aria-hidden="true">
        {/* Contributions: a straight line, because they are a straight line. */}
        <path d="M8 112 L292 64" className="mech-line mech-line--flat" />
        {/* Total: the same money, compounding. */}
        <path d="M8 112 C 110 106, 200 78, 292 10" className="mech-line mech-line--curve" />
        <circle cx="292" cy="64" r="3.5" className="mech-dot mech-dot--flat" />
        <circle cx="292" cy="10" r="4" className="mech-dot" />
        <text x="286" y="56" className="mech-label" textAnchor="end">
          what you put in
        </text>
        <text x="286" y="26" className="mech-label mech-label--growth" textAnchor="end">
          what it became
        </text>
      </svg>
      <figcaption>
        Contributions rise in a straight line. The balance does not. The gap between them is
        the part nobody paid for.
      </figcaption>
    </figure>
  )
}

/** The same curve, pointed the other way. */
function Reverse() {
  return (
    <figure className="mech-fig">
      <svg viewBox="0 0 300 120" className="mech-svg" aria-hidden="true">
        <path d="M8 20 C 100 24, 190 48, 292 110" className="mech-line mech-line--drag" />
        <circle cx="8" cy="20" r="4" className="mech-dot mech-dot--drag" />
        <text x="16" y="16" className="mech-label">
          what you borrowed
        </text>
        <text x="286" y="104" className="mech-label mech-label--drag" textAnchor="end">
          what you repaid
        </text>
      </svg>
      <figcaption>
        On a debt the same curve runs against you: interest is charged on interest, so the
        total owed grows faster the longer the balance survives.
      </figcaption>
    </figure>
  )
}

/** Tax brackets as stacked layers, each taxed at its own rate. */
function Bracket() {
  const layers = [
    { rate: '10%', height: 18 },
    { rate: '12%', height: 34 },
    { rate: '22%', height: 28 },
  ]
  return (
    <figure className="mech-fig">
      <div className="mech-brackets">
        {[...layers].reverse().map((l) => (
          <div key={l.rate} className="mech-bracket" style={{ height: `${l.height}px` }}>
            <span>{l.rate}</span>
          </div>
        ))}
      </div>
      <figcaption>
        Each slice of income is taxed at its own rate. Moving into a higher bracket only
        changes the rate on the slice above the line.
      </figcaption>
    </figure>
  )
}
