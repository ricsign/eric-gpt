import { useMemo } from 'react'
import {
  DOMAIN,
  TEAR_SEGMENTS,
  barcodeUnits,
  barcodeWidths,
  deltaLine,
  receiptCode,
  receiptLines,
  receiptSeed,
  stampText,
  tearClipPath,
  type ReceiptInput,
} from '../lib/receipt'
import { money } from '../lib/format'
import './Receipt.css'

/** Matches the canvas renderer's tear depth, scaled to the on-screen card. */
const TEAR_DEPTH = 11

/**
 * The receipt, on screen.
 *
 * Designed backwards from the feed: the first test of this card is not whether
 * it reads at 362px, it is whether it reads at 120px in someone's timeline. So
 * the hierarchy is brutal — a solid ink brand band, one enormous figure, one
 * stamp — and everything else is deliberately small grey texture that is allowed
 * to dissolve at thumbnail size.
 *
 * Every irregular thing on it (the tear, the bars) is seeded from the player's
 * own result via lib/receipt, so this card and the shared PNG are the same
 * object and re-rendering never reshuffles it.
 */
export function Receipt({ input }: { input: ReceiptInput }) {
  const seed = receiptSeed(input)
  const lines = receiptLines(input)
  const delta = deltaLine(input)

  const clipPath = useMemo(() => tearClipPath(seed, TEAR_DEPTH, TEAR_SEGMENTS), [seed])
  const barcode = useMemo(() => {
    const widths = barcodeWidths(seed)
    const startOf = (i: number) => widths.slice(0, i).reduce((a, b) => a + b, 0)
    // Even modules are ink, odd are paper; only the ink ones need a rect.
    const bars = widths.flatMap((w, i) => (i % 2 === 0 ? [{ x: startOf(i), w }] : []))
    return { bars, units: barcodeUnits(widths) }
  }, [seed])

  return (
    <div className="receipt-lift">
      <figure className="receipt" style={{ clipPath }}>
        <div className="receipt-brand">
          <span className="receipt-brand-text">COMPOUND</span>
        </div>

        <div className="receipt-meta">
          <span>No.{input.callNo}</span>
          <span>{input.date}</span>
        </div>

        <div className="receipt-rule" />

        <h2 className="receipt-title">{input.title}</h2>

        <div className="receipt-rule" />

        <ul className="receipt-items">
          {lines.map((line, i) => (
            <li
              className="receipt-item"
              key={`${line.label}-${i}`}
              data-emphasis={line.emphasis === true ? '' : undefined}
            >
              <span className="receipt-k">{line.label}</span>
              <span className="receipt-leader" aria-hidden="true" />
              <span className="receipt-v num">{line.value}</span>
            </li>
          ))}
        </ul>

        <div className="receipt-rule receipt-rule--double" />

        <div className="receipt-total">
          <span className="receipt-k">Total at 65</span>
          <strong className="receipt-hero num">{money(input.at65)}</strong>
        </div>

        <div className="receipt-item receipt-delta" data-verdict={input.verdict}>
          <span className="receipt-k">{delta.label}</span>
          <span className="receipt-leader" aria-hidden="true" />
          <span className="receipt-v num">{delta.value}</span>
        </div>

        <div className="receipt-stamp" data-verdict={input.verdict}>
          <span className="receipt-stamp-text">*** {stampText(input.verdict)} ***</span>
        </div>

        <svg
          className="receipt-barcode"
          viewBox={`0 0 ${barcode.units} 10`}
          preserveAspectRatio="none"
          aria-hidden="true"
        >
          {barcode.bars.map((b) => (
            <rect key={b.x} x={b.x} y="0" width={b.w} height="10" />
          ))}
        </svg>

        <div className="receipt-foot">
          <span className="num">{receiptCode(input)}</span>
          <span className="receipt-domain">{DOMAIN}</span>
        </div>
      </figure>
    </div>
  )
}
