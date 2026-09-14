import { useMemo } from 'react'
import {
  PEN_SEGMENTS,
  TEAR_SEGMENTS,
  napkinSeed,
  penPathData,
  penStroke,
  shareHandle,
  tearClipPath,
  type NapkinInput,
} from '../lib/receipt'
import './Receipt.css'

/** Matches the canvas renderer's tear depth, scaled to the on-screen card. */
const TEAR_DEPTH = 11

/**
 * The napkin, on screen.
 *
 * Designed backwards from the feed: the first test of this card is not whether
 * it reads at 362px, it is whether a stranger scrolling past at 120px can tell
 * what it is. So the hierarchy is brutal — the question, then the answer, and
 * everything else is small grey texture that is allowed to dissolve.
 *
 * What it does *not* show is the point of it. There is no projection, no gap to
 * the right answer, and no verdict, because the card that carried those was a
 * picture of the sender being wrong and nobody sends that. The sender's own
 * guess appears only when they ask for it, in the smallest type on the paper.
 *
 * Every irregular thing on it — the torn edge, the pen stroke — is seeded from
 * what is printed on it via lib/receipt, so this card and the shared PNG are the
 * same object and re-rendering never reshuffles it.
 */
export function Napkin({ input }: { input: NapkinInput }) {
  const seed = napkinSeed(input)
  const handle = shareHandle(input.questionNo)

  const clipPath = useMemo(() => tearClipPath(seed, TEAR_DEPTH, TEAR_SEGMENTS), [seed])
  // Drawn in a 100 x 2 box and stretched to whatever width the figure ends up
  // being; `vector-effect` below keeps the nib from stretching with it.
  const stroke = useMemo(() => penPathData(penStroke(seed, 100, PEN_SEGMENTS), 1), [seed])

  return (
    <div className="napkin-lift">
      <figure className="napkin" style={{ clipPath }}>
        <header className="napkin-head">
          <span className="napkin-mark">Napkin</span>
          <span className="napkin-meta num">
            Question {input.questionNo} · {input.date}
          </span>
        </header>

        <div className="napkin-rule" />

        <p className="napkin-scene">{input.scene}</p>

        <h2 className="napkin-question">{input.question}</h2>

        <ul className="napkin-givens">
          {input.givens.map((g) => (
            <li className="napkin-given" key={g.k}>
              <span className="napkin-k">{g.k}</span>
              <span className="napkin-leader" aria-hidden="true" />
              <span className="napkin-v num">{g.v}</span>
            </li>
          ))}
        </ul>

        <div className="napkin-rule" />

        <div className="napkin-answer">
          <p className="napkin-kicker">The answer</p>
          {/* The figure and its stroke are one inline-block so the stroke ends
              where the words do, the way a pen would, rather than running the
              width of the column. */}
          <span className="napkin-figure-wrap">
            <strong className="napkin-figure num">{input.answer}</strong>
            <svg
              className="napkin-stroke"
              viewBox="0 0 100 2"
              preserveAspectRatio="none"
              aria-hidden="true"
            >
              <path d={stroke} vectorEffect="non-scaling-stroke" />
            </svg>
          </span>
          <p className="napkin-note">{input.note}</p>
        </div>

        <div className="napkin-rule" />

        <div className="napkin-block">
          <p className="napkin-kicker">The rule</p>
          <p className="napkin-rule-text">{input.rule}</p>
        </div>

        {input.guess !== undefined && <p className="napkin-guess">I guessed {input.guess}.</p>}

        <footer className="napkin-foot">
          <span>Estimates, not advice.</span>
          {handle !== '' && <span className="napkin-where">{handle}</span>}
        </footer>
      </figure>
    </div>
  )
}
