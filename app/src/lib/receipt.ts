/**
 * The napkin: the share artifact, and the only thing that leaves the app.
 *
 * What this replaces was a receipt, and it was the wrong object twice over. A
 * receipt is proof of money already spent; this product is a rough estimate you
 * work out yourself, which is a napkin. And the old card was built around the
 * sender's score — a stamp reading MONEY LEFT BEHIND over a six-figure figure in
 * red — so all three test readers said, unprompted, that they would never post
 * it. One of them: "that is a picture of me getting money wrong, which I would
 * be posting to people I work with."
 *
 * So the card carries the fact, not the score. The question, the givens it was
 * set with, and the real answer — the thing a reader said they *would* send.
 * The sender's own guess is optional and small, and off unless they ask for it.
 * Nothing here knows what anyone answered unless they choose to say.
 *
 * Two renderings of one design — the DOM card in ui/Receipt.tsx and the PNG
 * drawn here — plus five lines of plain text for group chats where images die.
 * Everything that decides *shape* (the torn edge, the pen stroke, the wording)
 * lives in this file as a pure function and is imported by both renderers. An
 * edge that differs between the screenshot and the PNG is a bug nobody reports
 * and everybody notices.
 *
 * Nothing here ever touches the profile. `NapkinInput` deliberately has no
 * salary field and no `Profile`, so "the share never carries personal financial
 * data" is structural rather than a rule someone has to remember — and
 * `sharedFacts` enforces the same thing for the givens, which are authored
 * templates that the profile would otherwise fill in.
 *
 * The file and the `shareReceipt` entry point keep their old names because the
 * screen and the shell already reach for them by those names.
 */

import type { CallFact, CallVariable } from '../calls/types'
import { money } from '../lib/format'
import { pathForCall } from './route'

/**
 * Where every share points.
 *
 * This was hardcoded to a domain that belongs to a different live company, so
 * every successful share the product ever made sent someone to a competitor.
 * It is read from the environment now, and falls back to wherever the app is
 * actually being served from, which cannot be wrong by construction. Empty only
 * when there is no browser and no configured origin — a test, or a render on a
 * server — and every caller below degrades to saying nothing rather than to
 * printing a broken address.
 */
export const DOMAIN: string =
  import.meta.env.VITE_PUBLIC_ORIGIN ?? (typeof location === 'undefined' ? '' : location.host)

/** The origin with any scheme or trailing slash taken off: what a person reads. */
function bareHost(origin: string): string {
  return origin.replace(/^https?:\/\//i, '').replace(/\/+$/, '')
}

/**
 * The address printed on the card and in the share text: `napkin.example/4`.
 *
 * No scheme, no query, no tracking — a bare host and the question number, which
 * is short enough to read off a screenshot and type by hand. Empty when there is
 * no origin to name.
 */
export function shareHandle(questionNo: number, origin: string = DOMAIN): string {
  const host = bareHost(origin)
  return host ? `${host}${pathForCall(questionNo)}` : ''
}

/**
 * The absolute URL the copy-link button puts on the clipboard.
 *
 * Built from the same origin as the text and the PNG rather than from
 * `location` directly: a preview build configured with a canonical origin would
 * otherwise print one address on the card and copy a different one, and the two
 * would only disagree in the one place nobody checks.
 */
export function shareLink(questionNo: number, origin: string = DOMAIN): string {
  const host = bareHost(origin)
  if (!host) return pathForCall(questionNo)
  const scheme = /^http:\/\//i.test(origin) ? 'http://' : 'https://'
  return `${scheme}${host}${pathForCall(questionNo)}`
}

/** The optimum, with any profile dependence already resolved by the caller. */
export type Answer = number | { min: number; max: number }

/** Everything a napkin needs. Assembled by the screen, never read from storage. */
export interface NapkinInput {
  /** Which of the ten. Also the deep link the card points at. */
  questionNo: number
  /** Already formatted for display, e.g. `SEP 13`. */
  date: string
  /** The scene the question sits in. One sentence, from the registry. */
  scene: string
  /** The question the dial answers. The hook, and the largest text on the card. */
  question: string
  /** The givens. Guaranteed profile-free — see `sharedFacts`. */
  givens: CallFact[]
  /** The real answer, in the words the dial used: `12mo or less`. */
  answer: string
  /** What the answer is measured in: `to pay it off in full`. */
  note: string
  /** The takeaway, one line. */
  rule: string
  /** The sender's own answer. Absent unless they asked for it. */
  guess?: string
}

export type ShareOutcome = 'shared' | 'copied' | 'cancelled' | 'failed'
export type ShareMode = 'image' | 'text' | 'link'

export interface CardSize {
  width: number
  height: number
}

/** 9:16 for stories. */
export const STORY: CardSize = { width: 1080, height: 1920 }
/** 1:1 for feeds and chat previews, which crop anything taller. */
export const SQUARE: CardSize = { width: 1080, height: 1080 }

/* ---- Wording ------------------------------------------------------------------
 * The four functions that turn a call record into the sentences on the card.
 * Pure and exported so every one of the ten can be swept in a test: the card is
 * the one surface where a sentence that reads badly is seen by strangers first.
 */

/**
 * Dials whose number is money.
 *
 * Five of the ten carry an empty unit, because the readout prints the unit
 * straight after the figure and "3200$" is not a thing — but only four of those
 * five are dollars; the fifth counts the market's best days. Nothing in the
 * record separates them, so this does. Keyed by the variable's own name so that
 * renaming a dial fails a test rather than quietly printing "$4" for four days.
 */
const MONEY_DIALS = new Set(['toLowerRate', 'refund', 'repairSpend', 'expenseRatio'])

/** Whether this dial's number should be read as dollars. */
export function isMoneyDial(v: CallVariable): boolean {
  return v.unit === '' && MONEY_DIALS.has(v.key)
}

/** One position on a dial, as a noun: `6%`, `$3,000`, `owe $500`, `4`. */
export function dialText(value: number, v: CallVariable): string {
  if (!isMoneyDial(v)) return `${value}${v.unit}`
  // Only the refund dial goes below zero, and there a negative is not a
  // negative amount of money — it is a bill, and it reads as one.
  return value < 0 ? `owe ${money(-value)}` : money(value)
}

/**
 * The answer, in one phrase.
 *
 * A band does not have a single answer, it has an edge, and which edge matters
 * depends on where the band sits on the dial. Where a band runs to the end of
 * the track, naming the far end would invent a ceiling the maths does not have:
 * saving more than the employer match is not a mistake, and "6% to 15%" would
 * say it was. So a band that touches an end of the dial is stated as an open
 * one, and only a band with room on both sides is stated as a span.
 */
export function answerText(answer: Answer, v: CallVariable): string {
  if (typeof answer === 'number') return dialText(answer, v)
  if (answer.min <= v.min) return `${dialText(answer.max, v)} or less`
  if (answer.max >= v.max) return `${dialText(answer.min, v)} or more`
  return `${dialText(answer.min, v)} to ${dialText(answer.max, v)}`
}

/**
 * What the answer is measured in — the dial's own caption, read after the figure.
 *
 * Money captions open with the word "dollars" because the dial prints a bare
 * number above them. The card prints `$3`, so leaving the word in would produce
 * "$3, dollars a year per $10,000" — the unit said twice, which reads as a typo.
 *
 * One dial crosses zero, and its caption is written for the side it usually
 * lands on. The refund question captions "back at tax time", which is true of
 * a refund and a contradiction of a bill: the right answer there is to owe a
 * little, so the card was printing "owe $500, back at tax time". A caption
 * that cannot survive the answer is dropped rather than reworded, because the
 * phrase "owe $500" already says everything the caption was there to say.
 */
export function answerNote(v: CallVariable, answer?: Answer): string {
  const lowest = typeof answer === 'number' ? answer : answer?.min
  if (isMoneyDial(v) && lowest !== undefined && lowest < 0) return ''
  return isMoneyDial(v) ? v.label.replace(/^dollars\s+/i, '') : v.label
}

/**
 * The givens, minus anything the profile fills in.
 *
 * `fixed` values are templates: three of the ten interpolate the reader's own pay.
 * Those tiles are dropped rather than resolved, so the card cannot carry
 * personal financial data even if a future tile adds a new token — the test is
 * "does this contain a placeholder", not "is this the salary one". Every
 * question keeps at least one given, which registry.test-style sweeps here
 * confirm, so the card never loses its setup.
 */
export function sharedFacts(fixed: CallFact[]): CallFact[] {
  return fixed.filter((f) => !f.v.includes('{{'))
}

/* ---- Determinism ------------------------------------------------------------
 * Every irregular thing on the napkin is seeded, never random. The same card
 * must draw the same way every time it is rendered — on screen, in the PNG, and
 * after a reload — or the sender notices the paper change shape under them.
 */

/** FNV-1a. Small, fast, and — unlike a sum of char codes — sensitive to order. */
function hash32(s: string): number {
  let h = 0x811c9dc5
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i)
    h = Math.imul(h, 0x01000193)
  }
  return h >>> 0
}

/** xorshift32. Deterministic, no global state, uniform enough for decoration. */
function prng(seed: number): () => number {
  // 0 is xorshift's fixed point: it would return 0 forever and every napkin
  // seeded with it would tear in a perfectly straight line.
  let s = seed >>> 0 || 0x9e3779b9
  return () => {
    s ^= (s << 13) >>> 0
    s >>>= 0
    s ^= s >>> 17
    s ^= (s << 5) >>> 0
    s >>>= 0
    return s / 0x1_0000_0000
  }
}

function seedOf(seed: string | number): number {
  return hash32(String(seed))
}

/**
 * The seed for one napkin.
 *
 * Built from what is printed on it — the question and its answer — and not from
 * the sender. Two people sharing the same fact get the same paper, which is
 * correct now that the card is about the fact; and toggling the optional guess
 * on does not re-tear the paper under the sender's hands.
 */
export function napkinSeed(input: NapkinInput): string {
  return `${input.questionNo}:${input.answer}`
}

/* ---- Torn edge --------------------------------------------------------------- */

export interface TearPoint {
  /** 0..width, monotonically increasing. */
  x: number
  /** 0..1 — a fraction of the tear depth the caller chooses. */
  y: number
}

/**
 * The irregular edge of paper pulled off a stack.
 *
 * `y` is normalised so the same path serves a 10px CSS clip and a 26px canvas
 * one. Depth is deliberately not uniform: a tear is mostly shallow fibre with the
 * occasional deep bite, and drawing uniform noise instead gives a sawtooth — the
 * one thing this must not look like.
 */
export function tearPath(seed: string | number, width: number, segments: number): TearPoint[] {
  const n = Math.max(1, Math.floor(segments))
  const rnd = prng(seedOf(seed))
  const points: TearPoint[] = []

  for (let i = 0; i <= n; i++) {
    const t = i / n
    // Jitter is capped well under half a slot, which keeps x monotonic without a
    // sort and stops two points ever crossing into a bowtie.
    const jitter = i === 0 || i === n ? 0 : ((rnd() - 0.5) * 0.72) / n
    const r = rnd()
    points.push({
      x: (t + jitter) * width,
      y: r > 0.82 ? 0.62 + rnd() * 0.38 : r * 0.55,
    })
  }
  return points
}

/** Top and bottom tear differently, but both follow from the one napkin seed. */
export function tearSeeds(seed: string | number): { top: string; bottom: string } {
  return { top: `${seed}/top`, bottom: `${seed}/bottom` }
}

export const TEAR_SEGMENTS = 15

/**
 * A `clip-path` polygon for the DOM card: x in percent, y in pixels from the
 * nearest edge, so the card can be any height without stretching the tear.
 */
export function tearClipPath(seed: string | number, depth: number, segments = TEAR_SEGMENTS): string {
  const { top, bottom } = tearSeeds(seed)
  const pct = (x: number) => `${x.toFixed(3)}%`
  const px = (y: number) => `${(y * depth).toFixed(2)}px`

  const head = tearPath(top, 100, segments).map((p) => `${pct(p.x)} ${px(p.y)}`)
  const foot = tearPath(bottom, 100, segments)
    .reverse()
    .map((p) => `${pct(p.x)} calc(100% - ${px(p.y)})`)

  return `polygon(${[...head, ...foot].join(',')})`
}

/* ---- Pen stroke --------------------------------------------------------------- */

export const PEN_SEGMENTS = 28

export interface StrokePoint {
  /** 0..width, monotonically increasing. */
  x: number
  /** -1..1 — a fraction of the amplitude the caller chooses. */
  y: number
}

/**
 * The line someone draws under the number that matters.
 *
 * This is the napkin's answer to the old barcode: one irregular mark, derived
 * from what is on the card, so two different facts are visibly two different
 * objects and the same fact always redraws identically.
 *
 * A pen wanders rather than jumps, so each point steps from the last one with a
 * pull back toward the baseline — white noise at the same amplitude reads as a
 * zigzag, and an unpulled random walk sticks to one rail and reads as a wave.
 * Both ends taper to zero, because a stroke that stops mid-wobble looks cut off
 * rather than lifted.
 */
export function penStroke(
  seed: string | number,
  width: number,
  segments = PEN_SEGMENTS,
): StrokePoint[] {
  const n = Math.max(1, Math.floor(segments))
  const rnd = prng(seedOf(seed))
  const points: StrokePoint[] = []

  let y = 0
  for (let i = 0; i <= n; i++) {
    const t = i / n
    y = Math.max(-1, Math.min(1, y * 0.88 + (rnd() - 0.5) * 0.5))
    // 0 at both ends, ~1 across the middle: the pen coming down and lifting.
    // The ends are pinned rather than trusted to the curve — `Math.sin(Math.PI)`
    // is 1.2e-16, which taper turns into a visible-in-a-diff 5e-10 rather than
    // the zero the shape is defined by.
    const taper = i === 0 || i === n ? 0 : Math.pow(Math.sin(Math.PI * t), 0.6)
    // A negative wobble times a zero taper is -0, which prints as "-0.000" in
    // the path data and fails an Object.is check against the baseline.
    const wobble = y * taper
    points.push({ x: t * width, y: wobble === 0 ? 0 : wobble })
  }
  return points
}

/** `M0,1 L3.5,1.2 …` — the stroke as an SVG path, y centred on `mid`. */
export function penPathData(points: StrokePoint[], mid: number): string {
  return points
    .map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x.toFixed(2)},${(mid + p.y).toFixed(3)}`)
    .join(' ')
}

/* ---- Plain text ---------------------------------------------------------------
 * The share path that always works. Exactly five lines: any more and iMessage
 * collapses it behind a "read more", which is the same as not sending it.
 *
 * The sender's guess is not in here at all. The image can carry it in small
 * type where it reads as a footnote; five lines of chat text cannot, and a
 * whole line spent on "I said 18mo" would put the score back at the centre of
 * the thing this rewrite exists to take it out of.
 */

export function napkinText(input: NapkinInput, origin: string = DOMAIN): string {
  const handle = shareHandle(input.questionNo, origin)
  // The disclaimer rides with the invitation rather than taking a line of its
  // own, because the five are all spoken for and the fact is worth more than
  // the whitespace.
  const foot = handle
    ? `Guess it yourself at ${handle}. Estimates, not advice.`
    : 'Estimates, not advice.'

  return [
    `Napkin · question ${input.questionNo}`,
    input.scene,
    input.question,
    `The answer: ${input.answer}, ${input.note}.`,
    foot,
  ].join('\n')
}

/* ---- Canvas -------------------------------------------------------------------
 * The PNG is the same design drawn by hand. It reads its colours and font stacks
 * out of the stylesheet rather than restating them, so the token file stays the
 * single source of truth and a palette change cannot leave the share artifact
 * behind on the old one.
 */

type Palette = Record<'bg' | 'paper' | 'ink', string>
type Faces = Record<'display' | 'body' | 'data', string>

/** Blur and offset are proportional to the card, so both sizes lift the same. */
const SHADOW = { blur: 0.055, dy: 0.024, color: 'rgba(0,0,0,0.62)' }

/** The paper is never square to the frame. One degree, never two. */
const ROTATION = (1.05 * Math.PI) / 180

function readTokens(): { palette: Palette; faces: Faces } {
  const cs = getComputedStyle(document.documentElement)
  const v = (name: string) => cs.getPropertyValue(name).trim()

  const palette = { bg: v('--bg'), paper: v('--paper'), ink: v('--ink') }
  const faces = { display: v('--display'), body: v('--body'), data: v('--data') }

  if (!palette.paper || !faces.data) {
    // Canvas has no cascade to fall back through: an unresolved token silently
    // paints the previous fill, which would produce a black-on-black card.
    throw new Error('napkin: design tokens unavailable')
  }
  return { palette, faces }
}

/**
 * Monospace tracking, drawn per glyph.
 *
 * `ctx.letterSpacing` only landed in Safari 17.4 and this is the one property the
 * whole data style depends on — losing it collapses every label into an
 * unrecognisable block, so the tracking is applied by hand instead.
 */
function trackedWidth(ctx: CanvasRenderingContext2D, text: string, track: number): number {
  return ctx.measureText(text).width + track * Math.max(0, text.length - 1)
}

function drawTracked(
  ctx: CanvasRenderingContext2D,
  text: string,
  x: number,
  y: number,
  track: number,
  align: 'left' | 'right' = 'left',
): void {
  let cursor = align === 'right' ? x - trackedWidth(ctx, text, track) : x
  for (const ch of text) {
    ctx.fillText(ch, cursor, y)
    cursor += ctx.measureText(ch).width + track
  }
}

function wrap(ctx: CanvasRenderingContext2D, text: string, max: number, maxLines: number): string[] {
  const words = text.split(/\s+/)
  const lines: string[] = []
  let line = ''
  for (const w of words) {
    const next = line ? `${line} ${w}` : w
    if (ctx.measureText(next).width > max && line) {
      lines.push(line)
      line = w
      if (lines.length === maxLines) return lines
    } else {
      line = next
    }
  }
  if (line) lines.push(line)
  return lines.slice(0, maxLines)
}

/* Card-local geometry, in the natural units the design was drawn at. */
const CARD_W = 860
const PAD = 52
const TEAR_D = 26
const ROW_H = 46

/**
 * Set at the largest size that does not lose a word.
 *
 * `wrap` truncates silently, and a sentence clipped in the PNG is one the sender
 * never sees go missing. The smaller sizes are a floor rather than a design:
 * every real question fits the first size today, and they cost nothing.
 */
function fitText(
  ctx: CanvasRenderingContext2D,
  text: string,
  inner: number,
  sizes: number[],
  maxLines: number,
  font: (size: number) => string,
): { size: number; lineHeight: number; lines: string[] } {
  for (const size of sizes) {
    ctx.font = font(size)
    // One more line than allowed, purely to detect an overflow the cap would hide.
    const lines = wrap(ctx, text, inner, maxLines + 1)
    const last = size === sizes[sizes.length - 1]
    if (lines.length <= maxLines || last) {
      return { size, lineHeight: Math.round(size * 1.16), lines: lines.slice(0, maxLines) }
    }
  }
  throw new Error('napkin: no size fits')
}

function tornOutline(ctx: CanvasRenderingContext2D, seed: string, w: number, h: number): void {
  const { top, bottom } = tearSeeds(seed)
  ctx.beginPath()
  const head = tearPath(top, w, TEAR_SEGMENTS)
  head.forEach((p, i) => {
    const y = p.y * TEAR_D
    if (i === 0) ctx.moveTo(p.x, y)
    else ctx.lineTo(p.x, y)
  })
  for (const p of tearPath(bottom, w, TEAR_SEGMENTS).reverse()) {
    ctx.lineTo(p.x, h - p.y * TEAR_D)
  }
  ctx.closePath()
}

function dashedRule(ctx: CanvasRenderingContext2D, y: number, w: number, ink: string): void {
  ctx.save()
  ctx.globalAlpha = 0.3
  ctx.fillStyle = ink
  for (let x = PAD; x < w - PAD; x += 22) ctx.fillRect(x, y, 10, 4)
  ctx.restore()
}

/**
 * Draws one napkin into the card's own coordinate space and returns its height.
 * Called twice: once with `measure` to size the card, once to actually paint it.
 */
function paintCard(
  ctx: CanvasRenderingContext2D,
  input: NapkinInput,
  tokens: { palette: Palette; faces: Faces },
  measure: boolean,
): number {
  const { palette, faces } = tokens
  const w = CARD_W
  const inner = w - PAD * 2
  const font = (weight: number, size: number, face: keyof Faces) =>
    `${weight} ${size}px ${faces[face]}`

  const question = fitText(ctx, input.question, inner, [54, 47, 41], 3, (s) =>
    font(700, s, 'display'),
  )
  const scene = fitText(ctx, input.scene, inner, [30], 3, (s) => font(400, s, 'body'))
  const answer = fitText(ctx, input.answer, inner, [76, 64, 52], 2, (s) => font(700, s, 'display'))
  const note = fitText(ctx, input.note, inner, [28], 3, (s) => font(400, s, 'body'))
  const rule = fitText(ctx, input.rule, inner, [30], 3, (s) => font(400, s, 'body'))

  let y = TEAR_D + 54

  if (!measure) {
    // The wordmark is set in the data face, not the display one: it has to read
    // as a stamp on paper rather than as the first line of the card's copy.
    ctx.fillStyle = palette.ink
    ctx.font = font(700, 30, 'data')
    drawTracked(ctx, 'NAPKIN', PAD, y, 12)
    ctx.font = font(400, 24, 'data')
    ctx.globalAlpha = 0.55
    drawTracked(ctx, `QUESTION ${input.questionNo} · ${input.date}`, w - PAD, y, 3.4, 'right')
    ctx.globalAlpha = 1
  }

  y += 24
  if (!measure) dashedRule(ctx, y, w, palette.ink)

  y += 52
  if (!measure) {
    ctx.fillStyle = palette.ink
    ctx.globalAlpha = 0.62
    ctx.font = font(400, scene.size, 'body')
    scene.lines.forEach((l, i) => ctx.fillText(l, PAD, y + i * scene.lineHeight))
    ctx.globalAlpha = 1
  }
  y += (scene.lines.length - 1) * scene.lineHeight + 62

  if (!measure) {
    ctx.fillStyle = palette.ink
    ctx.font = font(700, question.size, 'display')
    question.lines.forEach((l, i) => ctx.fillText(l, PAD, y + i * question.lineHeight))
  }
  y += (question.lines.length - 1) * question.lineHeight + 50

  for (const given of input.givens) {
    if (!measure) {
      ctx.fillStyle = palette.ink
      ctx.font = font(400, 23, 'data')
      ctx.globalAlpha = 0.55
      drawTracked(ctx, given.k, PAD, y, 3.6)
      const labelEnd = PAD + trackedWidth(ctx, given.k, 3.6)

      ctx.font = font(700, 25, 'data')
      ctx.globalAlpha = 1
      drawTracked(ctx, given.v, w - PAD, y, 1.3, 'right')
      const valueStart = w - PAD - trackedWidth(ctx, given.v, 1.3)

      // Dotted leader, drawn as discrete squares rather than a dashed stroke so
      // the dots land on whole pixels at every scale.
      ctx.globalAlpha = 0.3
      for (let x = labelEnd + 14; x < valueStart - 14; x += 12) ctx.fillRect(x, y - 5, 5, 5)
      ctx.globalAlpha = 1
    }
    y += ROW_H
  }

  y += 22
  if (!measure) dashedRule(ctx, y, w, palette.ink)

  y += 46
  if (!measure) {
    ctx.fillStyle = palette.ink
    ctx.globalAlpha = 0.55
    ctx.font = font(400, 23, 'data')
    drawTracked(ctx, 'THE ANSWER', PAD, y, 3.6)
    ctx.globalAlpha = 1
  }

  y += answer.size + 10
  if (!measure) {
    ctx.fillStyle = palette.ink
    ctx.font = font(700, answer.size, 'display')
    answer.lines.forEach((l, i) => ctx.fillText(l, PAD, y + i * answer.lineHeight))
  }
  y += (answer.lines.length - 1) * answer.lineHeight + 26

  if (!measure) {
    // The stroke runs under the figure only, not the column: a pen underlines
    // what it is underlining. Measured in the face the figure was set in, which
    // the branch above has already selected.
    ctx.font = font(700, answer.size, 'display')
    const widest = answer.lines.reduce((a, l) => Math.max(a, ctx.measureText(l).width), 0)
    const strokeW = Math.min(inner, widest + 24)
    ctx.save()
    ctx.strokeStyle = palette.ink
    ctx.lineWidth = 7
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    ctx.globalAlpha = 0.85
    ctx.beginPath()
    for (const [i, p] of penStroke(napkinSeed(input), strokeW).entries()) {
      const py = y + p.y * 9
      if (i === 0) ctx.moveTo(PAD + p.x, py)
      else ctx.lineTo(PAD + p.x, py)
    }
    ctx.stroke()
    ctx.restore()
  }

  y += 42
  if (!measure) {
    ctx.fillStyle = palette.ink
    ctx.globalAlpha = 0.62
    ctx.font = font(400, note.size, 'body')
    note.lines.forEach((l, i) => ctx.fillText(l, PAD, y + i * note.lineHeight))
    ctx.globalAlpha = 1
  }
  y += (note.lines.length - 1) * note.lineHeight + 54

  if (!measure) dashedRule(ctx, y, w, palette.ink)

  y += 44
  if (!measure) {
    ctx.fillStyle = palette.ink
    ctx.globalAlpha = 0.55
    ctx.font = font(400, 23, 'data')
    drawTracked(ctx, 'THE RULE', PAD, y, 3.6)
    ctx.globalAlpha = 1
  }

  y += 42
  if (!measure) {
    ctx.fillStyle = palette.ink
    ctx.font = font(400, rule.size, 'body')
    rule.lines.forEach((l, i) => ctx.fillText(l, PAD, y + i * rule.lineHeight))
  }
  y += (rule.lines.length - 1) * rule.lineHeight + 48

  if (input.guess !== undefined) {
    if (!measure) {
      // Sentence case, unlike every other label on the card: the guess prints
      // a dial reading, and "8MO" is not how the answer two lines above it is
      // written. One value, two spellings, on the same piece of paper.
      ctx.fillStyle = palette.ink
      ctx.globalAlpha = 0.45
      ctx.font = font(400, 23, 'data')
      drawTracked(ctx, `I guessed ${input.guess}.`, PAD, y, 2.6)
      ctx.globalAlpha = 1
    }
    y += 42
  }

  if (!measure) {
    ctx.fillStyle = palette.ink
    ctx.globalAlpha = 0.45
    ctx.font = font(400, 21, 'data')
    drawTracked(ctx, 'ESTIMATES, NOT ADVICE.', PAD, y, 3)
    const handle = shareHandle(input.questionNo)
    if (handle) {
      ctx.globalAlpha = 0.8
      ctx.font = font(700, 21, 'data')
      drawTracked(ctx, handle.toUpperCase(), w - PAD, y, 3, 'right')
    }
    ctx.globalAlpha = 1
  }

  return y + 34 + TEAR_D
}

/**
 * Makes sure the faces this draw needs are actually in memory.
 *
 * `document.fonts.ready` is necessary and not sufficient: a `@font-face` is
 * fetched lazily, when something *rendered* asks for it, and a canvas draw is
 * not a render — so `ready` resolves perfectly happily having never fetched a
 * face that only the PNG uses, and the share goes out in Helvetica. Today the
 * card is on screen behind the share sheet and pulls the faces in for us, which
 * is exactly the kind of accident that holds until someone shares from anywhere
 * else. `load()` asks for them by name instead.
 */
async function loadFaces(faces: Faces, sizes: number[]): Promise<void> {
  const specs = Object.values(faces).flatMap((stack) =>
    sizes.flatMap((px) => [`400 ${px}px ${stack}`, `700 ${px}px ${stack}`]),
  )
  // A face that will not load is not a reason to refuse the share: the draw
  // still produces a legible card in the fallback stack.
  await Promise.all(specs.map((spec) => document.fonts.load(spec).catch(() => [])))
  await document.fonts.ready
}

/** Draws the napkin to a canvas and returns a PNG. */
export async function renderNapkin(input: NapkinInput, size: CardSize): Promise<Blob> {
  const tokens = readTokens()
  // One size per family is enough — a face is one file, not one file per size —
  // but both weights are asked for, because the labels and the figures differ.
  await loadFaces(tokens.faces, [64])

  const canvas = document.createElement('canvas')
  canvas.width = size.width
  canvas.height = size.height
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('napkin: 2d context unavailable')

  ctx.fillStyle = tokens.palette.bg
  ctx.fillRect(0, 0, size.width, size.height)

  const cardH = paintCard(ctx, input, tokens, true)

  // The card is sized to the frame rather than the frame to the card, so the
  // story and the square crops are the same drawing at two scales instead of two
  // layouts that can drift apart.
  const margin = size.width * 0.075
  const scale = Math.min((size.width - margin * 2) / CARD_W, (size.height - margin * 2) / cardH)

  ctx.save()
  ctx.translate(size.width / 2, size.height / 2)
  ctx.rotate(ROTATION)
  ctx.scale(scale, scale)
  ctx.translate(-CARD_W / 2, -cardH / 2)

  ctx.save()
  ctx.shadowColor = SHADOW.color
  // Shadow blur and offset are the one part of canvas state the transform does
  // not touch — they are output pixels — so both are scaled by hand. Scaling
  // only the offset kept the blur locked to the card's natural size and quietly
  // broke the lift on any frame the card had to shrink into.
  ctx.shadowBlur = CARD_W * SHADOW.blur * scale
  ctx.shadowOffsetY = CARD_W * SHADOW.dy * scale
  ctx.fillStyle = tokens.palette.paper
  tornOutline(ctx, napkinSeed(input), CARD_W, cardH)
  ctx.fill()
  ctx.restore()

  tornOutline(ctx, napkinSeed(input), CARD_W, cardH)
  ctx.clip()
  paintCard(ctx, input, tokens, false)
  ctx.restore()

  return await new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob)
      else reject(new Error('napkin: encoding failed'))
    }, 'image/png')
  })
}

/* ---- Sharing ------------------------------------------------------------------ */

/** The user dismissed the share sheet. Not a failure, and not a reason to copy. */
function isAbort(err: unknown): boolean {
  return err instanceof Error && err.name === 'AbortError'
}

async function copy(text: string): Promise<ShareOutcome> {
  try {
    await navigator.clipboard.writeText(text)
    return 'copied'
  } catch {
    return 'failed'
  }
}

async function shareAsText(text: string): Promise<ShareOutcome> {
  if (typeof navigator.share === 'function') {
    try {
      await navigator.share({ text })
      return 'shared'
    } catch (err) {
      if (isAbort(err)) return 'cancelled'
    }
  }
  return await copy(text)
}

/**
 * Out of the app, by whichever route the platform actually supports.
 *
 * Image first, because the card only works as a picture; text second, because
 * group chats strip images; clipboard last. Sharing *files* is a separate
 * capability from sharing text — `canShare({files})` has to be asked
 * specifically, and calling `share()` with a payload the platform does not
 * support throws rather than degrading.
 */
export async function shareReceipt(
  input: NapkinInput,
  mode: ShareMode = 'image',
  size: CardSize = SQUARE,
): Promise<ShareOutcome> {
  if (typeof navigator === 'undefined') return 'failed'

  if (mode === 'link') return await copy(shareLink(input.questionNo))
  if (mode === 'text') return await copy(napkinText(input))

  const text = napkinText(input)
  try {
    const blob = await renderNapkin(input, size)
    const file = new File([blob], `napkin-${input.questionNo}.png`, { type: 'image/png' })
    const payload = { files: [file] }
    if (typeof navigator.share === 'function' && navigator.canShare?.(payload)) {
      try {
        await navigator.share(payload)
        return 'shared'
      } catch (err) {
        // A dismissed sheet is a decision, not an error: silently copying the
        // text after someone backed out would be the app talking over them.
        if (isAbort(err)) return 'cancelled'
      }
    }
  } catch {
    // No canvas, no File constructor, or an encoder that gave up. The text path
    // below is the whole reason it exists.
  }

  return await shareAsText(text)
}
