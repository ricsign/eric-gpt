/**
 * The receipt: the share artifact, and the only thing that leaves the app.
 *
 * Three renderings of one design — the DOM card in ui/Receipt.tsx, the PNG drawn
 * here, and the five-line plain text for group chats where images die. They must
 * agree, so everything that decides *shape* (the tear, the barcode, the stamp
 * wording, the code line) lives in this file as a pure function and is imported
 * by both renderers. A tear that differs between the screenshot and the PNG is a
 * bug nobody reports and everybody notices.
 *
 * Nothing here ever touches the profile. A receipt discloses a game result — a
 * call number, a position on a slider, a projection — and never personal
 * financial data. `ReceiptInput` deliberately has no salary field so that
 * property is structural rather than a rule someone has to remember.
 */

import type { BreakdownLine, Verdict } from '../calls/types'
import { money, moneyCompact } from './format'
import { urlForCall } from './route'

/** Bare domain. Appears in the share text with no path, no parameters, no tracking. */
export const DOMAIN = 'compound.day'

/** Everything a receipt needs. Assembled by the screen, never read from storage. */
export interface ReceiptInput {
  /** Stable forever: "No.142" identifies the call for everyone, in every timezone. */
  callNo: number
  /** Already formatted for display, e.g. `SEP 13`. */
  date: string
  title: string
  verdict: Verdict
  /** Where the player left the control. */
  value: number
  /** Rendered straight after the value. Empty for money variables. */
  unit: string
  breakdown: BreakdownLine[]
  /** The hero figure. */
  at65: number
  /** Signed: this play against the optimal play, at 65. */
  delta: number
}

export type ShareOutcome = 'shared' | 'copied' | 'cancelled' | 'failed'
export type ShareMode = 'image' | 'text' | 'link'

export interface ReceiptSize {
  width: number
  height: number
}

/** 9:16 for stories. */
export const STORY: ReceiptSize = { width: 1080, height: 1920 }
/** 1:1 for feeds and chat previews, which crop anything taller. */
export const SQUARE: ReceiptSize = { width: 1080, height: 1080 }

/* ---- Determinism ------------------------------------------------------------
 * Every irregular thing on the receipt is seeded, never random. Two players who
 * get the same result must get the same barcode, and one player who screenshots
 * the card and then shares the PNG must get the same tear on both.
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
  // 0 is xorshift's fixed point: it would return 0 forever and every receipt
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
 * The seed for one player's receipt.
 *
 * Built from the *result*, not the call: two people who played the same call
 * differently get visibly different barcodes, which is the entire reason the
 * barcode is on there.
 */
export function receiptSeed(input: ReceiptInput): string {
  return [
    input.callNo,
    input.value,
    Math.round(input.at65),
    Math.round(input.delta),
    input.verdict,
  ].join(':')
}

/* ---- Barcode ----------------------------------------------------------------- */

/** Bars plus spaces, alternating, starting and ending on a bar. */
export const BARCODE_MODULES = 47
export const BARCODE_MIN_WIDTH = 1
export const BARCODE_MAX_WIDTH = 4

/**
 * Widths, in module units, of the alternating bars and spaces — even indices are
 * ink, odd indices are paper. Not a real symbology: it encodes nothing scannable,
 * it is a fingerprint of the result that two different plays cannot share.
 */
export function barcodeWidths(seed: string | number): number[] {
  const rnd = prng(seedOf(seed))
  const span = BARCODE_MAX_WIDTH - BARCODE_MIN_WIDTH + 1
  const out: number[] = new Array(BARCODE_MODULES)
  for (let i = 0; i < BARCODE_MODULES; i++) {
    out[i] = BARCODE_MIN_WIDTH + Math.floor(rnd() * span)
  }
  return out
}

/** Total module count, so a renderer can size one module to the space it has. */
export function barcodeUnits(widths: number[]): number {
  return widths.reduce((a, b) => a + b, 0)
}

/**
 * The digits under the barcode. Call, position (in hundredths, so a 7.5% play
 * does not round away), and the projection — the three numbers that produced the
 * bars above them.
 *
 * The widths are sized to the widest real input, not to the median one: the
 * repair call runs to $4,000 (six digits once scaled) and a 22-year-old on the
 * top of the salary slider clears eight figures at 65. A field that grew a digit
 * would push COMPOUND.DAY off the paper, because the foot is one no-wrap flex
 * row inside a clip-path — so the slice truncates from the left rather than let
 * anything past those ranges reflow the line.
 */
export function receiptCode(input: ReceiptInput): string {
  const pad = (n: number, len: number) =>
    String(Math.abs(Math.round(n))).padStart(len, '0').slice(-len)
  return `${pad(input.callNo, 4)} ${pad(input.value * 100, 6)} ${pad(input.at65, 8)}`
}

/* ---- Torn edge --------------------------------------------------------------- */

export interface TearPoint {
  /** 0..width, monotonically increasing. */
  x: number
  /** 0..1 — a fraction of the tear depth the caller chooses. */
  y: number
}

/**
 * The irregular edge of a torn-off receipt.
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

/** Top and bottom tear differently, but both follow from the one receipt seed. */
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

/* ---- Wording ------------------------------------------------------------------ */

/**
 * The stamp. `over` is not a loss — on most calls it means the player pushed
 * past the optimal position rather than fell short of it — so it gets its own
 * word instead of being folded into the red one.
 */
export function stampText(verdict: Verdict): string {
  switch (verdict) {
    case 'optimal':
      return 'OPTIMAL PLAY'
    case 'short':
      return 'MONEY LEFT BEHIND'
    case 'over':
      return 'OVERSHOT'
  }
}

/**
 * How the delta figure is coloured.
 *
 * Taken from the sign of the money, never from the verdict. The stamp is a
 * judgement on the play and is red whenever the play was not the best one; the
 * figure beside it is a fact, and colouring an `over` play's very real loss in
 * neutral ink — which is what keying this off `verdict === 'short'` did — left
 * the two marks disagreeing about the same number.
 */
export type DeltaTone = 'loss' | 'gain' | 'even'

/** The line under the total: this play against the best one, at 65. */
export function deltaLine(input: ReceiptInput): {
  label: string
  value: string
  tone: DeltaTone
} {
  const label = 'VS OPTIMAL'
  // `money` rounds to the dollar, so a delta under fifty cents would print as
  // "−$0" — a loss of nothing, stated as a loss.
  const rounded = Math.round(input.delta)
  if (input.verdict === 'optimal' || rounded === 0) return { label, value: 'EVEN', tone: 'even' }
  // A true minus sign, not a hyphen: at this size a hyphen next to a dollar sign
  // reads as a dash in the label above it.
  const sign = rounded < 0 ? '−' : '+'
  return {
    label,
    value: `${sign}${money(Math.abs(rounded))}`,
    tone: rounded < 0 ? 'loss' : 'gain',
  }
}

/** `7%`, `$350`, `12 MONTHS` — the position, formatted the way the control showed it. */
export function playValue(input: ReceiptInput): string {
  const n = Number.isInteger(input.value) ? String(input.value) : input.value.toFixed(2)
  return input.unit ? `${n}${input.unit}` : money(input.value)
}

/** The line items, capped at what fits above the total without shrinking type. */
export const MAX_LINES = 6

export function receiptLines(input: ReceiptInput): BreakdownLine[] {
  // Several compute functions open their breakdown by stating the position in
  // their own units, which put the same number on the receipt twice under two
  // labels: "YOUR PLAY 63bps" sitting directly above "EXPENSE RATIO 63 BP".
  //
  // Where that happens the compute function's line wins and the generic one is
  // dropped, because "EXPENSE RATIO" says what the number is and "YOUR PLAY"
  // only says that it was chosen. The play line is prepended everywhere else,
  // since a receipt that never states what you picked is unreadable to whoever
  // you sent it to.
  const rest = input.breakdown.filter((l) => l.label.toUpperCase() !== 'YOUR PLAY')
  // The check runs on the lines that will actually be printed, not on the whole
  // breakdown. Asking the full list whether the position is stated and then
  // cutting the list to six can drop the very line that stated it, leaving a
  // receipt that never says what was chosen — which is the one thing it must say.
  const shown = rest.slice(0, MAX_LINES)
  if (shown.some((l) => statesPlay(l, input.value))) return shown
  const play: BreakdownLine = { label: 'YOUR PLAY', value: playValue(input) }
  return [play, ...rest].slice(0, MAX_LINES)
}

/**
 * Whether a breakdown line already tells the player where they left the control.
 *
 * The emphasised line is excluded: it is the receipt's punchline, so on a call
 * whose headline figure happens to equal the position it cannot also be read as
 * a restatement of it, or the receipt would lose the play entirely.
 */
function statesPlay(line: BreakdownLine, value: number): boolean {
  if (line.emphasis) return false
  // At the bottom of a control everything on the receipt is zero, so any line
  // would match and the play would be dropped for looking like a restatement —
  // leaving a receipt that never says the player chose nothing, which on the
  // match call is the most consequential answer available.
  if (value === 0) return false
  const first = line.value.match(/-?\d[\d,]*(?:\.\d+)?/)
  return first !== null && Number(first[0].replace(/,/g, '')) === value
}

/* ---- Plain text ---------------------------------------------------------------
 * The share path that always works. Exactly five lines: any more and iMessage
 * collapses it behind a "read more", which is the same as not sending it.
 */

export function receiptText(input: ReceiptInput): string {
  const head = `COMPOUND No.${input.callNo} · ${input.date}`
  const at65 = `${moneyCompact(input.at65)} at 65`

  const verdict =
    input.verdict === 'optimal'
      ? 'OPTIMAL PLAY · nothing left on the table'
      : input.verdict === 'short'
        ? `LEFT BEHIND · ${moneyCompact(Math.abs(input.delta))} by 65`
        : `OVERSHOT · ${moneyCompact(Math.abs(input.delta))} off the optimal play`

  return [
    head,
    input.title,
    `My play: ${playValue(input)} · ${at65}`,
    verdict,
    `Play No.${input.callNo} at ${DOMAIN}`,
  ].join('\n')
}

/* ---- Canvas -------------------------------------------------------------------
 * The PNG is the same design drawn by hand. It reads its colours and font stacks
 * out of the stylesheet rather than restating them, so the token file stays the
 * single source of truth and a palette change cannot leave the share artifact
 * behind on the old one.
 */

type Palette = Record<'bg' | 'paper' | 'ink' | 'loss', string>
type Faces = Record<'display' | 'data', string>

/** Blur and offset are proportional to the card, so both sizes lift the same. */
const SHADOW = { blur: 0.055, dy: 0.024, color: 'rgba(0,0,0,0.62)' }

/** The receipt is never square to the frame. One degree, never two. */
const ROTATION = (1.05 * Math.PI) / 180

function readTokens(): { palette: Palette; faces: Faces } {
  const cs = getComputedStyle(document.documentElement)
  const v = (name: string) => cs.getPropertyValue(name).trim()

  const palette = { bg: v('--bg'), paper: v('--paper'), ink: v('--ink'), loss: v('--loss') }
  const faces = { display: v('--display'), data: v('--data') }

  if (!palette.paper || !faces.data) {
    // Canvas has no cascade to fall back through: an unresolved token silently
    // paints the previous fill, which would produce a black-on-black receipt.
    throw new Error('receipt: design tokens unavailable')
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
const PAD = 48
const TEAR_D = 26
const BRAND_H = 84
const ROW_H = 54

/** Display sizes the title may be set at, largest first. */
const TITLE_SIZES = [44, 38, 33]
const TITLE_MAX_LINES = 3

/**
 * The title, at the largest size that does not lose a word.
 *
 * `wrap` truncates silently, and a title clipped in the PNG is a sentence the
 * sender never sees go missing. Twelve words fit three lines at 44px today, so
 * the smaller sizes are a floor rather than a design, and they cost nothing.
 */
function fitTitle(
  ctx: CanvasRenderingContext2D,
  title: string,
  inner: number,
  font: (weight: number, size: number, face: keyof Faces) => string,
): { size: number; lineHeight: number; lines: string[] } {
  for (const size of TITLE_SIZES) {
    ctx.font = font(700, size, 'display')
    // One more line than allowed, purely to detect an overflow the cap would hide.
    const lines = wrap(ctx, title, inner, TITLE_MAX_LINES + 1)
    const last = size === TITLE_SIZES[TITLE_SIZES.length - 1]
    if (lines.length <= TITLE_MAX_LINES || last) {
      return { size, lineHeight: Math.round(size * 1.14), lines: lines.slice(0, TITLE_MAX_LINES) }
    }
  }
  throw new Error('receipt: no title size')
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
  ctx.globalAlpha = 0.38
  ctx.fillStyle = ink
  for (let x = PAD; x < w - PAD; x += 22) ctx.fillRect(x, y, 10, 5)
  ctx.restore()
}

/**
 * Draws one receipt into the card's own coordinate space and returns its height.
 * Called twice: once with `measure` to size the card, once to actually paint it.
 */
function paintCard(
  ctx: CanvasRenderingContext2D,
  input: ReceiptInput,
  tokens: { palette: Palette; faces: Faces },
  measure: boolean,
): number {
  const { palette, faces } = tokens
  const w = CARD_W
  const inner = w - PAD * 2
  const font = (weight: number, size: number, face: keyof Faces) =>
    `${weight} ${size}px ${faces[face]}`

  const lines = receiptLines(input)
  const stamp = stampText(input.verdict)
  const delta = deltaLine(input)

  const title = fitTitle(ctx, input.title.toUpperCase(), inner, font)

  let y = TEAR_D + BRAND_H

  if (!measure) {
    // Brand lockup is a solid ink band with knockout type. At thumbnail size the
    // words are gone but the band survives, and the band is the logo.
    ctx.fillStyle = palette.ink
    ctx.fillRect(0, TEAR_D, w, BRAND_H)
    ctx.fillStyle = palette.paper
    ctx.font = font(700, 37, 'data')
    ctx.textBaseline = 'middle'
    const track = 20
    drawTracked(ctx, 'COMPOUND', (w - trackedWidth(ctx, 'COMPOUND', track)) / 2, TEAR_D + BRAND_H / 2, track)
    ctx.textBaseline = 'alphabetic'
  }

  y += 58
  if (!measure) {
    ctx.font = font(400, 26, 'data')
    ctx.globalAlpha = 0.6
    ctx.fillStyle = palette.ink
    drawTracked(ctx, `NO.${input.callNo}`, PAD, y, 3.4)
    drawTracked(ctx, input.date, w - PAD, y, 3.4, 'right')
    ctx.globalAlpha = 1
  }

  y += 26
  if (!measure) dashedRule(ctx, y, w, palette.ink)

  y += 54
  if (!measure) {
    ctx.fillStyle = palette.ink
    ctx.font = font(700, title.size, 'display')
    title.lines.forEach((l, i) => ctx.fillText(l, PAD, y + i * title.lineHeight))
  }
  y += (title.lines.length - 1) * title.lineHeight + 34

  if (!measure) dashedRule(ctx, y, w, palette.ink)
  y += 44

  for (const line of lines) {
    if (!measure) {
      const bold = line.emphasis === true
      ctx.fillStyle = palette.ink
      ctx.font = font(bold ? 700 : 400, 27, 'data')
      ctx.globalAlpha = bold ? 1 : 0.62
      drawTracked(ctx, line.label.toUpperCase(), PAD, y, 4.3)
      const labelEnd = PAD + trackedWidth(ctx, line.label.toUpperCase(), 4.3)

      ctx.font = font(bold ? 700 : 400, 32, 'data')
      ctx.globalAlpha = 1
      drawTracked(ctx, line.value, w - PAD, y, 1.3, 'right')
      const valueStart = w - PAD - trackedWidth(ctx, line.value, 1.3)

      // Dotted leader, drawn as discrete squares rather than a dashed stroke so
      // the dots land on whole pixels at every scale.
      ctx.globalAlpha = 0.34
      for (let x = labelEnd + 15; x < valueStart - 15; x += 12) ctx.fillRect(x, y - 5, 5, 5)
      ctx.globalAlpha = 1
    }
    y += ROW_H
  }

  y += 6
  if (!measure) {
    ctx.fillStyle = palette.ink
    ctx.fillRect(PAD, y, inner, 7)
    ctx.fillRect(PAD, y + 12, inner, 7)
  }

  y += 82
  if (!measure) {
    ctx.fillStyle = palette.ink
    ctx.font = font(400, 27, 'data')
    ctx.globalAlpha = 0.62
    drawTracked(ctx, 'TOTAL AT 65', PAD, y - 10, 4.3)
    ctx.globalAlpha = 1
    ctx.font = font(700, 92, 'display')
    ctx.fillText(money(input.at65), PAD, y + 74)
  }

  y += 120
  if (!measure) {
    ctx.font = font(400, 30, 'data')
    ctx.fillStyle = delta.tone === 'loss' ? palette.loss : palette.ink
    ctx.globalAlpha = delta.tone === 'loss' ? 1 : 0.62
    drawTracked(ctx, delta.label, PAD, y, 4.3)
    ctx.globalAlpha = 1
    drawTracked(ctx, delta.value, w - PAD, y, 1.3, 'right')
  }

  y += 112
  if (!measure) {
    const ink = input.verdict === 'optimal' ? palette.ink : palette.loss
    ctx.save()
    ctx.translate(w / 2, y)
    ctx.rotate((-2.6 * Math.PI) / 180)
    ctx.font = font(700, 33, 'data')
    const track = 7
    const text = `*** ${stamp} ***`
    const tw = trackedWidth(ctx, text, track)
    const boxW = tw + 64
    ctx.globalAlpha = 0.9
    ctx.fillStyle = ink
    ctx.fillRect(-boxW / 2, -46, boxW, 4)
    ctx.fillRect(-boxW / 2, -38, boxW, 4)
    ctx.fillRect(-boxW / 2, 34, boxW, 4)
    ctx.fillRect(-boxW / 2, 42, boxW, 4)
    ctx.textBaseline = 'middle'
    drawTracked(ctx, text, -tw / 2, 0, track)
    ctx.textBaseline = 'alphabetic'
    ctx.restore()
  }

  y += 96
  if (!measure) {
    const widths = barcodeWidths(receiptSeed(input))
    const unit = inner / barcodeUnits(widths)
    ctx.fillStyle = palette.ink
    let x = PAD
    widths.forEach((mods, i) => {
      const bw = mods * unit
      if (i % 2 === 0) ctx.fillRect(x, y, bw, 88)
      x += bw
    })
  }

  y += 88 + 36
  if (!measure) {
    ctx.fillStyle = palette.ink
    ctx.font = font(400, 27, 'data')
    ctx.globalAlpha = 0.6
    drawTracked(ctx, receiptCode(input), PAD, y, 4.3)
    ctx.globalAlpha = 1
    ctx.font = font(700, 27, 'data')
    drawTracked(ctx, DOMAIN.toUpperCase(), w - PAD, y, 4.3, 'right')
  }

  return y + 30 + TEAR_D
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
  // still produces a legible receipt in the fallback stack.
  await Promise.all(specs.map((spec) => document.fonts.load(spec).catch(() => [])))
  await document.fonts.ready
}

/**
 * Draws the receipt to a canvas and returns a PNG.
 */
export async function renderReceipt(input: ReceiptInput, size: ReceiptSize): Promise<Blob> {
  const tokens = readTokens()
  // One size per family is enough — a face is one file, not one file per size —
  // but both weights are asked for, because the labels and the figures differ.
  await loadFaces(tokens.faces, [92])

  const canvas = document.createElement('canvas')
  canvas.width = size.width
  canvas.height = size.height
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('receipt: 2d context unavailable')

  ctx.fillStyle = tokens.palette.bg
  ctx.fillRect(0, 0, size.width, size.height)

  const cardH = paintCard(ctx, input, tokens, true)

  // The card is sized to the frame rather than the frame to the card, so the
  // story and the square crops are the same drawing at two scales instead of two
  // layouts that can drift apart.
  const margin = size.width * 0.085
  const scale = Math.min(
    (size.width - margin * 2) / CARD_W,
    (size.height - margin * 2) / cardH,
  )

  ctx.save()
  ctx.translate(size.width / 2, size.height / 2)
  ctx.rotate(ROTATION)
  ctx.scale(scale, scale)
  ctx.translate(-CARD_W / 2, -cardH / 2)

  ctx.save()
  ctx.shadowColor = SHADOW.color
  // Shadow blur and offset are the one part of canvas state the transform does
  // not touch — they are output pixels — so both are scaled by hand. Scaling
  // only the offset, as this did, kept the blur locked to the card's natural
  // size and quietly broke the lift on any frame the card had to shrink into.
  ctx.shadowBlur = CARD_W * SHADOW.blur * scale
  ctx.shadowOffsetY = CARD_W * SHADOW.dy * scale
  ctx.fillStyle = tokens.palette.paper
  tornOutline(ctx, receiptSeed(input), CARD_W, cardH)
  ctx.fill()
  ctx.restore()

  tornOutline(ctx, receiptSeed(input), CARD_W, cardH)
  ctx.clip()
  paintCard(ctx, input, tokens, false)
  ctx.restore()

  return await new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob)
      else reject(new Error('receipt: encoding failed'))
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

async function shareText(text: string): Promise<ShareOutcome> {
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
 * Image first, because the receipt only works as a picture; text second, because
 * group chats strip images; clipboard last. Sharing *files* is a separate
 * capability from sharing text — `canShare({files})` has to be asked
 * specifically, and calling `share()` with a payload the platform does not
 * support throws rather than degrading.
 */
export async function shareReceipt(
  input: ReceiptInput,
  mode: ShareMode = 'image',
  size: ReceiptSize = SQUARE,
): Promise<ShareOutcome> {
  if (typeof navigator === 'undefined') return 'failed'

  if (mode === 'link') return await copy(urlForCall(input.callNo))
  if (mode === 'text') return await copy(receiptText(input))

  const text = receiptText(input)
  try {
    const blob = await renderReceipt(input, size)
    const file = new File([blob], `compound-${input.callNo}.png`, { type: 'image/png' })
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

  return await shareText(text)
}
