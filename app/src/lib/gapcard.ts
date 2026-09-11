import type { CurvePoint } from '../ui/CurveDraw'
import type { StrokePoint } from './curve'
import { describeError } from './curve'

/**
 * The Gap Card.
 *
 * The share artifact: your hand-drawn line, the real curve on top of it, and one
 * sentence — "I guessed 61% low." It contains **no personal financial data at
 * all**. Not a balance, not a salary, not a net worth. The scenario is stated in
 * neutral, universal terms so the recipient can attempt the identical curve.
 *
 * That constraint is the whole reason it can travel. Money is the most taboo
 * category of personal data on the internet: debt and salary are the two most
 * taboo topics in America, and roughly 30% of people would tell even a close
 * friend their bank balance. Every finance share artifact that requires
 * disclosing a figure demos beautifully and shares at approximately zero. This
 * one discloses a percentage about your *intuition*, which people share freely.
 *
 * Rendered on a canvas rather than fetched from a server: this app has no
 * backend, and a request-time image would be the wrong architecture anyway —
 * iMessage fetches link previews from the sender's device with no proxy, so
 * generated-on-demand previews time out on a cellular connection.
 */

export interface GapCardInput {
  /** The learner's stroke, in 0..1 canvas space. */
  stroke: StrokePoint[]
  truth: CurvePoint[]
  yMax: number
  /** Signed relative error at the endpoint. */
  endpointError: number
  /** The scenario in universal terms: "$300 a month for 30 years at 7%". */
  scenario: string
  /** Right-hand axis label, e.g. "30 years". */
  xLabel: string
  /** Bare domain, burned into the image. */
  domain?: string
}

/** 1080x1350 — the 4:5 portrait that survives every feed crop and screenshot. */
const W = 1080
const H = 1350

interface Palette {
  bg: string
  ink: string
  muted: string
  faint: string
  growth: string
  panel: string
}

const PALETTE: Palette = {
  bg: '#06120D',
  ink: '#FFFFFF',
  muted: 'rgba(235,235,245,0.62)',
  faint: 'rgba(235,235,245,0.3)',
  growth: '#30D98A',
  panel: 'rgba(255,255,255,0.045)',
}

/**
 * Draws the card and returns it as a PNG blob.
 * Returns null when canvas is unavailable, so callers fall back to text.
 */
export async function renderGapCard(input: GapCardInput): Promise<Blob | null> {
  if (typeof document === 'undefined') return null

  const canvas = document.createElement('canvas')
  canvas.width = W
  canvas.height = H
  const ctx = canvas.getContext('2d')
  if (!ctx) return null

  const p = PALETTE
  const font = (size: number, weight = 400) =>
    `${weight} ${size}px -apple-system, "SF Pro Display", "Segoe UI", Roboto, sans-serif`

  ctx.fillStyle = p.bg
  ctx.fillRect(0, 0, W, H)

  const M = 80

  // ---- Headline. The only text that must survive thumbnailing. -------------
  ctx.fillStyle = p.ink
  ctx.font = font(84, 700)
  ctx.textBaseline = 'top'
  const headline = describeError(input.endpointError).replace(/^I /, '').replace(/\.$/, '')
  ctx.fillText('I ' + headline, M, 150)

  // ---- Scenario, in universal terms. Never the sharer's own numbers. -------
  ctx.fillStyle = p.muted
  ctx.font = font(34, 400)
  wrapText(ctx, input.scenario, M, 268, W - M * 2, 44)

  // ---- The chart ------------------------------------------------------------
  const chart = { x: M, y: 400, w: W - M * 2, h: 560 }

  // The panel wraps the plot and its axis labels, but not the key — the key sat
  // exactly on the panel's bottom edge and read as a rendering mistake.
  ctx.fillStyle = p.panel
  roundRect(ctx, chart.x - 24, chart.y - 24, chart.w + 48, chart.h + 92, 32)
  ctx.fill()

  // Baseline.
  ctx.strokeStyle = p.faint
  ctx.lineWidth = 2
  ctx.beginPath()
  ctx.moveTo(chart.x, chart.y + chart.h)
  ctx.lineTo(chart.x + chart.w, chart.y + chart.h)
  ctx.stroke()

  // The learner's stroke: softer and thicker, so it reads as a sketch.
  ctx.strokeStyle = p.muted
  ctx.lineWidth = 12
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'
  ctx.beginPath()
  input.stroke.forEach((pt, i) => {
    const x = chart.x + pt.x * chart.w
    const y = chart.y + (1 - pt.y) * chart.h
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
  })
  ctx.stroke()

  // The truth.
  ctx.strokeStyle = p.growth
  ctx.lineWidth = 10
  ctx.beginPath()
  input.truth.forEach((pt, i) => {
    const x = chart.x + pt.t * chart.w
    const y = chart.y + chart.h - (Math.min(pt.value, input.yMax) / input.yMax) * chart.h
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
  })
  ctx.stroke()

  // Axis labels.
  ctx.fillStyle = p.faint
  ctx.font = font(26, 500)
  ctx.fillText('now', chart.x, chart.y + chart.h + 18)
  ctx.textAlign = 'right'
  ctx.fillText(input.xLabel, chart.x + chart.w, chart.y + chart.h + 18)
  ctx.textAlign = 'left'

  // Key, clear of the panel.
  const keyY = chart.y + chart.h + 106
  ctx.font = font(26, 500)
  drawKey(ctx, chart.x, keyY, p.muted, 'your line')
  drawKey(ctx, chart.x + 250, keyY, p.growth, 'what actually happens')

  // ---- Footer: the invitation, and the disclosure --------------------------
  ctx.fillStyle = p.ink
  ctx.font = font(38, 600)
  ctx.fillText('Draw yours', M, H - 190)

  ctx.fillStyle = p.growth
  ctx.font = font(34, 500)
  ctx.fillText(input.domain ?? 'compound.money', M, H - 140)

  // The disclosure lives inside the image, not on a page the recipient will never
  // load. A card circulating in a group chat carries no site footer with it.
  ctx.fillStyle = p.faint
  ctx.font = font(23, 400)
  ctx.fillText('Illustration on a stated assumption. Not financial advice.', M, H - 82)

  return new Promise((resolve) => canvas.toBlob((b) => resolve(b), 'image/png'))
}

function drawKey(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  color: string,
  label: string,
) {
  ctx.strokeStyle = color
  ctx.lineWidth = 8
  ctx.beginPath()
  ctx.moveTo(x, y + 14)
  ctx.lineTo(x + 40, y + 14)
  ctx.stroke()
  ctx.fillStyle = color
  ctx.fillText(label, x + 54, y)
}

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
) {
  ctx.beginPath()
  ctx.moveTo(x + r, y)
  ctx.arcTo(x + w, y, x + w, y + h, r)
  ctx.arcTo(x + w, y + h, x, y + h, r)
  ctx.arcTo(x, y + h, x, y, r)
  ctx.arcTo(x, y, x + w, y, r)
  ctx.closePath()
}

function wrapText(
  ctx: CanvasRenderingContext2D,
  text: string,
  x: number,
  y: number,
  maxWidth: number,
  lineHeight: number,
) {
  let line = ''
  let cursor = y
  for (const word of text.split(' ')) {
    const test = line ? `${line} ${word}` : word
    if (ctx.measureText(test).width > maxWidth && line) {
      ctx.fillText(line, x, cursor)
      line = word
      cursor += lineHeight
    } else {
      line = test
    }
  }
  if (line) ctx.fillText(line, x, cursor)
}

/**
 * The plain-text fallback, and the screen-reader alt text.
 *
 * Also what gets copied when neither file sharing nor the clipboard's image
 * support is available. Contains no dollar amount, by the same rule as the image.
 */
export function gapCardText(input: Pick<GapCardInput, 'endpointError' | 'scenario'>): string {
  return `${describeError(input.endpointError)}\n${input.scenario}\ncompound.money`
}

/**
 * Shares the card, preferring the native sheet with the image attached.
 *
 * Falls back through: share with file → share text → copy text. Each step is a
 * real degradation rather than an error, because the share sheet's availability
 * varies by browser, by permission, and by whether the gesture is still trusted.
 */
export async function shareGapCard(
  input: GapCardInput,
): Promise<'shared' | 'copied' | 'cancelled' | 'failed'> {
  const text = gapCardText(input)
  const blob = await renderGapCard(input)

  if (blob && typeof navigator !== 'undefined' && navigator.share) {
    const file = new File([blob], 'compound.png', { type: 'image/png' })
    // canShare must be consulted: sharing files is a separate capability from
    // sharing text, and calling share() with an unsupported payload throws.
    if (navigator.canShare?.({ files: [file] })) {
      try {
        await navigator.share({ files: [file], text })
        return 'shared'
      } catch (err) {
        if (err instanceof Error && err.name === 'AbortError') return 'cancelled'
      }
    }
  }

  if (typeof navigator !== 'undefined' && navigator.share) {
    try {
      await navigator.share({ text })
      return 'shared'
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') return 'cancelled'
    }
  }

  try {
    await navigator.clipboard.writeText(text)
    return 'copied'
  } catch {
    return 'failed'
  }
}
