/**
 * Generates CSS `linear()` easing functions that reproduce real iOS spring physics.
 *
 * CSS transitions can only take an easing curve, not a physics simulation. `linear()`
 * closes that gap: we sample the analytic solution of a damped harmonic oscillator and
 * emit the samples as an easing. The result is visually indistinguishable from a
 * SwiftUI `.spring(response:bounce:)` for the same parameters, including the overshoot.
 *
 * Mapping from SwiftUI's modern spring API:
 *   response         = the natural period of the spring (seconds), omega = 2*pi/response
 *   bounce           = 1 - dampingFraction   (bounce 0 is critically damped, 0.3 is lively)
 *
 * Run: node scripts/gen-springs.mjs   (wired into `npm run build` via prebuild)
 */

import { writeFileSync, mkdirSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = dirname(fileURLToPath(import.meta.url))

/**
 * Unit-step response of a damped harmonic oscillator, normalised so f(0) = 0.
 * @param {number} t seconds
 * @param {number} omega natural angular frequency (rad/s)
 * @param {number} zeta damping ratio (1 = critically damped, <1 = overshoots)
 */
function springValue(t, omega, zeta) {
  if (zeta < 1) {
    const wd = omega * Math.sqrt(1 - zeta * zeta)
    return (
      1 -
      Math.exp(-zeta * omega * t) *
        (Math.cos(wd * t) + ((zeta * omega) / wd) * Math.sin(wd * t))
    )
  }
  if (zeta === 1) {
    return 1 - Math.exp(-omega * t) * (1 + omega * t)
  }
  // Overdamped — two real roots.
  const s = omega * Math.sqrt(zeta * zeta - 1)
  const r1 = -zeta * omega + s
  const r2 = -zeta * omega - s
  const c1 = -r2 / (r1 - r2)
  const c2 = r1 / (r1 - r2)
  return 1 - (c1 * Math.exp(r1 * t) + c2 * Math.exp(r2 * t))
}

/** Time (seconds) after which the spring stays within `epsilon` of 1 forever. */
function settlingTime(omega, zeta, epsilon = 0.001) {
  const dt = 1 / 1000
  let last = 0
  for (let t = 0; t < 10; t += dt) {
    if (Math.abs(springValue(t, omega, zeta) - 1) > epsilon) last = t
  }
  return Math.max(last + dt, 0.05)
}

/**
 * Samples the spring into a CSS `linear()` easing.
 * Samples are placed on a uniform time grid so the browser's linear interpolation
 * between them tracks the real curve; 60 steps is well inside the point budget and
 * keeps the error below a pixel at typical travel distances.
 */
function toLinearEasing(response, bounce, steps = 60) {
  const omega = (2 * Math.PI) / response
  const zeta = Math.max(0.05, 1 - bounce)
  const duration = settlingTime(omega, zeta)

  const stops = []
  for (let i = 0; i <= steps; i++) {
    const t = (i / steps) * duration
    const v = i === steps ? 1 : springValue(t, omega, zeta)
    stops.push(`${round(v)} ${round((i / steps) * 100)}%`)
  }
  return { easing: `linear(${stops.join(', ')})`, duration }
}

const round = (n) => {
  const r = Math.round(n * 1e4) / 1e4
  return Object.is(r, -0) ? 0 : r
}

/**
 * The app's spring vocabulary. Kept deliberately small — a consistent motion system
 * reads as "designed"; a dozen bespoke curves reads as noise.
 */
const SPRINGS = {
  // Buttons, toggles, chips. Fast, barely any overshoot.
  tap: { response: 0.25, bounce: 0.0 },
  // The default for anything that moves on screen.
  smooth: { response: 0.4, bounce: 0.0 },
  // Navigation pushes and sheet presentation — matches UIKit's default sheet feel.
  nav: { response: 0.5, bounce: 0.08 },
  // Things that should feel alive: number rolls, streak flames, celebration.
  snappy: { response: 0.42, bounce: 0.22 },
  // Big, delighted moments only. Used once or twice in the whole app.
  bouncy: { response: 0.55, bounce: 0.34 },
}

let css = `/*
 * GENERATED FILE — do not edit by hand.
 * Produced by scripts/gen-springs.mjs from real damped-harmonic-oscillator maths.
 * Each --ease-* is a CSS linear() sampling of a SwiftUI .spring(response:bounce:),
 * and each --dur-* is that spring's settling time, so pairing them reproduces iOS motion.
 */
:root {
`

for (const [name, { response, bounce }] of Object.entries(SPRINGS)) {
  const { easing, duration } = toLinearEasing(response, bounce)
  css += `  /* spring(response: ${response}, bounce: ${bounce}) */\n`
  css += `  --dur-${name}: ${Math.round(duration * 1000)}ms;\n`
  css += `  --ease-${name}: ${easing};\n`
}

css += `}

/*
 * Respect the system setting. Motion is how this app communicates growth, so rather
 * than deleting it we collapse every spring to a near-instant, non-overshooting move:
 * state changes stay legible, nothing travels or bounces.
 */
@media (prefers-reduced-motion: reduce) {
  :root {
${Object.keys(SPRINGS)
  .map((n) => `    --dur-${n}: 1ms;\n    --ease-${n}: linear;`)
  .join('\n')}
  }
}
`

const out = resolve(__dirname, '../src/styles/springs.css')
mkdirSync(dirname(out), { recursive: true })
writeFileSync(out, css)
console.log(`wrote ${out} (${Object.keys(SPRINGS).length} springs)`)
