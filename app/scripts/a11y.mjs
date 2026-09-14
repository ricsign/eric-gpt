/**
 * Accessibility smoke test.
 *
 * Not a full audit — it checks the things this product could plausibly break
 * while still looking correct in a screenshot:
 *
 *  - prefers-reduced-motion actually flattens the springs
 *  - the drag control is operable without a pointer (a control that only
 *    responds to a finger is not an input, and this control IS the product)
 *  - the tab order reaches real controls
 *  - nothing decorative is announced, and nothing meaningful is silent
 *
 *   npm run build && npx serve dist -p 5190
 *   npm run a11y
 */
import { launch } from './browser.mjs'

const BASE = process.env.BASE || 'http://127.0.0.1:5190/'
const failures = []
const browser = await launch()

/* ---- reduced motion --------------------------------------------------------- */

const reduced = await browser.newContext({
  viewport: { width: 402, height: 874 },
  colorScheme: 'dark',
  reducedMotion: 'reduce',
})
const rp = await reduced.newPage()
rp.on('pageerror', (e) => failures.push('pageerror: ' + e.message))
await rp.goto(BASE, { waitUntil: 'networkidle' })

const durations = await rp.evaluate(() => {
  const cs = getComputedStyle(document.documentElement)
  return ['screen', 'tap'].map((n) => cs.getPropertyValue(`--dur-${n}`).trim())
})
console.log('reduced-motion durations:', durations.join(' '))
if (durations.some((d) => d !== '1ms')) {
  failures.push(`reduced motion did not flatten the springs: ${durations.join(' ')}`)
}

/* ---- keyboard --------------------------------------------------------------- */

const ctx = await browser.newContext({ viewport: { width: 402, height: 874 }, colorScheme: 'dark' })
const page = await ctx.newPage()
page.on('pageerror', (e) => failures.push('pageerror: ' + e.message))
await page.goto(BASE, { waitUntil: 'networkidle' })

// Home is the entry now. Every question in the set must be reachable by
// keyboard, and the open one must actually be operable.
const homeRows = await page.locator('.home-row').count()
console.log('home rows:', homeRows)
if (homeRows !== 10) failures.push(`home lists ${homeRows} questions, expected 10`)

const openRow = page.locator('.home-row[data-state="open"]').first()
await openRow.waitFor({ state: 'visible', timeout: 10000 })
await openRow.focus()
const focused = await page.evaluate(() => document.activeElement?.className ?? '')
if (!focused.includes('home-row')) failures.push('the open question cannot take keyboard focus')
// A locked row must be genuinely disabled, not just dimmed — a keyboard user
// tabbing into a button that silently does nothing has no way to know why.
const lockedEnabled = await page.evaluate(() =>
  [...document.querySelectorAll('.home-row[data-state="locked"]')].some((b) => !b.disabled),
)
if (lockedEnabled) failures.push('a locked question is focusable but does nothing')
await page.keyboard.press('Enter')
await page.waitForTimeout(600)

// The call control. This is the one that matters most.
const bar = page.locator('[role="slider"]').first()
await bar.waitFor({ state: 'visible', timeout: 10000 })
await bar.focus()
const callBefore = await page.locator('.call-readout').first().innerText()
for (let i = 0; i < 4; i++) await page.keyboard.press('ArrowRight')
const callAfter = await page.locator('.call-readout').first().innerText()
console.log('call control by keyboard:', callBefore.trim(), '->', callAfter.trim())
if (callBefore === callAfter) failures.push('the block bar is not keyboard-operable')

const aria = await page.evaluate(() => {
  const el = document.querySelector('[role="slider"]')
  if (!el) return null
  return {
    label: el.getAttribute('aria-label'),
    min: el.getAttribute('aria-valuemin'),
    max: el.getAttribute('aria-valuemax'),
    now: el.getAttribute('aria-valuenow'),
    text: el.getAttribute('aria-valuetext'),
    tabindex: el.getAttribute('tabindex'),
  }
})
console.log('slider aria:', JSON.stringify(aria))
for (const k of ['label', 'min', 'max', 'now']) {
  if (!aria || !aria[k]) failures.push(`slider is missing aria-${k === 'label' ? 'label' : `value${k}`}`)
}

// Locking in must be reachable by keyboard, or the loop dead-ends.
const reached = []
for (let i = 0; i < 14; i++) {
  await page.keyboard.press('Tab')
  reached.push(
    await page.evaluate(() => {
      const el = document.activeElement
      if (!el || el === document.body) return 'body'
      return `${el.tagName.toLowerCase()}.${(el.className || '').toString().split(' ')[0]}`
    }),
  )
}
console.log('tab order:', [...new Set(reached)].join(' -> '))
if (!reached.some((r) => r.includes('call-lock'))) {
  failures.push('the lock-in button is not reachable by keyboard')
}

/* ---- semantics --------------------------------------------------------------- */

const semantics = await page.evaluate(() => ({
  h1: document.querySelectorAll('h1').length,
  unlabelledImgSvg: [...document.querySelectorAll('svg[role="img"]')].filter(
    (s) => !s.getAttribute('aria-label'),
  ).length,
  // A decorative element that is announced is as bad as a meaningful one that
  // is not.
  ariaHiddenFocusable: [...document.querySelectorAll('[aria-hidden="true"]')].filter((el) =>
    el.querySelector('button, a, input, [tabindex]:not([tabindex="-1"])'),
  ).length,
}))
console.log('semantics:', JSON.stringify(semantics))
if (semantics.h1 !== 1) failures.push(`expected exactly one h1, found ${semantics.h1}`)
if (semantics.unlabelledImgSvg > 0) {
  failures.push(`${semantics.unlabelledImgSvg} role="img" svg(s) without an aria-label`)
}
if (semantics.ariaHiddenFocusable > 0) {
  failures.push(`${semantics.ariaHiddenFocusable} aria-hidden container(s) hold focusable content`)
}

await browser.close()

if (failures.length) {
  console.error('\nFAILURES:')
  for (const f of failures) console.error(' - ' + f)
  process.exit(1)
}
console.log('\nall accessibility checks passed')
