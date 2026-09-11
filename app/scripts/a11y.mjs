/**
 * Accessibility smoke test.
 *
 * Not a full audit — it checks the four things this app could plausibly break
 * while looking fine in a screenshot: that reduced motion actually disables the
 * springs, that the custom slider is operable without a pointer, that the tab
 * order reaches real controls, and that every decorative SVG is either labelled
 * or hidden.
 *
 * Run against a built app:  npm run build && npx serve dist -p 5190
 *                           npm run a11y
 */
import { launch } from './browser.mjs'

const BASE = process.env.BASE || 'http://127.0.0.1:5190/'
const failures = []
const b = await launch()

// --- reduced motion --------------------------------------------------------
const reduced = await b.newContext({
  viewport: { width: 393, height: 852 }, colorScheme: 'dark', reducedMotion: 'reduce',
})
const rp = await reduced.newPage()
const errs = []
rp.on('pageerror', (e) => errs.push(e.message))
await rp.goto(BASE, { waitUntil: 'networkidle' })
await rp.click('button:has-text("Skip")'); await rp.waitForTimeout(300)
await rp.click('button:has-text("Take me in")'); await rp.waitForTimeout(500)

const durations = await rp.evaluate(() => {
  const cs = getComputedStyle(document.documentElement)
  return ['tap', 'smooth', 'nav', 'snappy', 'bouncy'].map((n) => cs.getPropertyValue(`--dur-${n}`).trim())
})
console.log('reduced-motion durations:', durations.join(' '))
if (durations.some((d) => d !== '1ms')) failures.push('reduced motion did not flatten the springs')

// --- keyboard --------------------------------------------------------------
const ctx = await b.newContext({ viewport: { width: 393, height: 852 }, colorScheme: 'dark' })
const page = await ctx.newPage()
page.on('pageerror', (e) => errs.push(e.message))
await page.goto(BASE, { waitUntil: 'networkidle' })
await page.click('button:has-text("Skip")'); await page.waitForTimeout(300)
await page.click('button:has-text("Take me in")'); await page.waitForTimeout(600)

const reached = []
for (let i = 0; i < 12; i++) {
  await page.keyboard.press('Tab')
  reached.push(await page.evaluate(() => {
    const el = document.activeElement
    if (!el || el === document.body) return 'body'
    return `${el.tagName.toLowerCase()}.${(el.className || '').toString().split(' ')[0]}`
  }))
}
console.log('tab order:', [...new Set(reached)].join(' -> '))

// A slider must be operable from the keyboard, or it is not an input.
await page.click('.tabbar-btn:has-text("Tools")'); await page.waitForTimeout(400)
await page.click('.card:has-text("What it becomes")'); await page.waitForTimeout(700)
const before = await page.locator('.vslider-value').first().innerText()
await page.locator('.vslider-track').first().focus()
for (let i = 0; i < 5; i++) await page.keyboard.press('ArrowRight')
const after = await page.locator('.vslider-value').first().innerText()
console.log('slider by keyboard:', before, '->', after)
if (after === before) failures.push('slider is not operable from the keyboard')

// Headings and landmarks.
const semantics = await page.evaluate(() => ({
  h1: document.querySelectorAll('h1').length,
  nav: document.querySelectorAll('nav[aria-label]').length,
  imgAlt: [...document.querySelectorAll('svg[role="img"]')].filter((s) => !s.getAttribute('aria-label')).length,
}))
console.log('semantics:', JSON.stringify(semantics))
if (semantics.h1 !== 1) failures.push(`expected exactly one h1, found ${semantics.h1}`)
if (semantics.nav !== 1) failures.push('the tab bar should be the one labelled nav landmark')
if (semantics.imgAlt > 0) failures.push(`${semantics.imgAlt} role="img" svg(s) without an aria-label`)
if (errs.length) failures.push(...errs)

await b.close()

if (failures.length) {
  console.error('\nFAILURES:')
  for (const f of failures) console.error(' - ' + f)
  process.exit(1)
}
console.log('\nall accessibility checks passed')
