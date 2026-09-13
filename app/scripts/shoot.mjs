/**
 * Drives the whole loop on a simulated iPhone and captures every screen.
 *
 * Doubles as the smoke test: any console error or page exception fails the run.
 * A screenshot of a blank screen is easy to miss in a diff; a thrown exception
 * is not.
 *
 *   node scripts/shoot.mjs [outDir]
 *   BASE=https://… node scripts/shoot.mjs   # against a deployment
 *   THEME is ignored — the product has one theme by design.
 */
import { launch } from './browser.mjs'
import { mkdirSync } from 'node:fs'

const OUT = process.argv[2] || '/tmp/shots'
const BASE = process.env.BASE || 'http://127.0.0.1:5180'
mkdirSync(OUT, { recursive: true })

const browser = await launch()
const context = await browser.newContext({
  // The exact logical resolution the product is authored at.
  viewport: { width: 402, height: 874 },
  deviceScaleFactor: 3,
  isMobile: true,
  hasTouch: true,
  colorScheme: 'dark',
  userAgent:
    'Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1',
})

const problems = []
const page = await context.newPage()
page.on('console', (m) => m.type() === 'error' && problems.push(`console.error: ${m.text()}`))
page.on('pageerror', (e) => problems.push(`pageerror: ${e.message}`))

const shot = async (name) => {
  await page.waitForTimeout(500)
  await page.screenshot({ path: `${OUT}/${name}.png` })
  console.log(`  ${name}`)
}

const tap = async (selector, label) => {
  const el = page.locator(selector).first()
  await el.waitFor({ state: 'visible', timeout: 10000 })
  await el.click()
  if (label) console.log(`  tapped ${label}`)
  await page.waitForTimeout(400)
}

console.log('cold open')
await page.goto(BASE, { waitUntil: 'networkidle' })
await shot('01-salary')

// Drag the salary track rather than skipping, so the first-run gesture is
// exercised on every run.
const track = await page.locator('.salary-track').first().boundingBox()
if (track) {
  await page.mouse.move(track.x + track.width * 0.1, track.y + track.height / 2)
  await page.mouse.down()
  for (let i = 1; i <= 12; i++) {
    await page.mouse.move(track.x + track.width * (0.1 + (0.45 * i) / 12), track.y + track.height / 2)
    await page.waitForTimeout(14)
  }
  await page.mouse.up()
  await shot('02-salary-dragged')
}
await tap('.salary-go', 'start')

console.log('the call')
await shot('03-call')

// The mechanic. Drag the block bar across most of its range, then back, so the
// same-frame recompute and the block tinting are both visible in the capture.
const bar = await page.locator('.blockbar').first().boundingBox()
if (bar) {
  const y = bar.y + bar.height / 2
  await page.mouse.move(bar.x + 4, y)
  await page.mouse.down()
  for (let i = 1; i <= 20; i++) {
    await page.mouse.move(bar.x + 4 + (bar.width - 8) * (i / 20), y)
    await page.waitForTimeout(16)
  }
  await shot('04-call-dragged-max')
  for (let i = 20; i >= 8; i--) {
    await page.mouse.move(bar.x + 4 + (bar.width - 8) * (i / 20), y)
    await page.waitForTimeout(16)
  }
  await page.mouse.up()
  await shot('05-call-settled')
} else {
  problems.push('no .blockbar found — the control is the product, this is fatal')
}

await tap('.call-lock', 'lock it in')

console.log('outcome')
await shot('06-outcome')
await page.evaluate(() => document.querySelector('.outcome')?.scrollTo(0, 400))
await shot('07-outcome-crowd')

await tap('.outcome-print', 'print receipt')
console.log('receipt')
await shot('08-receipt')

await tap('.receipt-next', 'tomorrow')
console.log('tomorrow')
await shot('09-tomorrow')

await tap('.tomorrow-link:has-text("Tab")', 'the tab')
await shot('10-tab')
await tap('[data-close], .tab-close', 'close')

await tap('.tomorrow-link:has-text("Rules")', 'rules')
await shot('11-rules')

await browser.close()

if (problems.length) {
  console.error('\nPAGE PROBLEMS:')
  for (const p of [...new Set(problems)]) console.error(' - ' + p)
  process.exit(1)
}
console.log('\nno console errors')
