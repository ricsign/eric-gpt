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
// The home screen is the whole of onboarding now: what this is, and the ten
// questions. If a stranger cannot read the pitch here, nothing downstream
// matters — so this is the first thing the smoke test captures.
await shot('01-home')

const pitch = await page.locator('.home-pitch').first().textContent()
if (!pitch || !pitch.trim()) problems.push('home screen states no pitch')
const rows = await page.locator('.home-row').count()
if (rows !== 10) problems.push(`home lists ${rows} questions, expected 10`)

await tap('.home-row[data-state="open"]', 'question 1')

console.log('the question')
await shot('02-question')

// The question has to be on the screen, not inferred from a headline and a
// bar. Six reviewers could not tell what was being asked of them.
const q = await page.locator('.call-question').first().textContent()
if (!q || !q.trim().endsWith('?')) problems.push('question screen shows no question')

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
  await shot('03-dragged-max')
  for (let i = 20; i >= 8; i--) {
    await page.mouse.move(bar.x + 4 + (bar.width - 8) * (i / 20), y)
    await page.waitForTimeout(16)
  }
  await page.mouse.up()
  await shot('04-settled')
} else {
  problems.push('no .blockbar found — the control is the product, this is fatal')
}

await tap('.call-lock', 'answer')

console.log('the reveal')
await shot('05-reveal')
await page.evaluate(() => document.querySelector('.outcome')?.scrollTo(0, 500))
await shot('06-reveal-why')
await page.evaluate(() => document.querySelector('.outcome')?.scrollTo(0, 1100))
await shot('07-reveal-action')

// Nothing on the reveal may shout a six-figure loss in red. That single
// element is what made a stranger with $4,000 close the tab.
const body = await page.evaluate(() => document.body.innerText)
for (const banned of ['OVERSHOT', 'OPTIMAL PLAY', 'walked past by 65', 'of players']) {
  if (body.includes(banned)) problems.push(`reveal still says "${banned}"`)
}

await tap('.outcome-print, .outcome-share', 'share card')
console.log('the card')
await shot('08-card')

await tap('.share-next', 'next')
console.log('next')
await shot('09-next')

await tap('.tomorrow-link:has-text("list"), .tomorrow-link:has-text("List")', 'my list')
await shot('10-list')
await tap('.rules-action:has-text("Close")', 'close')

// Closing the list lands on home, which is where a returning player starts.
// Progress is reached from there, so that is the route the smoke test takes.
await tap('.home-link:has-text("Progress")', 'progress')
await shot('11-progress')

await browser.close()

if (problems.length) {
  console.error('\nPAGE PROBLEMS:')
  for (const p of [...new Set(problems)]) console.error(' - ' + p)
  process.exit(1)
}
console.log('\nno console errors')
