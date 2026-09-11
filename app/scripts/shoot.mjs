/**
 * Drives the app on a simulated iPhone and captures each screen.
 *
 * Also fails the run on any console error or page exception, so this doubles as a
 * smoke test: a screenshot of a white screen is easy to miss, a thrown error is not.
 */
import { launch } from './browser.mjs'
import { mkdirSync } from 'node:fs'

const OUT = process.argv[2] || '/tmp/shots'
const BASE = process.env.BASE || 'http://127.0.0.1:5180'
mkdirSync(OUT, { recursive: true })

const browser = await launch()
const context = await browser.newContext({
  // iPhone 16 logical resolution, at device scale 3 for a crisp capture.
  viewport: { width: 393, height: 852 },
  deviceScaleFactor: 3,
  isMobile: true,
  hasTouch: true,
  userAgent:
    'Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1',
  colorScheme: process.env.THEME === 'light' ? 'light' : 'dark',
})

const problems = []
const page = await context.newPage()
page.on('console', (m) => {
  if (m.type() === 'error') problems.push(`console.error: ${m.text()}`)
})
page.on('pageerror', (e) => problems.push(`pageerror: ${e.message}`))

const shot = async (name) => {
  await page.waitForTimeout(650)
  await page.screenshot({ path: `${OUT}/${name}.png` })
  console.log(`  ${name}`)
}

const tap = async (selector, label) => {
  const el = page.locator(selector).first()
  await el.waitFor({ state: 'visible', timeout: 8000 })
  await el.click()
  if (label) console.log(`  tapped ${label}`)
  await page.waitForTimeout(450)
}

console.log('onboarding')
await page.goto(BASE, { waitUntil: 'networkidle' })
await shot('01-intro')

await tap('button:has-text("Start")')
await shot('02-age')
await tap('button:has-text("Next")')
await shot('03-saving')
await tap('button:has-text("Next")')
await shot('04-debt')
await tap('button:has-text("Next")')
await shot('05-jurisdiction')
await tap('button:has-text("United States")')
await shot('06-payoff')
await tap('button:has-text("Take me in")')

console.log('app')
await shot('07-today')

await tap('.today-section:has-text("One question") .card', 'drill card')
await shot('08-drill')

// Keep answering until the drill resolves, so the run works whichever position
// today's correct answer happens to sit in.
for (let i = 0; i < 4; i++) {
  const open = page.locator('.drill-option:not([disabled])')
  if ((await open.count()) === 0) break
  await open.first().click()
  await page.waitForTimeout(500)
  if (i === 0) await shot('09-drill-answered')
}
await shot('10-drill-result')
await tap('button:has-text("Today")', 'back')

console.log('lesson')
await tap('.today-section:has-text("Draw this one next") .card')
await shot('11-lesson-anchor')
await tap('button:has-text("Continue")')
await shot('12-lesson-probe')

// Exercise the wedge: draw a straight line across the canvas, which is the
// canonical wrong answer and the case the whole product is built around.
const canvas = page.locator('.curve-canvas')
if (await canvas.count()) {
  const box = await canvas.first().boundingBox()
  if (box) {
    const y0 = box.y + box.height * 0.92
    await page.mouse.move(box.x + 14, y0)
    await page.mouse.down()
    for (let i = 1; i <= 24; i++) {
      const f = i / 24
      await page.mouse.move(box.x + 14 + f * (box.width - 28), y0 - f * box.height * 0.5)
      await page.waitForTimeout(12)
    }
    await page.mouse.up()
    await page.waitForTimeout(300)
    await shot('12b-curve-drawn')
    await tap("button:has-text(\"That's my guess\")", 'commit curve')
    await page.waitForTimeout(1400)
    await shot('13-curve-revealed')
  }
}

const lockIn = page.locator('button:has-text("Lock it in")')
if (await lockIn.count()) {
  await lockIn.first().click()
  await page.waitForTimeout(900)
  await shot('13-lesson-probe-result')
}
await tap('button:has-text("Continue")')
await shot('14-lesson-reveal')
await tap('button:has-text("Continue")')
await shot('15-lesson-mechanism')
await tap('button:has-text("Continue")')
await shot('16-lesson-worked')
await tap('button:has-text("Continue")')
await shot('17-lesson-practice')
await tap('.lp-q-option >> nth=0')
await shot('18-lesson-practice-answered')
await tap('button:has-text("Continue")')
await shot('19-lesson-rule')
await tap('button:has-text("Continue")')
await shot('20-lesson-action')
await tap('.lp-footer-actions button >> nth=0')

console.log('tabs')
await shot('21-today-after')
await tap('.tabbar-btn:has-text("Tools")')
await shot('22-tools')
await tap('.card:has-text("What it becomes")')
await shot('23-tool-growth')
await tap('button:has-text("Tools")', 'back')
await tap('.card:has-text("Debt-free date")')
await shot('24-tool-debt')
await tap('button:has-text("Tools")', 'back')
await tap('.card:has-text("Marginal vs effective")')
await shot('25-tool-tax')
await tap('button:has-text("Tools")', 'back')

await tap('.tabbar-btn:has-text("Learn")')
await shot('26-learn')
await tap('.tabbar-btn:has-text("You")')
await shot('27-you')
await tap('.row:has-text("Sources")')
await shot('28-sources')

await browser.close()

if (problems.length) {
  console.error('\nPAGE PROBLEMS:')
  for (const p of [...new Set(problems)]) console.error(' - ' + p)
  process.exit(1)
}
console.log('\nno console errors')
