import { chromium } from 'playwright'
const b = await chromium.launch()
const p = await b.newPage({ viewport: { width: 1100, height: 900 }, deviceScaleFactor: 2 })
await p.goto('http://localhost:5199/__shot.html', { waitUntil: 'networkidle' })
await p.evaluate(() => document.fonts.ready)
await p.screenshot({ path: '/tmp/claude-0/-home-user-eric-gpt/1514f08f-d322-56ee-9858-88a12416deeb/scratchpad/receipt.png', fullPage: true })
await b.close()
