import { chromium } from 'playwright'
const b = await chromium.launch()
const p = await b.newPage({ viewport: { width: 1400, height: 900 }, deviceScaleFactor: 1 })
p.on('console', m => console.log('LOG', m.text()))
p.on('pageerror', e => console.log('ERR', e.message))
await p.goto('http://localhost:5199/__canvas.html', { waitUntil: 'networkidle' })
await p.waitForFunction(() => document.title === 'done', { timeout: 20000 })
await p.screenshot({ path: '/tmp/claude-0/-home-user-eric-gpt/1514f08f-d322-56ee-9858-88a12416deeb/scratchpad/canvas.png' })
await b.close()
