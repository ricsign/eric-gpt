/**
 * Renders the app icon SVG into the PNG sizes iOS and Android need.
 *
 * Chromium is already in this environment for screenshot testing, so it doubles as
 * the rasteriser — one fewer native image dependency to install and pin.
 *
 * Run: node scripts/gen-icons.mjs
 */
import { launch } from './browser.mjs'
import { readFileSync, writeFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = dirname(fileURLToPath(import.meta.url))
const pub = resolve(__dirname, '../public')
const svg = readFileSync(resolve(pub, 'icon.svg'), 'utf8')

const TARGETS = [
  { file: 'icon-192.png', size: 192, inset: 0 },
  { file: 'icon-512.png', size: 512, inset: 0 },
  // Maskable icons are cropped to a circle on Android, so the artwork is inset to
  // keep it inside the safe zone.
  { file: 'icon-maskable.png', size: 512, inset: 0.1 },
  // iOS applies its own rounded mask and does not want transparency.
  { file: 'apple-touch-icon.png', size: 180, inset: 0 },
  { file: 'og.png', size: 0, width: 1200, height: 630, og: true },
]

const browser = await launch()
const page = await browser.newPage()

for (const t of TARGETS) {
  const width = t.width ?? t.size
  const height = t.height ?? t.size
  await page.setViewportSize({ width, height })

  const body = t.og
    ? `<div class="og">
         <div class="mark">${svg}</div>
         <div class="copy">
           <h1>Compound</h1>
           <p>One money call a day. Drag the number, watch your future move, take the receipt.</p>
         </div>
       </div>`
    : `<div class="icon" style="padding:${t.inset * 100}%">${svg}</div>`

  await page.setContent(`<!doctype html><html><head><style>
    *{margin:0;padding:0;box-sizing:border-box}
    body{width:${width}px;height:${height}px;overflow:hidden;
         font-family:-apple-system,'SF Pro Display','Segoe UI',Roboto,sans-serif}
    .icon{width:100%;height:100%;background:#08080A}
    .icon svg{width:100%;height:100%;display:block}
    .og{width:100%;height:100%;background:#08080A;color:#fff;
        display:flex;flex-direction:column;justify-content:center;gap:36px;padding:72px}
    .og .mark{width:220px}
    .og .mark svg{width:100%;height:auto;display:block}
    .og h1{font-size:84px;letter-spacing:-.03em;font-weight:800;text-transform:uppercase}
    .og p{font-size:30px;line-height:1.4;color:#8A8A90;max-width:26ch;margin-top:14px}
  </style></head><body>${body}</body></html>`)

  await page.screenshot({ path: resolve(pub, t.file), omitBackground: false })
  console.log(`wrote ${t.file} (${width}x${height})`)
}

await browser.close()
