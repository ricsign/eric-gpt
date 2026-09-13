/**
 * Downloads the latin subsets of the three brand faces into public/fonts.
 *
 * Self-hosted rather than linked, for three reasons: the app must work offline,
 * a third-party font request is a third-party request in a product whose whole
 * pitch is that nothing about you leaves the device, and a render-blocking
 * stylesheet on someone else's domain is the single slowest thing a first paint
 * can wait on.
 *
 * Only the `latin` subset is kept — the app is English-only and the latin-ext
 * subsets roughly double the payload for glyphs nothing ever renders.
 *
 * Run: node scripts/fetch-fonts.mjs
 */
import { writeFileSync, mkdirSync } from 'node:fs'
import { resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = dirname(fileURLToPath(import.meta.url))
const out = resolve(__dirname, '../public/fonts')
mkdirSync(out, { recursive: true })

// A modern UA is required or Google serves legacy TTF instead of woff2.
const UA =
  'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36'

const FAMILIES = [
  { css: 'Archivo+Black', file: 'archivo-black-400', weights: ['400'] },
  { css: 'Space+Grotesk:wght@400;500;700', file: 'space-grotesk', weights: ['400', '500', '700'] },
  { css: 'Courier+Prime:wght@400;700', file: 'courier-prime', weights: ['400', '700'] },
]

const faces = []

for (const family of FAMILIES) {
  const css = await (
    await fetch(`https://fonts.googleapis.com/css2?family=${family.css}&display=swap`, {
      headers: { 'User-Agent': UA },
    })
  ).text()

  // Each @font-face block is preceded by a /* subset */ comment; keep latin only.
  const blocks = css.split('/*').filter((b) => b.trim().startsWith('latin */'))

  for (const block of blocks) {
    const url = /url\((https:\/\/[^)]+\.woff2)\)/.exec(block)?.[1]
    const weight = /font-weight:\s*(\d+)/.exec(block)?.[1] ?? '400'
    const name = /font-family:\s*'([^']+)'/.exec(block)?.[1]
    if (!url || !name) continue

    const bytes = Buffer.from(await (await fetch(url)).arrayBuffer())
    const filename = `${family.file}${family.weights.length > 1 ? `-${weight}` : ''}.woff2`
    writeFileSync(resolve(out, filename), bytes)
    faces.push({ name, weight, filename, size: bytes.length })
    console.log(`${filename}  ${(bytes.length / 1024).toFixed(1)}kB`)
  }
}

// Emit the @font-face rules so the CSS can never drift from what was downloaded.
const css = `/*
 * GENERATED — do not edit. Produced by scripts/fetch-fonts.mjs.
 * Latin subsets only, self-hosted so the app works offline and makes no
 * third-party request.
 */
${faces
  .map(
    (f) => `@font-face {
  font-family: '${f.name}';
  font-style: normal;
  font-weight: ${f.weight};
  font-display: swap;
  src: url('/fonts/${f.filename}') format('woff2');
}`,
  )
  .join('\n\n')}
`

writeFileSync(resolve(__dirname, '../src/styles/fonts.css'), css)
console.log(`\nwrote src/styles/fonts.css (${faces.length} faces, ${(faces.reduce((s, f) => s + f.size, 0) / 1024).toFixed(1)}kB total)`)
