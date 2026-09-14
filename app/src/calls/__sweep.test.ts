import { describe, it } from 'vitest'
import { writeFileSync } from 'node:fs'
import { COMPUTE } from './compute'
import { DEFAULT_PROFILE } from './types'

const out: string[] = []
const P = DEFAULT_PROFILE
const sweep = (name: string, min: number, max: number, step: number, profile = P) => {
  const rows: [number, number, number, number][] = []
  for (let v = min; v <= max + 1e-9; v += step) {
    const o = COMPUTE[name](v, profile)
    rows.push([v, o.at65, o.cost, o.benefit])
  }
  const best = rows.reduce((a, b) => (b[1] > a[1] ? b : a))
  out.push(`== ${name} (sal ${profile.salary}) ==  argmax at65 = ${best[0]}  (${best[1].toFixed(0)})`)
  out.push(rows.map(r => `  ${r[0]}: at65=${r[1].toFixed(1)} cost=${r[2].toFixed(2)} ben=${r[3].toFixed(1)}`).join('\n'))
}

describe('sweep', () => {
  it('runs', () => {
    sweep('employerMatch', 0, 15, 1)
    sweep('debtSplit', 0, 500, 25)
    sweep('emergencyFund', 0, 12, 1)
    sweep('promoDeadline', 1, 24, 1)
    sweep('anchorOffer', 0, 20, 1)
    sweep('rothSplit', 0, 100, 5)
    sweep('rothSplit', 0, 100, 5, { salary: 150_000, age: 30 })
    sweep('rothSplit', 0, 100, 5, { salary: 30_000, age: 30 })
    sweep('repairOrReplace', 0, 4000, 100)
    sweep('rentVsBuy', 0, 15, 1)
    sweep('timingMarket', 0, 30, 1)
    writeFileSync('/tmp/sweep.txt', out.join('\n'))
  })
})
