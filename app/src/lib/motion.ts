import type { Transition } from 'motion/react'

/**
 * The React-side twin of `scripts/gen-springs.mjs`.
 *
 * The CSS custom properties and these transitions describe the *same* five springs,
 * so a card that scales in CSS and a sheet that slides via Motion move with one
 * shared motion language. Keep the two in sync: same names, same numbers.
 *
 * `visualDuration` is Motion's perceptual-duration parameter and lines up with
 * SwiftUI's `.spring(duration:bounce:)`, which is what the generator models.
 */
export const spring = {
  tap: { type: 'spring', visualDuration: 0.25, bounce: 0 },
  smooth: { type: 'spring', visualDuration: 0.4, bounce: 0 },
  nav: { type: 'spring', visualDuration: 0.5, bounce: 0.08 },
  snappy: { type: 'spring', visualDuration: 0.42, bounce: 0.22 },
  bouncy: { type: 'spring', visualDuration: 0.55, bounce: 0.34 },
} satisfies Record<string, Transition>

/**
 * Rubber-banding, as UIScrollView does it.
 *
 * Past the boundary, travel is compressed logarithmically so the surface feels
 * attached to something rather than free. `dimension` is the size the resistance is
 * measured against (sheet height, screen width); `c` is Apple's empirical constant.
 */
export function rubberBand(offset: number, dimension: number, c = 0.55): number {
  if (offset === 0 || dimension === 0) return 0
  const sign = Math.sign(offset)
  const x = Math.abs(offset)
  return sign * (1 - 1 / ((x * c) / dimension + 1)) * dimension
}

/**
 * Where a flick would come to rest, given iOS's deceleration rate.
 *
 * Used to decide whether a drag should commit: a fast flick that has barely moved
 * should still dismiss, because the *projection* has cleared the threshold.
 */
export function projectFling(velocity: number, decelerationRate = 0.998): number {
  return (velocity / 1000) * (decelerationRate / (1 - decelerationRate))
}
