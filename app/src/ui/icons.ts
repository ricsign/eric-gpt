/**
 * Icon paths, drawn on a 24x24 grid to match SF Symbols' optical weight.
 *
 * Each tab icon comes in two weights — outline for inactive, filled for active —
 * because that weight change, not just a colour change, is what makes an iOS tab
 * bar read as native.
 */

export const ICONS = {
  // A rising curve rather than a clock: the tab is "today's move", and the mark
  // is the same one the product is named for.
  today: {
    outline:
      'M3.3 18.1a1 1 0 0 1-.6-1.8c3-2.2 5.2-4.7 7-7.2l2.2 2.6a1 1 0 0 0 1.5.05L19.6 6H16a1 1 0 1 1 0-2h5a1 1 0 0 1 1 1v5a1 1 0 1 1-2 0V7.5l-6 6.6a1 1 0 0 1-1.5-.04l-2.2-2.6c-1.7 2.3-3.8 4.6-6.5 6.5a1 1 0 0 1-.5.15Z',
    filled:
      'M21 4h-5a1 1 0 1 0 0 2h3.6l-6.2 6.75a1 1 0 0 1-1.5-.05l-2.2-2.6c-1.8 2.5-4 5-7 7.2a1 1 0 0 0 1.2 1.6c2.7-2 4.8-4.2 6.5-6.5l2.2 2.6a1 1 0 0 0 1.5.04L20 8.5V11a1 1 0 1 0 2 0V5a1 1 0 0 0-1-1Z',
  },
  tools: {
    outline:
      'M4 5a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V5Zm2 0v14h12V5H6Zm2 2h8v3H8V7Zm0 5h2v2H8v-2Zm3 0h2v2h-2v-2Zm3 0h2v5h-2v-5Zm-6 3h2v2H8v-2Zm3 0h2v2h-2v-2Z',
    filled:
      'M6 3a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V5a2 2 0 0 0-2-2H6Zm2 4h8v3H8V7Zm0 5h2v2H8v-2Zm3 0h2v2h-2v-2Zm3 0h2v5h-2v-5Zm-6 3h2v2H8v-2Zm3 0h2v2h-2v-2Z',
  },
  learn: {
    outline:
      'M12 3 2 8l10 5 8-4v5.5a1 1 0 1 0 2 0V8L12 3Zm0 2.24L17.53 8 12 10.76 6.47 8 12 5.24ZM6 12.2V16c0 1.5 2.7 3 6 3s6-1.5 6-3v-3.8l-2 1v2.6c-.3.4-1.8 1.2-4 1.2s-3.7-.8-4-1.2v-2.6l-2-1Z',
    filled:
      'M12 3 2 8l10 5 8-4v5.5a1 1 0 1 0 2 0V8L12 3ZM6 12.2V16c0 1.5 2.7 3 6 3s6-1.5 6-3v-3.8l-6 3-6-3Z',
  },
  you: {
    outline:
      'M12 4a3.5 3.5 0 1 1 0 7 3.5 3.5 0 0 1 0-7Zm0-2a5.5 5.5 0 1 0 0 11 5.5 5.5 0 0 0 0-11Zm0 12c-4.2 0-7.6 2.2-8.5 5.2A1 1 0 0 0 4.5 21h15a1 1 0 0 0 1-1.8C19.6 16.2 16.2 14 12 14Zm0 2c2.9 0 5.4 1.2 6.4 3H5.6c1-1.8 3.5-3 6.4-3Z',
    filled:
      'M12 2a5.5 5.5 0 1 0 0 11 5.5 5.5 0 0 0 0-11Zm0 12c-4.2 0-7.6 2.2-8.5 5.2A1 1 0 0 0 4.5 21h15a1 1 0 0 0 1-1.8C19.6 16.2 16.2 14 12 14Z',
  },
} as const

/** Single-path glyphs used inline. */
export const GLYPH = {
  flame:
    'M12 2s1 3.2-1.2 5.6C9 9.3 7 10.6 7 13.5A5 5 0 0 0 12 19a5 5 0 0 0 5-5.5c0-2.4-1.3-3.6-2.3-5.3-.5 1-1.2 1.6-1.9 2 .4-2.6-.3-6-.8-8.2Z',
  share:
    'M12 3a1 1 0 0 1 .7.3l3.5 3.5a1 1 0 0 1-1.4 1.4L13 6.4V15a1 1 0 1 1-2 0V6.4L9.2 8.2a1 1 0 0 1-1.4-1.4l3.5-3.5A1 1 0 0 1 12 3ZM5 13a1 1 0 0 1 1 1v5h12v-5a1 1 0 1 1 2 0v5a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2v-5a1 1 0 0 1 1-1Z',
  check: 'M20 6.5 9.5 17 4 11.5l1.5-1.5 4 4 9-9L20 6.5Z',
  info: 'M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20Zm0 4.5a1.3 1.3 0 1 1 0 2.6 1.3 1.3 0 0 1 0-2.6ZM11 11h2v7h-2v-7Z',
  chevron: 'M9 5l7 7-7 7',
  bolt: 'M13 2 4 14h6l-1 8 9-12h-6l1-8Z',
  lock: 'M12 2a5 5 0 0 0-5 5v3H6a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V11a1 1 0 0 0-1-1h-1V7a5 5 0 0 0-5-5Zm0 2a3 3 0 0 1 3 3v3H9V7a3 3 0 0 1 3-3Z',
  sliders:
    'M4 7h8a1 1 0 1 0 0-2H4a1 1 0 0 0 0 2Zm16-2h-2a1 1 0 1 0 0 2h2a1 1 0 1 0 0-2Zm-6 5H4a1 1 0 1 0 0 2h10a1 1 0 1 0 0-2Zm6 0h-2a1 1 0 1 0 0 2h2a1 1 0 1 0 0-2ZM8 15H4a1 1 0 1 0 0 2h4a1 1 0 1 0 0-2Zm12 0h-8a1 1 0 1 0 0 2h8a1 1 0 1 0 0-2Z',
} as const

/** Convenience for rendering a GLYPH path. */
export function iconPath(d: string, size = 20) {
  return { viewBox: '0 0 24 24', width: size, height: size, d }
}
