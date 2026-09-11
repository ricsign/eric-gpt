import { GLYPH } from './icons'

/** Renders one of the app's inline glyphs at a given size. */
export function Icon({
  name,
  size = 20,
  className = '',
  stroke = false,
}: {
  name: keyof typeof GLYPH
  size?: number
  className?: string
  /** Chevrons and similar are strokes, not fills. */
  stroke?: boolean
}) {
  return (
    <svg
      viewBox="0 0 24 24"
      width={size}
      height={size}
      className={className}
      aria-hidden="true"
      focusable="false"
    >
      <path
        d={GLYPH[name]}
        fill={stroke ? 'none' : 'currentColor'}
        stroke={stroke ? 'currentColor' : undefined}
        strokeWidth={stroke ? 2.4 : undefined}
        strokeLinecap={stroke ? 'round' : undefined}
        strokeLinejoin={stroke ? 'round' : undefined}
      />
    </svg>
  )
}
