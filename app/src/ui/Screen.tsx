import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type ReactNode,
  type UIEvent,
} from 'react'
import './Screen.css'

export interface ScreenProps {
  /** The large title. Also used as the collapsed inline title. */
  title?: string
  /** Sits above the large title in small caps — context, not a second title. */
  eyebrow?: string
  /** Leading nav-bar slot: a back chevron, a close button. */
  left?: ReactNode
  /** Trailing nav-bar slot: one action, occasionally two. */
  right?: ReactNode
  /** Renders the title inline in the bar from the start, iOS "standard" style. */
  inlineTitle?: boolean
  /** Pinned above the tab bar, outside the scroll area. */
  footer?: ReactNode
  /** Skip the tab-bar-height bottom inset on screens presented without tabs. */
  noTabBar?: boolean
  children: ReactNode
}

/** Distance the large title travels before the inline title has fully taken over. */
const COLLAPSE_DISTANCE = 44

/**
 * One screen of the app: a fixed nav bar over a single scroll container.
 *
 * Reproduces the iOS large-title navigation bar, including the part most web
 * versions skip — the crossfade. As the large title scrolls under the bar, the
 * inline title fades in, and the bar's material and hairline fade in with it, so
 * the bar is invisible at rest and opaque the moment content passes beneath it.
 *
 * The scroll progress is written to a CSS custom property rather than to React
 * state, so the effect runs entirely in the compositor and does not re-render the
 * screen on every scroll frame.
 */
export function Screen({
  title,
  eyebrow,
  left,
  right,
  inlineTitle = false,
  footer,
  noTabBar = false,
  children,
}: ScreenProps) {
  const barRef = useRef<HTMLElement>(null)
  const [collapsed, setCollapsed] = useState(inlineTitle)
  const ticking = useRef(false)

  const apply = useCallback(
    (scrollTop: number) => {
      const progress = inlineTitle ? 1 : Math.min(1, Math.max(0, scrollTop / COLLAPSE_DISTANCE))
      barRef.current?.style.setProperty('--collapse', String(progress))
      // State only flips at the ends, so this re-renders twice per screen at most.
      setCollapsed((c) => {
        const next = progress > 0.92
        return c === next ? c : next
      })
    },
    [inlineTitle],
  )

  const onScroll = (e: UIEvent<HTMLDivElement>) => {
    const top = e.currentTarget.scrollTop
    if (ticking.current) return
    ticking.current = true
    requestAnimationFrame(() => {
      apply(top)
      ticking.current = false
    })
  }

  // Seed the CSS variable once, on mount. No setState here: `collapsed` is already
  // initialised to `inlineTitle`, which is its correct value at scrollTop 0.
  useEffect(() => {
    barRef.current?.style.setProperty('--collapse', inlineTitle ? '1' : '0')
  }, [inlineTitle])

  return (
    <section className="screen">
      <header className="navbar" ref={barRef} data-collapsed={collapsed}>
        <div className="navbar-material" aria-hidden="true" />
        <div className="navbar-row">
          <div className="navbar-slot navbar-slot--left">{left}</div>
          <div className="navbar-inline-title" aria-hidden={!collapsed}>
            {title}
          </div>
          <div className="navbar-slot navbar-slot--right">{right}</div>
        </div>
      </header>

      <div
        className="screen-scroll scroll"
        onScroll={onScroll}
        data-no-tabbar={noTabBar || undefined}
      >
        {title && !inlineTitle && (
          <div className="screen-titleblock">
            {eyebrow && <p className="screen-eyebrow">{eyebrow}</p>}
            {/* The visible large title. The bar's copy is aria-hidden until it takes
                over, so assistive tech only ever sees one heading. */}
            <h1 className="screen-title">{title}</h1>
          </div>
        )}
        {children}
      </div>

      {footer && <div className="screen-footer">{footer}</div>}
    </section>
  )
}
