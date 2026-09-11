import { useEffect, useState, type ReactNode } from 'react'
import './AppShell.css'

/**
 * The device shell.
 *
 * On a phone the app is simply full-bleed — no frame, no chrome, nothing between
 * the content and the glass. On a desktop window it renders inside an iPhone
 * bezel instead of stretching a phone layout across 1600px, which is both the
 * honest presentation of a phone-first product and a far better demo.
 *
 * The frame is decorative: it is aria-hidden, and the app inside it is the only
 * thing in the accessibility tree.
 */
export function AppShell({ children }: { children: ReactNode }) {
  const [framed, setFramed] = useState(false)

  useEffect(() => {
    // Frame only when there is genuinely room for it, and never on a touch device
    // that merely happens to be wide (an iPad in landscape should run full-bleed).
    const mq = window.matchMedia('(min-width: 900px) and (min-height: 780px) and (hover: hover)')
    const sync = () => setFramed(mq.matches)
    sync()
    mq.addEventListener('change', sync)
    return () => mq.removeEventListener('change', sync)
  }, [])

  if (!framed) return <div className="appshell appshell--bare">{children}</div>

  return (
    <div className="appshell appshell--framed">
      <div className="appshell-backdrop" aria-hidden="true" />

      <div className="device">
        <div className="device-frame" aria-hidden="true">
          <span className="device-btn device-btn--silence" />
          <span className="device-btn device-btn--up" />
          <span className="device-btn device-btn--down" />
          <span className="device-btn device-btn--power" />
        </div>

        <div className="device-screen">
          <div className="device-island" aria-hidden="true" />
          {children}
          <div className="device-home" aria-hidden="true" />
        </div>
      </div>

      <p className="appshell-note">
        Built phone-first. Open it on your iPhone and add it to your Home Screen —
        it runs full-screen, offline, with no browser chrome.
      </p>
    </div>
  )
}
