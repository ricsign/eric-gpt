import { useEffect, useState, type ReactNode } from 'react'
import './Device.css'

/**
 * The device shell.
 *
 * On a phone the app is full-bleed — nothing between the content and the glass.
 * On a desktop window it renders inside a 402x874 frame rather than stretching a
 * phone layout across 1600px. That is the honest presentation of a phone-first
 * product, and it is what makes the eventual Swift port a straight lift: the
 * layout is authored at exactly the dimensions it will ship at.
 *
 * The frame is decorative and aria-hidden; only the app inside it is in the
 * accessibility tree.
 */
export function Device({ children }: { children: ReactNode }) {
  const [framed, setFramed] = useState(false)

  useEffect(() => {
    // Frame only where there is genuinely room, and never on a touch device
    // that merely happens to be wide.
    const mq = window.matchMedia('(min-width: 900px) and (min-height: 800px) and (hover: hover)')
    const sync = () => setFramed(mq.matches)
    sync()
    mq.addEventListener('change', sync)
    return () => mq.removeEventListener('change', sync)
  }, [])

  if (!framed) return <div className="device device--bare">{children}</div>

  return (
    <div className="device device--framed">
      <div className="device-stage">
        <div className="device-body" aria-hidden="true" />
        <div className="device-screen">
          <div className="device-island" aria-hidden="true" />
          {children}
          <div className="device-home" aria-hidden="true" />
        </div>
      </div>
    </div>
  )
}
