import type { ReactNode } from 'react'
import './Card.css'

export function Card({
  children,
  tone = 'default',
  padded = true,
  onClick,
  className = '',
}: {
  children: ReactNode
  /** Tints the card for the two semantic cases the app leans on constantly. */
  tone?: 'default' | 'growth' | 'drag' | 'spark'
  padded?: boolean
  onClick?: () => void
  className?: string
}) {
  const Tag = onClick ? 'button' : 'div'
  return (
    <Tag
      className={`card card--${tone} ${padded ? 'card--padded' : ''} ${
        onClick ? 'card--tappable pressable' : ''
      } ${className}`}
      onClick={onClick}
      type={onClick ? 'button' : undefined}
    >
      {children}
    </Tag>
  )
}

/** A short all-caps label above a card's content. */
export function CardLabel({ children }: { children: ReactNode }) {
  return <p className="card-label">{children}</p>
}
