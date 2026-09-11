import type { ReactNode } from 'react'
import './List.css'

/**
 * The iOS inset-grouped list.
 *
 * A section is a rounded card; rows inside it are separated by hairlines that stop
 * short of the leading edge, aligned to where the row's text starts. That inset is
 * the detail that makes a list look like Settings rather than like a table.
 */
export function ListGroup({
  header,
  footer,
  children,
}: {
  header?: string
  /** Explanatory text below the group — where iOS puts the fine print. */
  footer?: ReactNode
  children: ReactNode
}) {
  return (
    <section className="listgroup">
      {header && <h2 className="listgroup-header">{header}</h2>}
      <div className="listgroup-body">{children}</div>
      {footer && <p className="listgroup-footer">{footer}</p>}
    </section>
  )
}

export function Row({
  icon,
  label,
  detail,
  value,
  accessory,
  onClick,
  destructive = false,
}: {
  icon?: ReactNode
  label: ReactNode
  /** Secondary line under the label. */
  detail?: ReactNode
  /** Trailing value, right-aligned and dimmed. */
  value?: ReactNode
  /** `chevron` for a navigating row, or any node (a switch, a checkmark). */
  accessory?: 'chevron' | ReactNode
  onClick?: () => void
  destructive?: boolean
}) {
  const Tag = onClick ? 'button' : 'div'
  return (
    <Tag
      className="row"
      data-interactive={onClick ? '' : undefined}
      data-destructive={destructive || undefined}
      onClick={onClick}
      type={onClick ? 'button' : undefined}
    >
      {icon && <span className="row-icon">{icon}</span>}
      <span className="row-text">
        <span className="row-label">{label}</span>
        {detail && <span className="row-detail">{detail}</span>}
      </span>
      {value != null && <span className="row-value num">{value}</span>}
      {accessory === 'chevron' ? (
        <svg className="row-chevron" viewBox="0 0 24 24" aria-hidden="true">
          <path
            d="M9 5l7 7-7 7"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.4"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      ) : (
        accessory
      )}
    </Tag>
  )
}
