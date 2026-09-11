import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from 'react'
import { haptic, type HapticStyle } from '../lib/haptics'
import './Button.css'

type Variant = 'primary' | 'secondary' | 'tinted' | 'plain' | 'destructive'
type Size = 'sm' | 'md' | 'lg'

export interface ButtonProps extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, 'onClick'> {
  variant?: Variant
  size?: Size
  /** Stretches to the container. The default for anything in a footer. */
  block?: boolean
  loading?: boolean
  icon?: ReactNode
  /** Which haptic to fire. `null` opts out — use for anything that fires its own. */
  feedback?: HapticStyle | null
  onClick?: (e: React.MouseEvent<HTMLButtonElement>) => void
}

/**
 * The app's button.
 *
 * Haptics are wired in here rather than at every call site: a tap that produces no
 * physical response is the clearest possible signal that something is a web page.
 */
export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  {
    variant = 'primary',
    size = 'md',
    block = false,
    loading = false,
    icon,
    feedback = 'light',
    children,
    onClick,
    disabled,
    className = '',
    ...rest
  },
  ref,
) {
  return (
    <button
      ref={ref}
      className={`btn btn--${variant} btn--${size} ${block ? 'btn--block' : ''} ${className}`}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      onClick={(e) => {
        if (feedback) haptic(feedback)
        onClick?.(e)
      }}
      {...rest}
    >
      {loading ? (
        <span className="btn-spinner" aria-hidden="true" />
      ) : (
        <>
          {icon && <span className="btn-icon">{icon}</span>}
          {children}
        </>
      )}
    </button>
  )
})
