import { motion } from 'motion/react'
import { haptic } from '../lib/haptics'
import { spring } from '../lib/motion'
import './TabBar.css'

export interface TabDef<T extends string> {
  id: T
  label: string
  /** Two paths: the outline for inactive, the filled variant for active — as SF Symbols do. */
  icon: { outline: string; filled: string }
  /** Small red count, e.g. reviews that are due. */
  badge?: number
}

export interface TabBarProps<T extends string> {
  tabs: TabDef<T>[]
  active: T
  onChange: (id: T) => void
}

/**
 * The bottom tab bar.
 *
 * Two details do most of the work of making this read as native: the icon swaps
 * between outline and filled weights rather than just changing colour, and the
 * label under the active tab is a heavier weight. Both are what iOS actually does,
 * and both are what web tab bars usually miss.
 */
export function TabBar<T extends string>({ tabs, active, onChange }: TabBarProps<T>) {
  return (
    <nav className="tabbar" aria-label="Main">
      <div className="tabbar-material" aria-hidden="true" />
      <ul className="tabbar-row">
        {tabs.map((tab) => {
          const isActive = tab.id === active
          return (
            <li key={tab.id} className="tabbar-item">
              <button
                className="tabbar-btn"
                data-active={isActive || undefined}
                aria-current={isActive ? 'page' : undefined}
                onClick={() => {
                  if (isActive) return
                  haptic('selection')
                  onChange(tab.id)
                }}
              >
                <span className="tabbar-icon-wrap">
                  <motion.svg
                    viewBox="0 0 24 24"
                    className="tabbar-icon"
                    aria-hidden="true"
                    animate={{ scale: isActive ? 1 : 0.94 }}
                    transition={spring.tap}
                  >
                    <path d={isActive ? tab.icon.filled : tab.icon.outline} />
                  </motion.svg>
                  {tab.badge != null && tab.badge > 0 && (
                    <span className="tabbar-badge" aria-hidden="true">
                      {tab.badge > 99 ? '99+' : tab.badge}
                    </span>
                  )}
                </span>
                <span className="tabbar-label">{tab.label}</span>
                {tab.badge != null && tab.badge > 0 && (
                  <span className="visually-hidden">{tab.badge} due</span>
                )}
              </button>
            </li>
          )
        })}
      </ul>
    </nav>
  )
}
