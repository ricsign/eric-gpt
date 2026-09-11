import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from 'react'

/**
 * Tracks whether any navigation stack has a screen pushed.
 *
 * UIKit calls this `hidesBottomBarWhenPushed`: pushing a detail screen slides the
 * tab bar away, because a pushed screen owns the full height and its own bottom
 * action. Without it the tab bar floats over the pushed screen's footer and
 * swallows taps on it — which is exactly the bug this exists to fix.
 */
interface Chrome {
  pushedDepth: number
  setDepth: (stackId: string, depth: number) => void
}

const ChromeContext = createContext<Chrome>({ pushedDepth: 0, setDepth: () => {} })

export function ChromeProvider({ children }: { children: ReactNode }) {
  const [depths, setDepths] = useState<Record<string, number>>({})

  const setDepth = useCallback((stackId: string, depth: number) => {
    setDepths((d) => (d[stackId] === depth ? d : { ...d, [stackId]: depth }))
  }, [])

  const value = useMemo<Chrome>(
    () => ({
      // Inactive tabs are unmounted, so at most one stack reports a depth at a time.
      pushedDepth: Math.max(0, ...Object.values(depths)),
      setDepth,
    }),
    [depths, setDepth],
  )

  return <ChromeContext.Provider value={value}>{children}</ChromeContext.Provider>
}

export function useChrome(): Chrome {
  return useContext(ChromeContext)
}
