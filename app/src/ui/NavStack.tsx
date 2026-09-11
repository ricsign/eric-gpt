import { AnimatePresence, motion, useMotionValue, animate } from 'motion/react'
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react'
import { useChrome } from './Chrome'
import { projectFling, rubberBand, spring } from '../lib/motion'
import { haptic } from '../lib/haptics'
import './NavStack.css'

interface StackEntry {
  key: string
  render: () => ReactNode
}

interface NavApi {
  push: (key: string, render: () => ReactNode) => void
  pop: () => void
  popToRoot: () => void
  depth: number
}

const NavContext = createContext<NavApi | null>(null)

/** Access the enclosing navigation stack. Throws outside a NavStack, deliberately. */
export function useNav(): NavApi {
  const ctx = useContext(NavContext)
  if (!ctx) throw new Error('useNav must be used inside a <NavStack>')
  return ctx
}

/** Edge strip width, in px, that starts an interactive back gesture. */
const EDGE_WIDTH = 28

/**
 * A UINavigationController-style push/pop stack.
 *
 * The part web apps almost always omit is the *interactive* back gesture: on iOS,
 * dragging from the left edge moves both screens with your finger and can be
 * abandoned halfway. Without it, an app that otherwise looks native immediately
 * feels wrong to anyone who tries the gesture out of habit. So:
 *
 *  - Both layers track the drag, the outgoing screen at 1:1 and the one underneath
 *    at 30% parallax, which is UIKit's ratio.
 *  - Release commits on *projected* position, so a quick flick goes back even
 *    though the finger barely travelled.
 *  - Dragging back past the closed position rubber-bands rather than overshooting.
 */
export function NavStack({ root }: { root: ReactNode }) {
  const [stack, setStack] = useState<StackEntry[]>([])
  const stackId = useId()
  const { setDepth } = useChrome()
  const x = useMotionValue(0)
  const containerRef = useRef<HTMLDivElement>(null)
  const [dragging, setDragging] = useState(false)

  const drag = useRef({
    active: false,
    startX: 0,
    lastX: 0,
    lastT: 0,
    velocity: 0,
    /**
     * Captured at pointerdown rather than read during render.
     *
     * Reading `containerRef.current` while rendering gives null on the first pass,
     * and the old `window.innerWidth` fallback was simply wrong inside the desktop
     * phone frame — a 1600px window against a 393px stack meant the commit
     * threshold could never be reached by a real gesture.
     */
    width: 0,
  })

  const push = useCallback((key: string, render: () => ReactNode) => {
    haptic('light')
    x.set(0)
    setStack((s) => [...s, { key, render }])
  }, [x])

  const pop = useCallback(() => {
    setStack((s) => (s.length ? s.slice(0, -1) : s))
    x.set(0)
  }, [x])

  const popToRoot = useCallback(() => {
    setStack([])
    x.set(0)
  }, [x])

  const api = useMemo<NavApi>(
    () => ({ push, pop, popToRoot, depth: stack.length }),
    [push, pop, popToRoot, stack.length],
  )

  // Report depth up so the shell can slide the tab bar out of the way.
  useEffect(() => {
    setDepth(stackId, stack.length)
    return () => setDepth(stackId, 0)
  }, [stackId, stack.length, setDepth])

  const onPointerDown = (e: React.PointerEvent) => {
    if (stack.length === 0 || e.button !== 0) return
    const rect = containerRef.current?.getBoundingClientRect()
    if (!rect) return
    // Only the left edge starts a back gesture — anywhere else is content.
    if (e.clientX - rect.left > EDGE_WIDTH) return

    drag.current = {
      active: true,
      startX: e.clientX,
      lastX: e.clientX,
      lastT: e.timeStamp,
      velocity: 0,
      width: rect.width,
    }
    setDragging(true)
    ;(e.currentTarget as Element).setPointerCapture(e.pointerId)
  }

  const onPointerMove = (e: React.PointerEvent) => {
    const d = drag.current
    if (!d.active) return
    if (e.cancelable) e.preventDefault()

    const delta = e.clientX - d.startX
    // Forward (rightward) travel is real; pulling left past closed is resisted.
    x.set(delta >= 0 ? delta : rubberBand(delta, d.width))

    const dt = e.timeStamp - d.lastT
    if (dt > 0) {
      const instant = ((e.clientX - d.lastX) / dt) * 1000
      d.velocity = d.velocity * 0.7 + instant * 0.3
      d.lastX = e.clientX
      d.lastT = e.timeStamp
    }
  }

  const endDrag = () => {
    const d = drag.current
    if (!d.active) return
    d.active = false
    setDragging(false)

    const projected = x.get() + projectFling(d.velocity)
    if (projected > d.width * 0.4 || d.velocity > 550) {
      haptic('light')
      // Finish the slide out, then unmount — popping mid-travel snaps visibly.
      animate(x, d.width, { ...spring.nav, onComplete: pop })
    } else {
      animate(x, 0, spring.nav)
    }
  }

  const top = stack[stack.length - 1]
  const beneath = stack.length > 1 ? stack[stack.length - 2] : null

  return (
    <NavContext.Provider value={api}>
      <div
        className="navstack"
        ref={containerRef}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
      >
        {/* The layer underneath: root, or the screen below the top one. */}
        <motion.div
          className="navstack-layer navstack-layer--under"
          // Parallax at 30% of the top layer's travel, as UIKit does.
          style={{ x: dragging ? undefined : 0 }}
          animate={stack.length > 0 ? { x: '-22%', opacity: 0.72 } : { x: '0%', opacity: 1 }}
          transition={spring.nav}
          aria-hidden={stack.length > 0}
          // Covered screens must not be reachable by tab or VoiceOver swipe.
          inert={stack.length > 0}
        >
          {beneath ? beneath.render() : root}
        </motion.div>

        <AnimatePresence initial={false}>
          {top && (
            <motion.div
              key={top.key}
              className="navstack-layer navstack-layer--top"
              style={{ x }}
              initial={{ x: '100%' }}
              animate={{ x: 0 }}
              exit={{ x: '100%' }}
              transition={spring.nav}
            >
              {top.render()}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </NavContext.Provider>
  )
}

/** The standard leading nav-bar control: a chevron plus the word Back. */
export function BackButton({ label = 'Back' }: { label?: string }) {
  const { pop } = useNav()
  return (
    <button
      className="navbtn"
      onClick={() => {
        haptic('light')
        pop()
      }}
    >
      <svg viewBox="0 0 24 24" width="19" height="19" aria-hidden="true">
        <path
          d="M15 5l-7 7 7 7"
          fill="none"
          stroke="currentColor"
          strokeWidth="2.6"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
      {label}
    </button>
  )
}
