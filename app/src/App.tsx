import { useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { AppShell } from './ui/AppShell'
import { TabBar, type TabDef } from './ui/TabBar'
import { ICONS } from './ui/icons'
import { NavStack } from './ui/NavStack'
import { ChromeProvider, useChrome } from './ui/Chrome'
import { useStore } from './state/store'
import { Onboarding } from './screens/Onboarding'
import { TodayScreen } from './screens/Today'
import { ToolsScreen } from './screens/Tools'
import { LearnScreen } from './screens/Learn'
import { YouScreen } from './screens/You'
import { dayKey } from './lib/format'
import { spring } from './lib/motion'
import { drillNumber } from './lib/daily'
import './App.css'

type TabId = 'today' | 'tools' | 'learn' | 'you'

export function App() {
  const { state } = useStore()

  if (!state.onboarded) {
    return (
      <AppShell>
        <Onboarding />
      </AppShell>
    )
  }

  return (
    <ChromeProvider>
      <TabbedApp />
    </ChromeProvider>
  )
}

function TabbedApp() {
  const { state } = useStore()
  const [tab, setTab] = useState<TabId>('today')
  const { pushedDepth } = useChrome()

  const todayDrill = state.drills[drillNumber(dayKey())]
  const tabs: TabDef<TabId>[] = [
    { id: 'today', label: 'Today', icon: ICONS.today, badge: todayDrill ? 0 : 1 },
    { id: 'tools', label: 'Tools', icon: ICONS.tools },
    { id: 'learn', label: 'Learn', icon: ICONS.learn },
    { id: 'you', label: 'You', icon: ICONS.you },
  ]

  return (
    <AppShell>
      <div className="app">
        {/*
          Tabs cross-fade rather than slide. Sliding implies a spatial relationship
          between tabs that does not exist — iOS does not animate tab changes either,
          and a fast fade reads as instant while still smoothing the repaint.
        */}
        <div className="app-body">
          <AnimatePresence mode="wait" initial={false}>
            <motion.div
              key={tab}
              className="app-page"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.14 }}
            >
              {/* Each tab owns its own navigation stack, so pushing a lesson from
                  Today and coming back finds Today exactly where it was. */}
              {tab === 'today' && <NavStack root={<TodayScreen />} />}
              {tab === 'tools' && <NavStack root={<ToolsScreen />} />}
              {tab === 'learn' && <NavStack root={<LearnScreen />} />}
              {tab === 'you' && <NavStack root={<YouScreen />} />}
            </motion.div>
          </AnimatePresence>
        </div>

        {/* Slides out when a screen is pushed, as UIKit does. Kept mounted and
            inert rather than unmounted, so the slide can animate both ways. */}
        <motion.div
          className="app-tabbar"
          animate={{ y: pushedDepth > 0 ? '110%' : '0%' }}
          transition={spring.nav}
          aria-hidden={pushedDepth > 0}
          inert={pushedDepth > 0}
        >
          <TabBar tabs={tabs} active={tab} onChange={setTab} />
        </motion.div>
      </div>
    </AppShell>
  )
}
