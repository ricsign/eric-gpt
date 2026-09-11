import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useReducer,
  type ReactNode,
} from 'react'
import { dayKey, daysBetween } from '../lib/format'
import { read, write, clearAll } from '../lib/storage'
import { initialReview, review as reviewConcept, type Grade, type ReviewState } from '../lib/scheduler'
import { setHapticsEnabled } from '../lib/haptics'

/**
 * The user's own numbers.
 *
 * Every one of these is optional and self-reported. The app asks for the fewest
 * fields that let it compute something startling — there is no bank link, no
 * account, and nothing leaves the device. That is both the trust position and the
 * reason onboarding can be four taps instead of a signup flow.
 */
export interface Profile {
  age?: number
  /** Gross annual income. */
  income?: number
  /** Everything already invested or saved. */
  invested?: number
  /** What they can put away each month, today. */
  monthly?: number
  /** Highest-rate debt balance, if any. */
  debtBalance?: number
  debtApr?: number
  /** Age they would like to stop needing a salary. */
  targetAge?: number
  /** Expected annual return. Exposed so the app never hides its assumption. */
  assumedReturn: number
  /** Expected annual inflation, for showing values in today's money. */
  assumedInflation: number
}

export interface LessonProgress {
  completedAt: string
  /** Correct answers on first attempt, out of total questions. */
  score: number
  total: number
}

export interface Streak {
  current: number
  longest: number
  /** Day key of the most recent qualifying session. */
  lastDay: string | null
  /**
   * Unspent "rest days". Missing one day with a freeze available keeps the streak.
   * Two are granted up front and one is restored each full week of activity — enough
   * that a normal life does not reset the counter, few enough that it still means
   * something.
   */
  freezes: number
}

export interface Settings {
  theme: 'system' | 'light' | 'dark'
  haptics: boolean
  /** Values shown in today's purchasing power rather than future dollars. */
  realTerms: boolean
}

/** One day's attempt at the Daily Drill, keyed by drill number. */
export interface DrillRecord {
  /** true for a correct attempt, false for a wrong one, in order. */
  attempts: boolean[]
  solved: boolean
  /** Day key it was played, so the archive can show when. */
  day: string
}

export interface AppState {
  /** False until onboarding finishes. */
  onboarded: boolean
  profile: Profile
  lessons: Record<string, LessonProgress>
  reviews: Record<string, ReviewState>
  streak: Streak
  settings: Settings
  /** Concept ids the user has explicitly saved to revisit. */
  saved: string[]
  /** Daily Drill history, keyed by drill number. */
  drills: Record<number, DrillRecord>
  /**
   * Actions the user committed to at the end of a lesson, and whether they came
   * back and confirmed doing them. This — not lessons completed — is the number
   * the product is actually trying to move.
   */
  commitments: Commitment[]
  /** Which country's rules apply. Asked explicitly; never inferred from an IP address. */
  jurisdiction: 'US' | 'other' | null
}

export interface Commitment {
  id: string
  lessonId: string
  when: string
  then: string
  createdAt: string
  /** Set when the user confirms they did it. */
  doneAt?: string
  /** Dollar value the lesson computed for this action, if it could compute one honestly. */
  worth: number | null
}

const DEFAULT_STATE: AppState = {
  onboarded: false,
  profile: {
    // 7% nominal is the long-run US equity average after inflation is *not* removed;
    // the app states this assumption on every projection rather than burying it.
    assumedReturn: 0.07,
    assumedInflation: 0.025,
  },
  lessons: {},
  reviews: {},
  streak: { current: 0, longest: 0, lastDay: null, freezes: 2 },
  settings: { theme: 'system', haptics: true, realTerms: false },
  saved: [],
  drills: {},
  commitments: [],
  jurisdiction: null,
}

type Action =
  | { type: 'hydrate'; state: AppState }
  | { type: 'setProfile'; patch: Partial<Profile> }
  | { type: 'completeOnboarding' }
  | { type: 'completeLesson'; id: string; score: number; total: number }
  | { type: 'gradeConcept'; id: string; grade: Grade }
  | { type: 'recordActivity' }
  | { type: 'toggleSaved'; id: string }
  | { type: 'setSettings'; patch: Partial<Settings> }
  | { type: 'setJurisdiction'; value: 'US' | 'other' }
  | { type: 'recordDrillAttempt'; number: number; correct: boolean }
  | { type: 'commit'; commitment: Omit<Commitment, 'id' | 'createdAt'> }
  | { type: 'confirmCommitment'; id: string }
  | { type: 'reset' }

/**
 * Applies a day's activity to the streak.
 *
 * The freeze rule is the only interesting part: a single missed day is forgiven if
 * a freeze is available, because the alternative — a hard reset — is what makes
 * people abandon streak apps entirely after one bad week. Two missed days is a
 * genuine break and does reset.
 */
function advanceStreak(streak: Streak, today: string): Streak {
  if (streak.lastDay === today) return streak

  if (!streak.lastDay) {
    return { ...streak, current: 1, longest: Math.max(1, streak.longest), lastDay: today }
  }

  const gap = daysBetween(streak.lastDay, today)

  // A clock change or a device-time edit can produce a negative gap. Treat it as
  // same-day rather than letting it corrupt the count.
  if (gap <= 0) return streak

  let { current, freezes } = streak

  if (gap === 1) {
    current += 1
  } else if (gap === 2 && freezes > 0) {
    freezes -= 1
    current += 1
  } else {
    current = 1
  }

  // One freeze earned per full week of continuous activity, capped at two.
  if (current > 0 && current % 7 === 0) freezes = Math.min(2, freezes + 1)

  return {
    current,
    longest: Math.max(current, streak.longest),
    lastDay: today,
    freezes,
  }
}

function reducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case 'hydrate':
      return action.state

    case 'setProfile':
      return { ...state, profile: { ...state.profile, ...action.patch } }

    case 'completeOnboarding':
      return { ...state, onboarded: true, streak: advanceStreak(state.streak, dayKey()) }

    case 'completeLesson': {
      const today = dayKey()
      const existing = state.lessons[action.id]
      return {
        ...state,
        lessons: {
          ...state.lessons,
          // Keep the first completion date; a replay should not rewrite history,
          // but a better score should count.
          [action.id]: {
            completedAt: existing?.completedAt ?? today,
            score: Math.max(existing?.score ?? 0, action.score),
            total: action.total,
          },
        },
        streak: advanceStreak(state.streak, today),
      }
    }

    case 'gradeConcept': {
      const today = dayKey()
      const prev = state.reviews[action.id]
      return {
        ...state,
        reviews: {
          ...state.reviews,
          [action.id]: prev
            ? reviewConcept(prev, action.grade, today)
            : initialReview(action.grade, today),
        },
      }
    }

    case 'recordActivity':
      return { ...state, streak: advanceStreak(state.streak, dayKey()) }

    case 'toggleSaved':
      return {
        ...state,
        saved: state.saved.includes(action.id)
          ? state.saved.filter((s) => s !== action.id)
          : [...state.saved, action.id],
      }

    case 'setSettings':
      return { ...state, settings: { ...state.settings, ...action.patch } }

    case 'setJurisdiction':
      return { ...state, jurisdiction: action.value }

    case 'recordDrillAttempt': {
      const today = dayKey()
      const existing = state.drills[action.number]
      // A solved drill is final. Re-opening it must not let the result be rewritten,
      // or the shared glyph stops meaning anything.
      if (existing?.solved) return state

      const attempts = [...(existing?.attempts ?? []), action.correct]
      const record: DrillRecord = {
        attempts,
        solved: action.correct,
        day: existing?.day ?? today,
      }

      return {
        ...state,
        drills: { ...state.drills, [action.number]: record },
        // Showing up advances the streak, whether or not the answer was right.
        // Rewarding correctness is what turns practice into a performance.
        streak: advanceStreak(state.streak, today),
      }
    }

    case 'commit': {
      const today = dayKey()
      return {
        ...state,
        commitments: [
          ...state.commitments,
          {
            ...action.commitment,
            // Deterministic id: lesson plus day. Committing twice to the same action
            // on the same day should not create two rows to confirm.
            id: `${action.commitment.lessonId}:${today}`,
            createdAt: today,
          },
        ].filter(
          (c, i, all) => all.findIndex((o) => o.id === c.id) === i,
        ),
      }
    }

    case 'confirmCommitment':
      return {
        ...state,
        commitments: state.commitments.map((c) =>
          c.id === action.id ? { ...c, doneAt: dayKey() } : c,
        ),
      }

    case 'reset':
      clearAll()
      return DEFAULT_STATE

    default:
      return state
  }
}

interface Store {
  state: AppState
  dispatch: React.Dispatch<Action>
  /** Convenience wrappers, so screens never build action objects by hand. */
  setProfile: (patch: Partial<Profile>) => void
  completeLesson: (id: string, score: number, total: number) => void
  gradeConcept: (id: string, grade: Grade) => void
  setSettings: (patch: Partial<Settings>) => void
  recordDrillAttempt: (n: number, correct: boolean) => void
  commit: (c: Omit<Commitment, 'id' | 'createdAt'>) => void
  confirmCommitment: (id: string) => void
}

const StoreContext = createContext<Store | null>(null)

const STORAGE_KEY = 'state'

export function StoreProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, DEFAULT_STATE, () => {
    const stored = read<AppState | null>(STORAGE_KEY, null)
    // Merge rather than replace: a version of the app that adds a settings field
    // must not blow up on a profile saved by the previous one.
    return stored
      ? {
          ...DEFAULT_STATE,
          ...stored,
          profile: { ...DEFAULT_STATE.profile, ...stored.profile },
          settings: { ...DEFAULT_STATE.settings, ...stored.settings },
          streak: { ...DEFAULT_STATE.streak, ...stored.streak },
          drills: stored.drills ?? {},
          commitments: stored.commitments ?? [],
        }
      : DEFAULT_STATE
  })

  useEffect(() => {
    write(STORAGE_KEY, state)
  }, [state])

  // Settings that reach outside React.
  useEffect(() => {
    setHapticsEnabled(state.settings.haptics)
  }, [state.settings.haptics])

  useEffect(() => {
    const root = document.documentElement
    if (state.settings.theme === 'system') {
      root.removeAttribute('data-theme')
      try {
        localStorage.removeItem('compound.theme')
      } catch {
        /* storage unavailable; the attribute is already correct for this session */
      }
    } else {
      root.setAttribute('data-theme', state.settings.theme)
      try {
        localStorage.setItem('compound.theme', state.settings.theme)
      } catch {
        /* same */
      }
    }
  }, [state.settings.theme])

  const setProfile = useCallback((patch: Partial<Profile>) => dispatch({ type: 'setProfile', patch }), [])
  const completeLesson = useCallback(
    (id: string, score: number, total: number) => dispatch({ type: 'completeLesson', id, score, total }),
    [],
  )
  const gradeConcept = useCallback(
    (id: string, grade: Grade) => dispatch({ type: 'gradeConcept', id, grade }),
    [],
  )
  const setSettings = useCallback(
    (patch: Partial<Settings>) => dispatch({ type: 'setSettings', patch }),
    [],
  )
  const recordDrillAttempt = useCallback(
    (n: number, correct: boolean) => dispatch({ type: 'recordDrillAttempt', number: n, correct }),
    [],
  )
  const commit = useCallback(
    (c: Omit<Commitment, 'id' | 'createdAt'>) => dispatch({ type: 'commit', commitment: c }),
    [],
  )
  const confirmCommitment = useCallback(
    (id: string) => dispatch({ type: 'confirmCommitment', id }),
    [],
  )

  const value = useMemo<Store>(
    () => ({
      state,
      dispatch,
      setProfile,
      completeLesson,
      gradeConcept,
      setSettings,
      recordDrillAttempt,
      commit,
      confirmCommitment,
    }),
    [
      state,
      setProfile,
      completeLesson,
      gradeConcept,
      setSettings,
      recordDrillAttempt,
      commit,
      confirmCommitment,
    ],
  )

  return <StoreContext.Provider value={value}>{children}</StoreContext.Provider>
}

export function useStore(): Store {
  const ctx = useContext(StoreContext)
  if (!ctx) throw new Error('useStore must be used inside a <StoreProvider>')
  return ctx
}

export { DEFAULT_STATE, advanceStreak }
