import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useReducer,
  type ReactNode,
} from 'react'
import { dayKey } from '../lib/format'
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

/**
 * One recorded prediction, from a curve draw or a slider estimate.
 *
 * This replaces the streak as the app's only metric-like object, and it survives
 * the self-determination-theory objections a streak does not: it is a competence
 * signal rather than a token, it measures the actual learning objective
 * (exponential-growth bias) rather than a proxy for attendance, it is falsifiable
 * unlike a composite score, it goes *down* rather than up so it is not a brag
 * ladder, and missing a day cannot break it, so it carries no loss frame.
 */
export interface Prediction {
  /** Concept the probe measured. */
  concept: string
  /** Absolute relative error, 0 = perfect. */
  error: number
  day: string
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
  /** Newest last. Capped, because this is a signal, not an audit log. */
  predictions: Prediction[]
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
  /**
   * The headline date as it stood last time the user looked.
   *
   * Kept so Today can show what moved — "4 months earlier than last time", with
   * the old date struck through. A date that visibly moves because of something
   * you did is the only thing on the home screen that earns a reopen in month six.
   */
  lastDate: { value: string; seenOn: string } | null
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
  predictions: [],
  settings: { theme: 'system', haptics: true, realTerms: false },
  saved: [],
  drills: {},
  commitments: [],
  jurisdiction: null,
  lastDate: null,
}

type Action =
  | { type: 'hydrate'; state: AppState }
  | { type: 'setProfile'; patch: Partial<Profile> }
  | { type: 'completeOnboarding' }
  | { type: 'completeLesson'; id: string; score: number; total: number }
  | { type: 'gradeConcept'; id: string; grade: Grade }
  | { type: 'toggleSaved'; id: string }
  | { type: 'setSettings'; patch: Partial<Settings> }
  | { type: 'setJurisdiction'; value: 'US' | 'other' }
  | { type: 'snapshotDate'; value: string }
  | { type: 'recordDrillAttempt'; number: number; correct: boolean }
  | { type: 'recordPrediction'; concept: string; error: number }
  | { type: 'commit'; commitment: Omit<Commitment, 'id' | 'createdAt'> }
  | { type: 'confirmCommitment'; id: string }
  | { type: 'reset' }

function reducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case 'hydrate':
      return action.state

    case 'setProfile':
      return { ...state, profile: { ...state.profile, ...action.patch } }

    case 'completeOnboarding':
      return { ...state, onboarded: true }

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

    case 'snapshotDate':
      if (state.lastDate?.value === action.value) return state
      return { ...state, lastDate: { value: action.value, seenOn: dayKey() } }

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

      return { ...state, drills: { ...state.drills, [action.number]: record } }
    }

    case 'recordPrediction':
      return {
        ...state,
        // 200 is well past the point where the median stops moving, and keeps the
        // persisted blob small enough to write on every change without cost.
        predictions: [
          ...state.predictions,
          { concept: action.concept, error: action.error, day: dayKey() },
        ].slice(-200),
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
  recordPrediction: (concept: string, error: number) => void
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
          predictions: stored.predictions ?? [],
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
  const recordPrediction = useCallback(
    (concept: string, error: number) => dispatch({ type: 'recordPrediction', concept, error }),
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
      recordPrediction,
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
      recordPrediction,
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

/**
 * Median absolute prediction error, and whether it is improving.
 *
 * Median rather than mean: one wild first guess would otherwise dominate the
 * number for weeks, and the point of showing it is that it visibly falls.
 */
export function calibration(predictions: Prediction[]): {
  median: number | null
  count: number
  /** Median over the most recent third, for the trend arrow. */
  recent: number | null
  improving: boolean
} {
  if (predictions.length === 0) return { median: null, count: 0, recent: null, improving: false }

  const med = (xs: number[]) => {
    if (!xs.length) return null
    const sorted = [...xs].sort((a, b) => a - b)
    const mid = Math.floor(sorted.length / 2)
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
  }

  const all = predictions.map((p) => p.error)
  const window = Math.max(3, Math.ceil(predictions.length / 3))
  const recentErrors = all.slice(-window)
  const earlyErrors = all.slice(0, Math.max(1, all.length - window))

  const median = med(all)
  const recent = med(recentErrors)
  const early = med(earlyErrors)

  return {
    median,
    count: predictions.length,
    recent,
    // Needs enough history for the comparison to mean anything.
    improving: predictions.length >= 6 && recent != null && early != null && recent < early,
  }
}

export { DEFAULT_STATE }
