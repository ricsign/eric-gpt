import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import type { ActionState, CallResult, Profile } from '../calls/types'
import { DEFAULT_PROFILE } from '../calls/types'

/**
 * All of it lives on the device.
 *
 * There is no account, no server and nothing to sign up for. That is a product
 * position as much as a technical one: a money app that asks who you are before
 * it has shown you anything loses most of its funnel, and one that never
 * uploads your salary is one you can trust on first use. Auth is offered only
 * after the first receipt exists, as "save your tab".
 */

export interface Settings {
  /** Local hour the daily notification fires. */
  notifyHour: number
  /** Asked only after the third completed call — never on first run. */
  notifyAsked: boolean
  haptics: boolean
}

interface State {
  /** Null until the first-run salary question is answered or skipped. */
  profile: Profile | null
  /** Every locked-in answer, newest last. Immutable once written. */
  results: CallResult[]
  /**
   * Values this device has observed per call, as a histogram.
   *
   * Stands in for a real aggregate until there is a backend. Today it is
   * essentially the player's own answer, blended against the authored seed
   * distribution — see lib/crowd.ts for how the weighting decays.
   */
  localCrowd: Record<number, number[]>
  /**
   * Where the player is with each question's "do this week" errand.
   *
   * Separate from `results` because a result is a fact about a moment and is
   * immutable once written, while an action is a live piece of state the
   * player edits for as long as it matters to them.
   */
  actions: Record<number, ActionState>
  settings: Settings

  setProfile: (p: Profile) => void
  /** Records an answer. A repeat of a call already played is ignored. */
  lockIn: (result: CallResult, steps: number) => void
  observeCrowd: (callId: number, bucket: number, steps: number) => void
  setAction: (callId: number, state: ActionState) => void
  setSettings: (patch: Partial<Settings>) => void
  reset: () => void
}

const INITIAL = {
  profile: null as Profile | null,
  results: [] as CallResult[],
  localCrowd: {} as Record<number, number[]>,
  actions: {} as Record<number, ActionState>,
  settings: { notifyHour: 8, notifyAsked: false, haptics: true } satisfies Settings,
}

export const useStore = create<State>()(
  persist(
    (set, get) => ({
      ...INITIAL,

      setProfile: (profile) => set({ profile }),

      setAction: (callId, state) =>
        set((s) => ({ actions: { ...s.actions, [callId]: state } })),

      lockIn: (result, steps) => {
        const { results } = get()
        // A call closes once answered. Replays are flagged practice by the
        // caller and must not overwrite the real result or move the Tab.
        if (!result.practice && results.some((r) => r.callId === result.callId && !r.practice)) {
          return
        }
        set({ results: [...results, result] })
        if (!result.practice) get().observeCrowd(result.callId, result.value, steps)
      },

      observeCrowd: (callId, bucket, steps) => {
        const current = get().localCrowd[callId] ?? new Array(steps).fill(0)
        // Defensive: a call whose variable changed between app versions would
        // otherwise write past the end of a stale array and corrupt the chart.
        const sized =
          current.length === steps ? [...current] : new Array(steps).fill(0)
        const i = Math.max(0, Math.min(steps - 1, Math.round(bucket)))
        sized[i] += 1
        set({ localCrowd: { ...get().localCrowd, [callId]: sized } })
      },

      setSettings: (patch) => set({ settings: { ...get().settings, ...patch } }),

      reset: () => set({ ...INITIAL }),
    }),
    {
      // Renamed with the product. A returning player from the Compound build
      // starts clean rather than resuming a half-finished set under questions
      // that have since been rewritten.
      name: 'napkin',
      version: 2,
      storage: createJSONStorage(() => localStorage),
      // Zustand's persist already swallows storage errors, which matters here:
      // private windows and blocked site data both throw on getItem, and
      // neither should white-screen the app.
      partialize: (s) => ({
        profile: s.profile,
        results: s.results,
        localCrowd: s.localCrowd,
        actions: s.actions,
        settings: s.settings,
      }),
    },
  ),
)

/* ---- Selectors -------------------------------------------------------------
 * Derived rather than stored, so there is one source of truth and no way for a
 * cached total to drift from the results that produced it.
 */

/** The profile, falling back to the default so every screen can render. */
export function profileOrDefault(p: Profile | null): Profile {
  return p ?? DEFAULT_PROFILE
}

/** Real answers only. Practice runs never count. */
export function scoredResults(results: CallResult[]): CallResult[] {
  return results.filter((r) => !r.practice)
}

/** Has today's call been closed? */
export function resultFor(results: CallResult[], callId: number): CallResult | undefined {
  return results.find((r) => r.callId === callId && !r.practice)
}
