import type { Profile } from '../state/store'
import type { GrowthPoint } from '../lib/finance'
import type { CurvePoint } from '../ui/CurveDraw'

/**
 * The lesson format.
 *
 * Fixed at eight beats, in this order, for every lesson in the app. The order is
 * not stylistic — each beat earns its place from a specific finding:
 *
 *  1. ANCHOR      name a decision the learner plausibly faces, in their numbers.
 *  2. PROBE       make them commit to an answer BEFORE any instruction. This is the
 *                 generation effect, and it is the misconception detector. Showing
 *                 content first forfeits it and manufactures an illusion of fluency.
 *  3. REVEAL      their number against the real one, animated, on a linear axis.
 *  4. MECHANISM   one causal sentence and one image. Nothing more.
 *  5. WORKED      a worked example whose scaffolding fades with measured ability.
 *  6. PRACTICE    3-5 items, at least one in a different surface context.
 *  7. RULE        one named, portable heuristic — rules of thumb beat formulas,
 *                 and the gap is largest for the least confident learners.
 *  8. ACTION      one concrete next step bound to a real trigger.
 *
 * A lesson ends on the action. There is no congratulations screen: the moment of
 * completion is the moment the learner is most likely to act, and spending it on
 * confetti is the single most common way education apps waste their own leverage.
 */

export type BeatKind =
  | 'anchor'
  | 'probe'
  | 'reveal'
  | 'mechanism'
  | 'worked'
  | 'practice'
  | 'rule'
  | 'action'

export interface AnchorBeat {
  kind: 'anchor'
  /** Second person, present tense, their numbers. */
  body: string
  /** Optional framing note, e.g. which assumption is in play. */
  note?: string
}

/**
 * The prediction probe.
 *
 * Three modes, in descending order of preference:
 *
 *  - `draw` — the default wherever the answer is a curve. The learner draws the
 *    shape with a finger before seeing anything. It commits far more willingly
 *    than a number field, and the gap between the drawn line and the truth is a
 *    direct measurement of exponential-growth bias rather than a proxy for it.
 *  - `estimate` — a slider, for quantities that are not curves. Still a forced
 *    commit, still no keyboard and no arithmetic demanded.
 *  - `choice` — only for judgments that are not quantities at all, and only with
 *    distractors that each encode a real, named misconception.
 *
 * What is never used: an un-skippable free numeric entry. Demanding arithmetic on
 * the second screen from an anxious, low-numeracy learner is the highest-churn
 * design available.
 */
export type ProbeBeat =
  | {
      kind: 'probe'
      mode: 'draw'
      question: string
      /** The truth, sampled across the domain. Built lazily. */
      curve: () => CurvePoint[]
      /** Top of the y axis. Chosen so the truth uses most of the height. */
      yMax: number
      /** Right-hand axis label, e.g. "30 years". */
      xLabel: string
      /** Used in prose: "over 30 years". */
      domainLabel: string
      /**
       * The scenario stated in neutral, universal terms — "$300 a month for 30
       * years at 7%". Printed on the share card, so it must never contain the
       * learner's own figures.
       */
      scenario: string
      /** Shown after the reveal, never before. */
      because: string
      /** The concept id this probe measures, for the calibration score. */
      measures: string
    }
  | {
      kind: 'probe'
      mode: 'estimate'
      question: string
      /** The true answer. */
      answer: number
      unit: 'usd' | 'years' | 'percent' | 'count'
      /** Fractional band that counts as correct, e.g. 0.2 for "within 20%". */
      tolerance: number
      /** Bounds for the slider the learner drags. */
      min: number
      max: number
      /** Where the slider starts. Never at the answer. */
      start: number
      /** Shown after the answer is locked in, never before. */
      because: string
    }
  | {
      kind: 'probe'
      mode: 'choice'
      question: string
      /** Each wrong option encodes a real, documented misconception. */
      options: { label: string; correct: boolean; misconception?: string }[]
      because: string
    }

export interface RevealBeat {
  kind: 'reveal'
  headline: string
  body: string
  /** Built lazily: charts are expensive and most beats never render one. */
  chart?: () => { points: GrowthPoint[]; comparison?: GrowthPoint[]; comparisonLabel?: string }
  /** Two numbers placed side by side, when a chart would be overkill. */
  contrast?: { label: string; value: string; tone: 'growth' | 'drag' | 'neutral' }[]
}

export interface MechanismBeat {
  kind: 'mechanism'
  /** Exactly one causal sentence. If it needs two, the lesson is doing too much. */
  sentence: string
  /** A short elaboration, at most three lines. */
  detail: string
  /** Which built-in diagram to draw alongside it. */
  visual: 'doubling' | 'split' | 'reverse' | 'bracket' | 'none'
}

export interface WorkedStep {
  label: string
  /** The value, as it should read. */
  value: string
  /** When true, the learner supplies this step instead of reading it. */
  blankable?: boolean
  /** What the learner must type or pick when this step is blanked. */
  answer?: number
  unit?: 'usd' | 'years' | 'percent'
}

export interface WorkedBeat {
  kind: 'worked'
  setup: string
  steps: WorkedStep[]
  conclusion: string
}

export interface PracticeItem {
  prompt: string
  options: { label: string; correct: boolean; why: string }[]
  /**
   * Marks an item set in a different surface context from the anchor. At least one
   * per lesson: practice that never leaves the original wrapper teaches the
   * wrapper, not the idea.
   */
  transfer?: boolean
}

export interface PracticeBeat {
  kind: 'practice'
  items: PracticeItem[]
}

export interface RuleBeat {
  kind: 'rule'
  /** Named, so it can be referred to later: "the Rule of 72". */
  name: string
  statement: string
  /** One worked instance, so the rule is not abstract. */
  example: string
}

export interface ActionBeat {
  kind: 'action'
  /**
   * An implementation intention: "when X happens, do Y". Bound to a trigger rather
   * than to willpower, because that is what the evidence says survives the week.
   */
  when: string
  then: string
  /** What the learner picks from. Always includes an honest "not now". */
  options: { label: string; commits: boolean }[]
  /** Estimated annual value, when it can be computed honestly. Never invented. */
  worth?: (profile: Profile) => number | null
}

export type Beat =
  | AnchorBeat
  | ProbeBeat
  | RevealBeat
  | MechanismBeat
  | WorkedBeat
  | PracticeBeat
  | RuleBeat
  | ActionBeat

/**
 * When a lesson becomes relevant.
 *
 * The trigger layer is a first-class system, not a content tag. Without it,
 * "just-in-time" degrades into a curriculum with better microcopy and inherits the
 * weak effects of the thing it claims to replace.
 */
export type Trigger =
  | 'always'
  | 'has-debt'
  | 'has-employer-plan'
  | 'new-job'
  | 'first-paycheck'
  | 'open-enrollment'
  | 'tax-season'
  | 'statement-day'
  | 'cash-heavy'
  | 'market-drop'

export interface Lesson {
  id: string
  /**
   * The competence gained, phrased as something the learner can now DO. Progress in
   * this app is measured in named competences, never in points.
   */
  competence: string
  /** Short title for the card. */
  title: string
  /** The specific wrong belief this lesson exists to correct. */
  misconception: string
  /** The research behind it, shown to the user on request. */
  citation?: { text: string; url?: string }
  triggers: Trigger[]
  /** US-specific lessons are gated; the rest are jurisdiction-neutral. */
  jurisdiction: 'any' | 'US'
  minutes: number
  /** Concept ids this lesson teaches, for the spaced-review scheduler. */
  concepts: string[]
  /** Built from the learner's own numbers each time it is opened. */
  build: (profile: Profile) => Beat[]
}
