# Compound

A financial-literacy web app built to feel like a native iOS app, and built around
one bet: **teaching finance as a curriculum does not change behaviour, so don't
build one.**

```bash
cd app
npm install
npm run dev        # http://localhost:5173
npm test           # 165 unit tests, all of the money maths
npm run build
npm run shoot      # drives the app in Chromium and screenshots every screen
```

---

## The thesis

The strongest finding in the financial-education literature is an uncomfortable
one. Fernandes, Lynch & Netemeyer's meta-analysis of 201 studies found that
financial-education interventions explain about **0.1% of the variance in actual
financial behaviour**, with effects decaying to negligible within roughly twenty
months. "Module 1: Budgeting, Module 2: Credit, Module 3: Investing" is the exact
shape that finding indicts — and it is the shape of every finance-learning app on
the market.

What *does* move behaviour, per the follow-up literature (Kaiser & Menkhoff;
Kaiser et al. 2022), is a single counter-intuitive concept delivered close to a
real decision. Teaching compound interest at a moment of decision has moved real
retirement contributions by around 40%.

So this app is three things, in this order:

1. **A daily puzzle** — the ritual and the growth loop.
2. **A set of deterministic calculators** — the reason to come back in month six.
3. **Misconception-first lessons**, reachable from a result or a missed drill —
   never from a syllabus.

There is no unit 1, no lock icons, no prerequisite chain, and no "next lesson"
nudge at the end of a lesson.

## The five rules everything obeys

### 1. Compute, never conclude

A tool may show what $500/month becomes. It may never say "you should invest
$500/month." The legal line in this category is *personalisation, not topic* —
the Investment Advisers Act publisher's exclusion, as construed in *Lowe v. SEC*
(1985), protects impersonal advice and stops protecting it the moment output is
tailored to an individual's circumstances. Disclaimers do not cure that;
architecture does.

Every calculator in `src/lib/calculators.ts` returns a **date, a dollar amount, a
threshold, or a percentile**. Rent-vs-buy returns a break-even year. Payoff-vs-invest
returns the hurdle rate above which investing wins. Never a verdict. A test in
`src/data/content.test.ts` fails the build if lesson copy starts recommending.

### 2. Nothing that gets shared contains a dollar amount

Money is the most taboo category of personal data on the internet. Debt and salary
are the two most taboo topics in America, and only about 30% of people would tell a
close friend their bank balance. Every finance share artifact that requires
disclosing a figure — the net-worth card, the savings-score screenshot — demos
beautifully and shares at approximately zero.

So the shareable artifact is a Wordle-style glyph: how many tries, and your streak.

```
Compound #254 · 2/3 · 14🔥
⬛🟩
compound.money
```

Spoiler-free, so pasting it into a group chat cannot ruin the puzzle. No tracking
link — a bare domain on its own line, which is what Wordle settled on after
removing its own link. Zero personal disclosure, so the taboo simply does not apply.

### 3. Forward framing only

"Starting today puts you here," never "you already lost $47,000 by waiting."

The backward-regret genre (the latte factor and its descendants) is now widely
understood to have been built on ~11% return assumptions, and in 2026 it gets
quote-tweeted as the joke rather than shared as the insight. The maths is identical;
the framing is the difference between a useful lesson and a mockable one.

### 4. Reward showing up, never being right

The streak advances when you *attempt* the daily drill, not when you get it right.

Everything else the gamification playbook offers is deliberately absent:

| Included | Left out | Why |
|---|---|---|
| Forgiving streak with rest days | XP, levels, points | Deci et al. (1999), 128 studies: expected tangible rewards undermine intrinsic motivation at d = −0.28 to −0.40 |
| Named competences ("you can price a delay in dollars") | Leaderboards | Hanus & Fox (2015): 16 weeks of leaderboards produced *lower* motivation, satisfaction **and** exam scores |
| Informational feedback on every wrong answer | Hearts, lives, lockouts | A loss frame applied to people already anxious about money, and it makes errors unsafe |
| | Financial health scores out of 100 | Unfalsifiable, unactionable, and they invite exactly the gamification above |

The streak forgives one missed day automatically (two freezes, one earned back per
full week). Two missed days is a real break and does reset.

### 5. Show your sources, and your assumptions

Every figure the app states lives as a dated row in `src/data/facts.ts` with its
primary source and the date it was read — IRS Notice 2025-67, Rev. Proc. 2025-19,
the SSA Federal Register notice, TreasuryDirect, FDIC, myFICO, ICI. None of them
live in lesson prose. The **Sources** screen lists all of them with links, and
anything not read from the issuing authority is marked as such.

Every projection is driven by a return and inflation assumption the user can see
and drag. A model that hides its assumptions is asking to be trusted; one that
exposes them can be checked.

---

## Lesson anatomy

Every lesson is eight beats, in this fixed order. A test enforces it.

| # | Beat | Why it is there |
|---|---|---|
| 1 | **Anchor** | Names a decision in the learner's own numbers |
| 2 | **Prediction probe** | An answer committed *before* any instruction — the generation effect (d ≈ 0.40) and the misconception detector |
| 3 | **Gap reveal** | Their number against the real one, animated, on a linear axis |
| 4 | **Mechanism** | One causal sentence and one image. Nothing more |
| 5 | **Worked example** | Scaffolding that fades — expertise reversal means support that helps a novice hurts them three weeks later |
| 6 | **Practice** | 3–5 items, at least one in a different surface context |
| 7 | **Rule of thumb** | Named and portable. Rule-of-thumb training beats formula training, and the gap is largest for the least confident learners |
| 8 | **Action hook** | One concrete step bound to a real trigger |

**A lesson ends on the action.** There is no congratulations screen: completion is
the moment the learner is most likely to act, and spending it on confetti wastes
the product's own leverage.

Two ordering constraints are enforced by tests, because both are easy to erode:
content can never precede the probe, and the numeric slider never starts on the
right answer.

### Two schedulers, because there are two kinds of memory

- **Facts** (`rule of 72`, `APR ≠ APY`) use a half-life model in the FSRS family —
  scheduling on *recall probability* rather than on elapsed time. SM-2 loses to
  FSRS on log loss in ~99.6% of collections and cannot express a target retention.
- **Judgments** ("which debt do you pay first?") use a fixed expanding ladder —
  1, 3, 10, 30, 90, 270 days — with a **different surface scenario each time**.
  Item-level memory models do not model transfer, and a ladder that terminates at
  two weeks guarantees the knowledge is gone by the time the real decision arrives.

---

## Making the web feel like iOS

| Technique | Where |
|---|---|
| Real spring physics — `linear()` easings sampled from a damped harmonic oscillator, matching SwiftUI's `spring(response:bounce:)` | `scripts/gen-springs.mjs` → `styles/springs.css` |
| Interactive edge swipe-back, with parallax on the layer beneath and velocity-projected commit | `ui/NavStack.tsx` |
| Sheets with detents, rubber-banding past the top detent, and scroll-handoff arbitration | `ui/Sheet.tsx` |
| Large-title nav bar that crossfades to an inline title, driven by a CSS variable so scrolling costs no React renders | `ui/Screen.tsx` |
| Tab bar that slides away on push (`hidesBottomBarWhenPushed`) | `ui/Chrome.tsx` |
| Haptics on iOS via the hidden `<input type="checkbox" switch>` trick, since Safari has no `navigator.vibrate` | `lib/haptics.ts` |
| Safe areas, `dvh`, `overscroll-behavior: contain`, `touch-action: manipulation` | `styles/base.css` |
| Tabular figures everywhere, so an animating balance never reflows | `.num` |

Pinch-zoom is deliberately **not** disabled. Focus-zoom on inputs is prevented the
accessible way instead — every control is ≥16px — and the double-tap delay is
killed with `touch-action: manipulation`.

On a desktop window the app renders inside an iPhone bezel rather than stretching a
phone layout across 1600px. The frame fakes the safe-area insets, so the same
layout code positions content under the Dynamic Island there as it does on hardware.

### The honest gap

An iOS **widget** is unavailable to a pure web app, and at Duolingo the widget
performs comparably to push, with widget-installers disproportionately holding
6+ month streaks. Web push on iOS also requires the user to have added the app to
their Home Screen. These are real capability limits, not things to design around —
closing them needs a thin native shell.

---

## What is deliberately not here

- **No account.** Nothing to sign up for. State lives in `localStorage`.
- **No bank link.** Plaid costs 20–88% drop-off at the connect step; four sliders cost nothing.
- **No affiliate links, referral fees, or sponsored placements** — and no product-comparison surface at all, which is the only version of that promise that is verifiable.
- **No metering.** Every calculator is free and unlimited. They are the trust artifact.
- **No LLM chat surface.** A conversational wrapper over vague guidance is the commodity AI Overviews already give away, and it cannot produce a repeatable, auditable number.
- **No geo-detection.** Jurisdiction is asked explicitly at onboarding, and US-specific lessons are hidden rather than shown to non-US users with a caveat.

---

## Layout

```
src/
  lib/          finance.ts, calculators.ts, scheduler.ts, daily.ts — pure, tested, no React
  data/         facts.ts (dated + sourced), lessons.ts, drills.ts
  ui/           the iOS component kit
  screens/      Onboarding, Today, Drill, LessonPlayer, Tools, Learn, You
  state/        one reducer, persisted to localStorage
```

`src/lib` knows nothing about React, the DOM, or storage. A wrong number in a
finance app is the one bug that loses all trust, so the maths is isolated and every
figure quoted in lesson copy is cross-checked against the engine by a test.

## Tests

165 of them. The interesting ones are not the unit tests:

- **Content integrity** — every drill has exactly one correct answer, every option (including the right one) explains itself, no drill references personal data.
- **Arithmetic cross-checks** — the numbers printed in drill copy are recomputed from `finance.ts`. If an assumption changes, the copy fails rather than quietly lying.
- **Format invariants** — lessons build all eight beats in order, the probe always precedes explanation, every lesson ends on an action with an honest way out.
- **The compliance guard** — lesson output is scanned for recommendation language.
- **Smoke** — `npm run shoot` drives the whole app in Chromium and fails on any console error.
