# Compound

**One money call a day.** The same one for everyone, playable in under sixty
seconds. You drag a real financial variable with your thumb, watch your future
change on the same frame, and take a receipt.

Live: https://compound-rose-delta.vercel.app

```bash
cd app
npm install
npm run dev        # http://localhost:5173
npm test           # the money maths and the seams between agents
npm run build
npm run shoot      # drives the whole loop in Chromium and screenshots it
npm run a11y       # reduced motion, keyboard operation, landmarks
```

---

## The thesis

Financial literacy apps fail because they teach vocabulary. Nobody needs to know
what a 401(k) *is*. They need to have already felt what happens when they set it
to 2% instead of 6%.

So Compound never explains a rule before the player has run into it. The rule is
revealed **after** they commit to an answer, and only then does it get filed to
their permanent Rules list.

## The four non-negotiables

Everything else in this repo is replaceable. These are the product.

**1. The mechanic is a gesture, never multiple choice.**
Every call is a continuous variable you drag. The lesson is discovered through
the control's own behaviour. In call No.1 the blocks below the employer's 6% cap
fill acid lime and the ones above fill plain white — the player drags, sees the
green stop, sees their monthly cost keep climbing, and has understood that the
match has a ceiling in about four seconds without reading a word.

If a call cannot be expressed as a draggable variable, it is an article, and it
does not ship.

**2. One call a day, the same for everyone, closed once answered.**
No binging, no content library to churn through. Scarcity plus synchrony is what
makes "did you do today's Compound?" a sentence people say out loud. The day
turns over at 06:00 local, not midnight — someone playing at 00:30 is finishing
their evening, not starting a new day.

**3. The artifact is a receipt, not a grid.**
A block grid is Wordle's costume. Money has its own native object: torn edges,
monospace, dotted-leader line items, a stamp, and a barcode whose bar widths come
from the player's actual result.

**4. No paragraphs.**
Every screen is a number, a control, or one line of type. A second sentence means
the screen has failed.

## The loop

```
COLD OPEN → DRAG → LOCK IN → OUTCOME → RECEIPT → TOMORROW
   0s        4s      12s       25s       40s      (countdown)
```

No splash, no login wall, no tutorial. The app opens directly onto today's call,
already interactive. First run asks one question — annual salary, with a
draggable number and a skip — and that single input personalises every figure in
the product forever. No email, no age, no goals questionnaire.

Auth is offered only after the first receipt exists, as "save your tab".

## Why it feels like an app

| | |
|---|---|
| Drag runs on pointer events with pointer capture, `touch-action: none` | `ui/BlockBar.tsx` |
| Value quantizes to integer steps — the same finger position always gives the same value | `positionToValue`, property-tested for idempotence |
| Every dependent number recomputes on the **same frame** as the drag. The paint may be rAF'd; the value never is | `screens/Call.tsx` |
| Haptic tick on each integer crossing, not each pointermove | `lib/haptics.ts` |
| Spring screen transitions at ~300ms, damping ~28 — not fades | `App.tsx` |
| `overscroll-behavior: contain`, no body rubber-banding, safe-area insets | `styles/base.css` |
| Works offline once visited; `sw.js` is served `must-revalidate` so a new build actually reaches people | `public/sw.js`, `vercel.json` |
| Fonts self-hosted — no third-party request, no render-blocking stylesheet | `scripts/fetch-fonts.mjs` |

Pinch-zoom stays enabled. Focus-zoom is prevented the accessible way instead:
every control is ≥16px.

On desktop the app renders inside a 402×874 frame rather than stretching a phone
layout across 1600px. The layout is authored at exactly the dimensions it ships
at, so a Swift port is a straight lift.

## The visual system

```
Background  #08080A    Surface  #111114
Accent      #C6F24E  (acid lime — free money, optimal, affirmative)
Loss        #FF5A36    Caution  #FFD84E
Paper       #F2EDE3  (receipts only)    Ink  #131316

Display  Archivo Black  uppercase, -0.03em, 17-64px
Body     Space Grotesk  13-15px, sparingly
Data     Courier Prime  all labels, all figures, 0.1-0.16em
```

No rounded cards beyond 3px. No gradients. No shadows except the receipt. No
icons that are not geometric primitives. No emoji anywhere in the UI. Hairlines
are a 1px flex gap on a lighter ground, never a border — a border rounds with the
box and picks up corner antialiasing; a gap stays exactly one device pixel.

## Adding a call

One JSON-shaped record in `calls/registry.ts`, and occasionally one pure function
in `calls/compute.ts`. A non-engineer can ship a call.

```ts
{
  id: 1,
  title: 'Your boss pays 50c for every dollar you save.',  // max 12 words
  variable: { key: 'contribution', min: 0, max: 15, step: 1, unit: '%', start: 3 },
  fixed: [{ k: 'PAY', v: '{{salary}}' }, { k: 'MATCH', v: '50% UP TO 6% OF PAY' }],
  compute: 'employerMatch',
  optimal: 6,
  rule: 'Take the match before anything else.',           // under 10 words
  crowd: [14, 2, 3, 22, 6, 9, 13, 2, 5, 1, 12, 1, 3, 1, 1, 5],
  tomorrow: 'You have $500 spare. Both cards want it.',
}
```

`variable.start` is deliberately the *wrong* answer, and usually the one the real
world defaults to — 3% auto-enrolment, a $0 counter-offer, the whole repair quote.
If the control opened on the optimum the gesture would teach nothing.

`crowd` is a real-world prior, not a bell curve. Its shape is the second half of
the lesson: in call No.1 the tallest spike sits at the 3% auto-enrolment default,
which stops half a step short of the match. **The gap between where the crowd
piles up and where the optimum sits is the thing worth screenshotting**, and each
distribution carries a comment justifying its shape against real survey data.

## Tests

The interesting ones are not unit tests. They are the **seam tests** in
`calls/integration.test.ts`, which check the modules against each other rather
than against their own intentions:

- **The declared optimal really is the best play** — every call's full range is
  swept at four different salaries and the peak must land where the record
  claims. A call whose maths disagrees with its own `optimal` tells a correct
  player they left money behind.
- **No position produces a broken number** — no NaN, no Infinity, no negative
  monthly cost at any reachable value for any profile.
- **`crowd.length === stepCount(variable)`** — drift here misaligns every bar in
  the histogram against the value it claims to represent. The chart still
  renders; it just lies.
- **The control never starts on the optimum**, and always on a reachable step.
- **Moving the control changes something** — a call whose numbers do not move as
  you drag is an article wearing a control.
- **The breakdown is receipt-ready at every position** — 3-6 lines, exactly one
  emphasis, labels short enough for the monospace column.

## Deliberate choices

Override only with a reason.

- **No monetization in v1, and no streak-freeze purchases ever.** Selling
  protection against loss teaches the opposite of what the app teaches.
- **The wrong answer shows a loss in red, unsoftened.** It is the strongest viral
  lever and the honest one. The one concession is that the loss is always framed
  against the optimal play on the same call, never against the player's real
  finances — which they have not disclosed and we do not hold.
- **No friend leaderboard at launch.** It converts a learning tool into a flex and
  changes who plays.
- **No AI chat coach.** It would make the app generic and unfalsifiable overnight.
  The Rules page is the memory layer instead.
- **No fabricated player count.** The header shows a live count only when there is
  a real one. A made-up "9,400 people have already called it" in a product whose
  pitch is that the numbers are real would be the worst possible place to lie.

## Known limits

- **The crowd histogram is authored priors blended with local results.** Real
  aggregation needs a backend; the seed distributions are grounded in published
  survey data and each carries its source in a comment, but they are priors, not
  observations. `lib/crowd.ts` already decays the seed as real results arrive.
- **No iOS widget or Live Activity.** Unavailable to a pure web app; both need a
  native shell.
- **Web push on iOS requires a Home Screen install**, which is a small
  single-digit share of visitors. Notifications are a bonus surface, never the
  retention plan.
- **Projections use 7% nominal and state it.** Never presented as a promise. The
  employer match is the one figure stated as certain, because it is.
