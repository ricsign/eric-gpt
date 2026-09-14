import { haptic } from '../lib/haptics'
import type { Call } from '../calls/types'
import './Home.css'

/**
 * The first thing a stranger sees, and the answer to "what is this?".
 *
 * The screen this replaces asked for the visitor's salary under the words
 * "Compound · first run" and told them nothing else. Six independent
 * reviewers put that at the top of why the product was incomprehensible: a
 * cold visitor from a social link was being asked for her income by an
 * unnamed thing before she had been told the category, the benefit, or the
 * time cost. The salary question now comes after the first answer, where it
 * has been earned.
 *
 * The whole set is listed, unlocked ones tappable and the rest greyed. That
 * is the other half of the fix. Ten visible questions make the product finite
 * and completable in one glance — you can see the end of it — where an
 * endless daily drip reads as a content treadmill. It also means nobody has
 * to guess what "a money question" means: ten of them are right there.
 */
export function Home({
  calls,
  answered,
  nextId,
  onStart,
  onList,
  onProgress,
}: {
  calls: Call[]
  /** Ids already answered for real. Practice runs do not unlock anything. */
  answered: Set<number>
  /** The next unanswered question, or null when the set is finished. */
  nextId: number | null
  onStart: (id: number) => void
  onList: () => void
  onProgress: () => void
}) {
  const done = answered.size

  return (
    <div className="home scroll">
      <header className="home-head">
        <h1 className="home-mark">Napkin</h1>
        <p className="home-pitch">
          Ten money questions. One a day.
        </p>
        <p className="home-sub">
          You guess the number. We show you the real one, what most people guess, and the
          one thing to do about it.
        </p>
        <p className="home-terms data-sm">
          A minute each. No sign-up, no bank login. Nothing you type leaves this phone.
        </p>
      </header>

      {done > 0 && (
        <p className="home-progress data-sm num">
          {done} of {calls.length} answered
        </p>
      )}

      <ol className="home-list">
        {calls.map((c, i) => {
          const isDone = answered.has(c.id)
          // Everything up to and including the next unanswered question is
          // open. Locking the rest is what makes it one a day; showing them
          // anyway is what makes the set legible from the first second.
          const open = isDone || c.id === nextId
          return (
            <li key={c.id}>
              <button
                className="home-row press"
                data-state={isDone ? 'done' : open ? 'open' : 'locked'}
                disabled={!open}
                onClick={() => {
                  haptic('medium')
                  onStart(c.id)
                }}
              >
                <span className="home-no data-sm num">{i + 1}</span>
                <span className="home-q">{c.question}</span>
                <span className="home-state data-sm">
                  {isDone ? 'Answered' : open ? 'Start' : 'Tomorrow'}
                </span>
              </button>
            </li>
          )
        })}
      </ol>

      {/* The list is the thing people keep, so it cannot be reachable only
          from the end of a question. Someone returning on day four lands here,
          and without these they would have no route back to what they saved. */}
      {done > 0 && (
        <nav className="home-nav">
          <button className="home-link press" onClick={onList}>
            My list
          </button>
          <button className="home-link press" onClick={onProgress}>
            Progress
          </button>
        </nav>
      )}

      <p className="home-foot data-sm">Estimates, not advice.</p>
    </div>
  )
}
