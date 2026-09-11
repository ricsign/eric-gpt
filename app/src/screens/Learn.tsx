import { useMemo } from 'react'
import { Screen } from '../ui/Screen'
import { Card } from '../ui/Card'
import { Icon } from '../ui/Icon'
import { useNav } from '../ui/NavStack'
import { useStore } from '../state/store'
import { LESSONS, rankLessons } from '../data/lessons'
import { dueQueue } from '../lib/scheduler'
import { dayKey } from '../lib/format'
import { LessonPlayer } from './LessonPlayer'
import './Learn.css'

/**
 * The library.
 *
 * Presented as competences — things you can now do — rather than as a syllabus.
 * That is not a wording choice: topic-ordered curricula are the intervention shape
 * with the strongest evidence *against* them, so there is no unit 1, no lock icons,
 * no prerequisite chain, and no "next lesson" nudge at the end of a lesson.
 *
 * Ordering is by relevance to this person's actual numbers, recomputed each visit.
 */
export function LearnScreen() {
  const { state } = useStore()
  const nav = useNav()
  const today = dayKey()

  const ranked = useMemo(
    () => rankLessons(state.profile, state.lessons, state.jurisdiction ?? 'US'),
    [state.profile, state.lessons, state.jurisdiction],
  )

  const due = useMemo(() => {
    const withReviews = Object.entries(state.reviews).map(([id, review]) => ({ id, review }))
    const dueConcepts = new Set(dueQueue(withReviews, today).map((r) => r.id))
    return LESSONS.filter((l) => l.concepts.some((c) => dueConcepts.has(c)))
  }, [state.reviews, today])

  const done = ranked.filter((l) => state.lessons[l.id])
  const open = ranked.filter((l) => !state.lessons[l.id])

  const hidden = LESSONS.length - ranked.length

  return (
    <Screen title="Learn" eyebrow={`${done.length} of ${ranked.length} competences`}>
      <div className="learn">
        {due.length > 0 && (
          <section className="learn-section">
            <h2 className="learn-section-title">Due for review</h2>
            <p className="learn-section-note">
              Scheduled for roughly the point you would start to forget it — which is when
              revisiting does the most good.
            </p>
            {due.map((l) => (
              <LessonRow
                key={l.id}
                title={l.title}
                competence={l.competence}
                minutes={l.minutes}
                state="review"
                onOpen={() => nav.push(l.id, () => <LessonPlayer lessonId={l.id} />)}
              />
            ))}
          </section>
        )}

        {open.length > 0 && (
          <section className="learn-section">
            <h2 className="learn-section-title">Ready when you are</h2>
            <p className="learn-section-note">
              Ordered by what your own numbers suggest matters most. There is no sequence to
              follow and nothing is locked.
            </p>
            {open.map((l) => (
              <LessonRow
                key={l.id}
                title={l.title}
                competence={l.competence}
                minutes={l.minutes}
                onOpen={() => nav.push(l.id, () => <LessonPlayer lessonId={l.id} />)}
              />
            ))}
          </section>
        )}

        {done.length > 0 && (
          <section className="learn-section">
            <h2 className="learn-section-title">You can now</h2>
            {done.map((l) => (
              <LessonRow
                key={l.id}
                title={l.title}
                competence={l.competence}
                minutes={l.minutes}
                state="done"
                onOpen={() => nav.push(l.id, () => <LessonPlayer lessonId={l.id} />)}
              />
            ))}
          </section>
        )}

        {hidden > 0 && (
          <p className="learn-footnote">
            {hidden} {hidden === 1 ? 'lesson is' : 'lessons are'} hidden because they cover
            US-specific accounts and tax rules. Showing you those numbers for another country
            would be worse than showing you nothing — you can change your country in You.
          </p>
        )}
      </div>
    </Screen>
  )
}

function LessonRow({
  title,
  competence,
  minutes,
  state,
  onOpen,
}: {
  title: string
  competence: string
  minutes: number
  state?: 'done' | 'review'
  onOpen: () => void
}) {
  return (
    <Card onClick={onOpen} tone={state === 'review' ? 'spark' : 'default'}>
      <div className="learn-row">
        <div className="learn-row-text">
          <p className="learn-row-title">{title}</p>
          <p className="learn-row-competence">{competence}</p>
          <p className="learn-row-meta">
            {state === 'done' ? 'Completed' : state === 'review' ? 'Review' : `${minutes} min`}
          </p>
        </div>
        {state === 'done' ? (
          <span className="learn-row-check" aria-label="Completed">
            <Icon name="check" size={16} />
          </span>
        ) : (
          <Icon name="chevron" size={15} stroke className="learn-row-chevron" />
        )}
      </div>
    </Card>
  )
}
