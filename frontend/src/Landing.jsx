/*
 * Cuepoint landing page: 2.5D scroll story built from CSS/SVG only (no image assets).
 * Techniques: 6-layer depth model, pointer parallax, converging split headline,
 * sticky scrub timeline with word-by-word lighting, clip-path reveal.
 * Only transform / opacity / clip-path are animated. Reduced motion shows static content.
 */
import { useLayoutEffect, useRef } from 'react'
import gsap from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import './landing.css'

gsap.registerPlugin(ScrollTrigger)

const HEADLINE = 'Ask any video a question.'

const STEPS = [
  {
    key: 'upload',
    title: 'Add a video',
    body: 'Upload a recording of a meeting, lecture or tutorial.',
  },
  {
    key: 'listen',
    title: 'It listens',
    body: 'Everything that is said is turned into a searchable transcript.',
  },
  {
    key: 'look',
    title: 'It watches',
    body: 'Text shown on screen, like slides and diagrams, is read too.',
  },
  {
    key: 'answer',
    title: 'You get the answer',
    body: 'Ask in plain words. Every answer links to the exact moment in the video.',
  },
]

function Words({ text }) {
  // The space lives outside the inline-block span; inside it, it would collapse.
  return text.split(' ').map((w, i) => (
    <span key={i}><span className="w">{w}</span>{' '}</span>
  ))
}

function Visual({ kind }) {
  if (kind === 'upload') {
    return (
      <div className="vis vis-upload" aria-hidden="true">
        <div className="file-card">
          <svg viewBox="0 0 24 24" width="28" height="28"><path d="M8 5v14l11-7z" fill="currentColor" /></svg>
          <div><b>team-meeting.mp4</b><span>12:40 · 84 MB</span></div>
        </div>
        <div className="bar"><i /></div>
      </div>
    )
  }
  if (kind === 'listen') {
    return (
      <div className="vis" aria-hidden="true">
        {[['2:08', 'So where do we stand on the release?'], ['2:11', 'The fixes are merged and tested,'], ['2:14', 'so we ship the update on Friday.']].map(([t, s]) => (
          <div className="line" key={t}><code>{t}</code><span>{s}</span></div>
        ))}
      </div>
    )
  }
  if (kind === 'look') {
    return (
      <div className="vis vis-frames" aria-hidden="true">
        <div className="frame"><b>Agenda</b><span>Release status</span><code>0:30</code></div>
        <div className="frame warm"><b>Release plan</b><span>Ship on Friday</span><code>2:14</code></div>
      </div>
    )
  }
  return (
    <div className="vis vis-answer" aria-hidden="true">
      <div className="q">When do we ship the update?</div>
      <div className="a">The update ships on Friday. <span className="cite">▶ 2:14</span></div>
    </div>
  )
}

export default function Landing() {
  const root = useRef(null)

  useLayoutEffect(() => {
    const el = root.current
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const coarse = window.matchMedia('(pointer: coarse)').matches
    el.classList.toggle('reduced', reduce)
    if (reduce) return undefined

    let removePointer = () => {}
    const ctx = gsap.context(() => {
      // Entrance: words converge from alternating sides.
      gsap.from('.hero-title .w', {
        opacity: 0,
        x: (i) => (i % 2 ? 60 : -60),
        y: 20,
        duration: 1,
        ease: 'power3.out',
        stagger: 0.07,
      })
      gsap.from('.hero-sub, .hero-cta', { opacity: 0, y: 24, duration: 0.9, delay: 0.7, stagger: 0.12 })
      gsap.from('.frame-card', { opacity: 0, y: 80, scale: 0.92, duration: 1.2, delay: 0.3, ease: 'power3.out' })

      // Scroll: card grows, companions scatter outward by depth, copy drifts up.
      const hero = gsap.timeline({
        scrollTrigger: { trigger: '.hero', start: 'top top', end: 'bottom bottom', scrub: 0.6 },
      })
      hero
        .to('.frame-card', { scale: 1.28, y: -30, ease: 'none' }, 0)
        .to('.chip-a', { x: -220, y: -40, opacity: 0, ease: 'none' }, 0)
        .to('.chip-b', { x: 240, y: -20, opacity: 0, ease: 'none' }, 0)
        .to('.chip-c', { x: 160, y: 120, opacity: 0, ease: 'none' }, 0)
        .to('.hero-copy', { y: -120, opacity: 0, ease: 'none' }, 0.1)

      // Pointer parallax (fine pointers only).
      if (!coarse) {
        const layers = gsap.utils.toArray('.hero .layer[data-depth]')
        const setters = layers.map((l) => ({
          x: gsap.quickTo(l, 'x', { duration: 0.8, ease: 'power3' }),
          y: gsap.quickTo(l, 'y', { duration: 0.8, ease: 'power3' }),
          d: Number(l.dataset.depth),
        }))
        const move = (e) => {
          const nx = e.clientX / window.innerWidth - 0.5
          const ny = e.clientY / window.innerHeight - 0.5
          setters.forEach((s) => { s.x(-nx * s.d * 14); s.y(-ny * s.d * 10) })
        }
        window.addEventListener('pointermove', move)
        removePointer = () => window.removeEventListener('pointermove', move)
      }
    }, el)

    // Story: one sticky stage, scrubbed through the four steps.
    const mm = gsap.matchMedia()
    mm.add('(min-width: 0px)', () => {
      const steps = gsap.utils.toArray('.story .step')
      gsap.set(steps, { autoAlpha: 0, y: 40 })
      gsap.set(steps[0], { autoAlpha: 1, y: 0 })
      gsap.set('.story .step .w', { opacity: 0.22 })
      const tl = gsap.timeline({
        scrollTrigger: { trigger: '.story', start: 'top top', end: 'bottom bottom', scrub: 0.6 },
      })
      steps.forEach((s, i) => {
        if (i > 0) {
          // Out finishes exactly as in starts, so two steps never overlap on screen.
          tl.to(steps[i - 1], { autoAlpha: 0, y: -40, duration: 0.2 }, i - 0.2)
          tl.to(s, { autoAlpha: 1, y: 0, duration: 0.3 }, i)
        }
        tl.to(s.querySelectorAll('.w'), { opacity: 1, duration: 0.15, stagger: 0.03 }, i + 0.1)
      })
      tl.to('.story-progress i', { scaleX: 1, ease: 'none', duration: steps.length }, 0)
      tl.to({}, { duration: 0.4 }, steps.length) // hold the last step before releasing
    }, el)

    // CTA: section opens like a curtain.
    const cta = gsap.fromTo(
      '.cta-inner',
      { clipPath: 'inset(0 0 100% 0)' },
      {
        clipPath: 'inset(0 0 0% 0)', ease: 'none',
        scrollTrigger: { trigger: '.cta', start: 'top 85%', end: 'top 25%', scrub: 0.6 },
      },
    )

    return () => {
      removePointer()
      ctx.revert()
      mm.revert()
      cta.scrollTrigger?.kill()
      cta.kill()
    }
  }, [])

  return (
    <div className="landing" ref={root}>
      <header className="nav">
        <a className="brand" href="#/" aria-label="Cuepoint home">
          <svg viewBox="0 0 32 32" width="26" height="26" aria-hidden="true"><rect width="32" height="32" rx="8" fill="#121521" /><path d="M11 8l14 8-14 8z" fill="#5ee7ff" /></svg>
          Cuepoint
        </a>
        <a className="btn" href="#/app">Open the app</a>
      </header>

      <section className="scene hero" aria-labelledby="hero-title">
        <div className="hero-stage">
          <div className="layer depth-0" data-depth="0" aria-hidden="true"><div className="grid" /></div>
          <div className="layer depth-1" data-depth="1" aria-hidden="true">
            <div className="glow g1" /><div className="glow g2" />
          </div>
          <div className="layer depth-2" data-depth="2" aria-hidden="true">
            <div className="chip chip-a float"><i>What was said</i>“…so we ship the update on Friday.”</div>
            <div className="chip chip-b float f2"><i>What was shown</i>Release plan · Ship on Friday</div>
            <div className="chip chip-c float f3"><i>Jump to</i><b>▶ 2:14</b></div>
          </div>
          <div className="layer depth-3" data-depth="3" aria-hidden="true">
            <div className="frame-card float f4">
              <div className="screen">
                <span className="s1">Release plan</span>
                <span className="s2">Ship on Friday</span>
              </div>
              <div className="scrub"><i /><em>2:14</em></div>
            </div>
          </div>
          <div className="layer depth-4" data-depth="4">
            <div className="hero-copy">
              <h1 id="hero-title" className="hero-title"><Words text={HEADLINE} /></h1>
              <p className="hero-sub">Upload a video, ask what you want to know, and jump straight to the moment that answers it.</p>
              <div className="hero-cta">
                <a className="btn btn-primary" href="#/app">Get started</a>
                <a className="btn" href="#how">How it works</a>
              </div>
            </div>
          </div>
          <div className="layer depth-5" data-depth="5" aria-hidden="true">
            {Array.from({ length: 10 }, (_, i) => <i className={`dot d${i}`} key={i} />)}
          </div>
        </div>
      </section>

      <section className="story" id="how" aria-label="How it works">
        <div className="story-stage">
          <div className="layer depth-1" aria-hidden="true"><div className="glow g3" /></div>
          <div className="story-progress" aria-hidden="true"><i /></div>
          {STEPS.map((s, i) => (
            <article className="step" key={s.key}>
              <div className="step-text">
                <span className="step-no">0{i + 1}</span>
                <h2>{s.title}</h2>
                <p><Words text={s.body} /></p>
              </div>
              <Visual kind={s.key} />
            </article>
          ))}
        </div>
      </section>

      <section className="cta">
        <div className="cta-inner">
          <h2>Stop scrubbing. Start asking.</h2>
          <p>Try it on your own video in a minute.</p>
          <a className="btn btn-primary" href="#/app">Get started</a>
        </div>
      </section>

      <footer className="foot">Cuepoint · every answer links back to its source</footer>
    </div>
  )
}
