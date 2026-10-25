import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api, ApiError, fmtTime, getToken, mediaUrl, setToken } from './api.js'
import './workspace.css'

const FINAL = new Set(['ready', 'partial', 'failed', 'uploaded'])
const SEARCHABLE = new Set(['ready', 'partial'])
const LABEL_RE = /\[E(\d+)\]/g

function StatusBadge({ video }) {
  const busy = video.status === 'queued' || video.status === 'processing'
  return (
    <span className={`badge ${video.status}`}>
      {busy ? `${video.status} · ${Math.round(video.progress * 100)}%` : video.status}
    </span>
  )
}

function Answer({ result, onSeek }) {
  const byLabel = useMemo(() => new Map(result.citations.map((c) => [c.label, c])), [result])
  const parts = result.answer.split(LABEL_RE)
  return (
    <p className="answer-text">
      {parts.map((part, i) => {
        if (i % 2 === 0) return part
        const c = byLabel.get(Number(part))
        if (!c) return null
        return (
          <button className="chip-cite" key={i} onClick={() => onSeek(c.video_id, c.start_seconds)}
            title={`Jump to ${fmtTime(c.start_seconds)}`}>
            ▶ {fmtTime(c.start_seconds)}
          </button>
        )
      })}
    </p>
  )
}

function EvidenceCard({ item, onSeek, videoName }) {
  const start = item.start_seconds
  const kinds = (item.modalities || []).join(' + ')
  return (
    <li className="evidence">
      {item.thumbnail_url && (
        <img src={mediaUrl(item.thumbnail_url)} alt={`Frame at ${fmtTime(start)}`} loading="lazy" />
      )}
      <div className="evidence-body">
        <div className="evidence-meta">
          <span className="kind">{kinds}</span>
          {videoName && <span className="vname">{videoName}</span>}
          <button className="chip-cite" onClick={() => onSeek(item.video_id, start)}>▶ {fmtTime(start)}</button>
        </div>
        <p>{item.snippet || 'Visual match'}</p>
      </div>
    </li>
  )
}

export default function Workspace() {
  const [videos, setVideos] = useState([])
  const [selectedId, setSelectedId] = useState(null)
  const [transcript, setTranscript] = useState([])
  const [now, setNow] = useState(0)
  const [authNeeded, setAuthNeeded] = useState(false)
  const [tokenDraft, setTokenDraft] = useState('')
  const [error, setError] = useState('')
  const [uploading, setUploading] = useState(false)
  const [dragging, setDragging] = useState(false)

  const [mode, setMode] = useState('ask')
  const [query, setQuery] = useState('')
  const [allVideos, setAllVideos] = useState(false)
  const [busy, setBusy] = useState(false)
  const [asked, setAsked] = useState(null) // { kind: 'ask'|'search', data, query }

  const videoEl = useRef(null)
  const pendingSeek = useRef(null)

  const selected = videos.find((v) => v.id === selectedId) || null
  const nameOf = useCallback((id) => videos.find((v) => v.id === id)?.original_filename, [videos])

  const handleError = useCallback((e) => {
    if (e instanceof ApiError && (e.status === 401 || e.status === 403)) setAuthNeeded(true)
    setError(e.message || 'Something went wrong.')
  }, [])

  const refresh = useCallback(async () => {
    try {
      const list = await api.listVideos()
      setVideos(list)
      setAuthNeeded(false)
      setSelectedId((cur) => cur ?? list.find((v) => SEARCHABLE.has(v.status))?.id ?? list[0]?.id ?? null)
    } catch (e) { handleError(e) }
  }, [handleError])

  useEffect(() => { refresh() }, [refresh])

  // Poll while anything is still being processed.
  const anyBusy = videos.some((v) => !FINAL.has(v.status))
  useEffect(() => {
    if (!anyBusy) return undefined
    const t = setInterval(refresh, 3000)
    return () => clearInterval(t)
  }, [anyBusy, refresh])

  // Load the transcript when the selection becomes searchable.
  const selectedStatus = selected?.status
  useEffect(() => {
    setTranscript([])
    setNow(0)
    if (!selectedId || !SEARCHABLE.has(selectedStatus)) return
    api.transcript(selectedId).then(setTranscript).catch(handleError)
  }, [selectedId, selectedStatus, handleError])

  const seekTo = useCallback((videoId, t) => {
    if (videoId !== selectedId) {
      pendingSeek.current = t
      setSelectedId(videoId)
      return
    }
    const el = videoEl.current
    if (!el) return
    el.currentTime = t
    el.play().catch(() => {})
    el.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }, [selectedId])

  const onLoadedMeta = () => {
    if (pendingSeek.current != null && videoEl.current) {
      videoEl.current.currentTime = pendingSeek.current
      videoEl.current.play().catch(() => {})
      pendingSeek.current = null
    }
  }

  const uploadFile = async (file) => {
    if (!file) return
    setError('')
    setUploading(true)
    try {
      const v = await api.upload(file)
      await api.process(v.id)
      setSelectedId(v.id)
      await refresh()
    } catch (e) { handleError(e) } finally { setUploading(false) }
  }

  const removeVideo = async (v) => {
    if (!window.confirm(`Delete "${v.original_filename}" and its search data?`)) return
    try {
      await api.remove(v.id)
      if (selectedId === v.id) setSelectedId(null)
      await refresh()
    } catch (e) { handleError(e) }
  }

  const submit = async (e) => {
    e.preventDefault()
    const q = query.trim()
    if (!q) return
    const scope = allVideos || !selected ? undefined : [selected.id]
    setBusy(true)
    setError('')
    try {
      const data = mode === 'ask' ? await api.ask(q, scope) : await api.search(q, scope)
      setAsked({ kind: mode, data, query: q })
    } catch (err) { handleError(err) } finally { setBusy(false) }
  }

  const saveToken = (e) => {
    e.preventDefault()
    setToken(tokenDraft.trim())
    setError('')
    refresh()
  }

  const currentSeg = transcript.findIndex((s) => now >= s.start_seconds && now < s.end_seconds)
  const readyCount = videos.filter((v) => SEARCHABLE.has(v.status)).length

  return (
    <div className="ws">
      <header className="ws-top">
        <a className="brand" href="#/">
          <svg viewBox="0 0 32 32" width="24" height="24" aria-hidden="true"><rect width="32" height="32" rx="8" fill="#121521" /><path d="M11 8l14 8-14 8z" fill="#5ee7ff" /></svg>
          Cuepoint
        </a>
        {getToken() && <button className="btn small" onClick={() => { setToken(''); setAuthNeeded(true) }}>Clear token</button>}
      </header>

      {authNeeded && (
        <form className="notice token" onSubmit={saveToken}>
          <label htmlFor="tok">This server needs an access token.</label>
          <input id="tok" type="password" autoComplete="off" value={tokenDraft}
            onChange={(e) => setTokenDraft(e.target.value)} placeholder="Access token" />
          <button className="btn btn-primary small" type="submit">Unlock</button>
        </form>
      )}
      {error && !authNeeded && <div className="notice error" role="alert">{error}</div>}

      <main className="ws-grid">
        <aside className="lib" aria-label="Video library">
          <div
            className={`drop ${dragging ? 'on' : ''}`}
            onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => { e.preventDefault(); setDragging(false); uploadFile(e.dataTransfer.files?.[0]) }}
          >
            <p><b>{uploading ? 'Uploading…' : 'Drop a video'}</b></p>
            <p className="hint">MP4, MOV, MKV, WebM · up to 500 MB · 15 min</p>
            <label className="btn small">
              Choose file
              <input type="file" accept="video/*,.mkv" className="sr-only" disabled={uploading}
                onChange={(e) => { uploadFile(e.target.files?.[0]); e.target.value = '' }} />
            </label>
          </div>

          <ul className="vlist">
            {videos.length === 0 && <li className="empty">No videos yet.</li>}
            {videos.map((v) => (
              <li key={v.id} className={v.id === selectedId ? 'active' : ''}>
                <button className="vitem" onClick={() => setSelectedId(v.id)}>
                  <span className="vname">{v.original_filename}</span>
                  <span className="vmeta">{v.duration_seconds ? fmtTime(v.duration_seconds) : '—'} · <StatusBadge video={v} /></span>
                </button>
                <button className="icon" aria-label={`Delete ${v.original_filename}`} onClick={() => removeVideo(v)}>✕</button>
              </li>
            ))}
          </ul>
        </aside>

        <section className="stage" aria-label="Player and transcript">
          {selected ? (
            <>
              <video
                ref={videoEl} key={selected.id} controls preload="metadata"
                src={mediaUrl(`/api/v1/videos/${selected.id}/media`)}
                onLoadedMetadata={onLoadedMeta}
                onTimeUpdate={(e) => setNow(e.currentTarget.currentTime)}
              />
              {selected.status === 'failed' && <div className="notice error">{selected.error_summary || 'Processing failed.'}</div>}
              {selected.status === 'partial' && <div className="notice">Some processing steps failed; search may be incomplete.</div>}
              {selected.status === 'uploaded' && (
                <button className="btn btn-primary" onClick={() => api.process(selected.id).then(refresh).catch(handleError)}>Process this video</button>
              )}
              <h2 className="sec">Transcript</h2>
              <ol className="transcript">
                {transcript.length === 0 && <li className="empty">{SEARCHABLE.has(selected.status) ? 'No speech detected.' : 'Available once processing finishes.'}</li>}
                {transcript.map((s, i) => (
                  <li key={s.idx} className={i === currentSeg ? 'now' : ''}>
                    <button onClick={() => seekTo(selected.id, s.start_seconds)}>
                      <code>{fmtTime(s.start_seconds)}</code> {s.text}
                    </button>
                  </li>
                ))}
              </ol>
            </>
          ) : (
            <div className="empty big">Select or upload a video to begin.</div>
          )}
        </section>

        <section className="qa" aria-label="Ask or search">
          <div className="tabs" role="tablist">
            {['ask', 'search'].map((m) => (
              <button key={m} role="tab" aria-selected={mode === m} className={mode === m ? 'on' : ''} onClick={() => { setMode(m); setAsked(null) }}>
                {m === 'ask' ? 'Ask' : 'Search'}
              </button>
            ))}
          </div>
          <form onSubmit={submit} className="qform">
            <label htmlFor="q" className="sr-only">{mode === 'ask' ? 'Question' : 'Search query'}</label>
            <input id="q" value={query} maxLength={500} onChange={(e) => setQuery(e.target.value)}
              placeholder={mode === 'ask' ? 'e.g. When is the Europe launch?' : 'e.g. revenue growth'} />
            <button className="btn btn-primary" disabled={busy || !query.trim() || readyCount === 0}>
              {busy ? '…' : mode === 'ask' ? 'Ask' : 'Search'}
            </button>
          </form>
          <label className="scope">
            <input type="checkbox" checked={allVideos} onChange={(e) => setAllVideos(e.target.checked)} />
            Search all videos ({readyCount} ready)
          </label>
          {readyCount === 0 && <p className="hint">Process a video first, then ask about it.</p>}

          {asked && asked.kind === 'ask' && (
            <div className="result">
              {asked.data.insufficient_evidence
                ? <p className="abstain">{asked.data.answer}</p>
                : <Answer result={asked.data} onSeek={seekTo} />}
              {asked.data.citations.length > 0 && (
                <>
                  <h3 className="sec">Sources</h3>
                  <ul className="evlist">
                    {asked.data.citations.map((c) => (
                      <EvidenceCard key={c.evidence_id} item={{ ...c, modalities: c.modalities }} onSeek={seekTo}
                        videoName={allVideos ? nameOf(c.video_id) : null} />
                    ))}
                  </ul>
                </>
              )}
              <p className="hint">{asked.data.elapsed_ms} ms</p>
            </div>
          )}

          {asked && asked.kind === 'search' && (
            <div className="result">
              {asked.data.results.length === 0
                ? <p className="abstain">No relevant moments found.</p>
                : (
                  <ul className="evlist">
                    {asked.data.results.map((r) => (
                      <EvidenceCard key={r.evidence_id} item={r} onSeek={seekTo}
                        videoName={allVideos ? nameOf(r.video_id) : null} />
                    ))}
                  </ul>
                )}
            </div>
          )}
        </section>
      </main>
    </div>
  )
}
