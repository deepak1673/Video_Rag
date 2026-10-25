// Thin API client. The demo token (if the server requires one) lives in sessionStorage.

const BASE = '/api/v1'
const TOKEN_KEY = 'cuepoint.token'

export const getToken = () => {
  try { return sessionStorage.getItem(TOKEN_KEY) || '' } catch { return '' }
}
export const setToken = (t) => {
  try { t ? sessionStorage.setItem(TOKEN_KEY, t) : sessionStorage.removeItem(TOKEN_KEY) } catch { /* private mode */ }
}

export class ApiError extends Error {
  constructor(status, code, message) {
    super(message)
    this.status = status
    this.code = code
  }
}

async function request(path, { method = 'GET', body, form } = {}) {
  const headers = {}
  const token = getToken()
  if (token) headers['X-Demo-Token'] = token
  if (body) headers['Content-Type'] = 'application/json'
  let res
  try {
    res = await fetch(BASE + path, { method, headers, body: body ? JSON.stringify(body) : form })
  } catch {
    throw new ApiError(0, 'network', 'Cannot reach the server. Is the API running?')
  }
  if (res.status === 204) return null
  let data = null
  try { data = await res.json() } catch { /* non-JSON error body */ }
  if (!res.ok) {
    const e = data?.error
    throw new ApiError(res.status, e?.code || `http_${res.status}`, e?.message || `Request failed (${res.status}).`)
  }
  return data
}

// <video>/<img> cannot set headers, so media URLs carry the token as a query parameter.
export const mediaUrl = (path) => {
  const t = getToken()
  return t ? `${path}${path.includes('?') ? '&' : '?'}token=${encodeURIComponent(t)}` : path
}

export const api = {
  health: () => request('/health'),
  listVideos: () => request('/videos'),
  getStatus: (id) => request(`/videos/${id}/status`),
  upload: (file) => {
    const form = new FormData()
    form.append('file', file)
    return request('/videos', { method: 'POST', form })
  },
  process: (id) => request(`/videos/${id}/process`, { method: 'POST' }),
  remove: (id) => request(`/videos/${id}`, { method: 'DELETE' }),
  transcript: (id) => request(`/videos/${id}/transcript`),
  search: (query, videoIds) => request('/search', { method: 'POST', body: { query, video_ids: videoIds } }),
  ask: (question, videoIds) => request('/ask', { method: 'POST', body: { question, video_ids: videoIds } }),
}

export const fmtTime = (s) => {
  const t = Math.max(0, Math.round(s))
  const m = Math.floor(t / 60)
  return `${m}:${String(t % 60).padStart(2, '0')}`
}
