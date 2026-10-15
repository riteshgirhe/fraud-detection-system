// Use the same origin as the frontend. Vite proxies local development requests
// to FastAPI, while nginx proxies production requests in Docker.
const API_BASE = (import.meta.env.VITE_API_BASE_URL || '/api/v1').replace(/\/$/, '')
const HEALTH_URL = API_BASE.startsWith('/') ? '/health' : `${new URL(API_BASE).origin}/health`

async function request(path, options = {}) {
  let response
  try {
    const url = path.startsWith('http') || path === '/health' ? path : `${API_BASE}${path}`
    response = await fetch(url, {
      ...options,
      headers: { ...(options.body ? { 'Content-Type': 'application/json' } : {}), ...options.headers },
      signal: options.signal || AbortSignal.timeout(10000),
    })
  } catch (error) {
    throw new Error(error.name === 'TimeoutError' ? 'The API request timed out.' : 'Cannot reach the API. Check that the backend is running.')
  }
  const body = await response.json().catch(() => ({}))
  if (!response.ok) {
    const detail = Array.isArray(body.detail) ? body.detail.map((item) => item.msg).join(', ') : body.detail
    throw new Error(detail || `API request failed (${response.status})`)
  }
  return body
}

export function predictTransaction(transaction) {
  return request('/predict', { method: 'POST', body: JSON.stringify(transaction) })
}
export function predictBatch(transactions) {
  return request('/predict/batch', { method: 'POST', body: JSON.stringify({ transactions }) })
}
export function getPredictions(offset = 0, limit = 50) {
  return request(`/predictions?offset=${offset}&limit=${limit}`)
}
export function patchLabel(transactionId, trueLabel) {
  return request(`/predictions/${encodeURIComponent(transactionId)}/label`, {
    method: 'PATCH',
    body: JSON.stringify({ true_label: trueLabel }),
  })
}

export async function getDashboardData() {
  const [health, summary, predictions, drift, model] = await Promise.all([
    request(HEALTH_URL),
    request('/monitoring/summary'),
    getPredictions(0, 100),
    request('/monitoring/drift'),
    request('/model/info'),
  ])
  return { health, summary, predictions, drift, model }
}
