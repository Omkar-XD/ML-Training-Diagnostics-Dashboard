// Use Vite environment variable for the API base URL.
// In production (Vercel), set VITE_API_BASE in the Vercel dashboard,
// or it will auto-detect and use the Render backend URL.
// In development, it falls back to localhost.
const RENDER_BACKEND = 'https://ml-training-diagnostics-dashboard.onrender.com'

function getApiBase() {
  // 1. Explicit env var always wins
  if (import.meta.env.VITE_API_BASE) {
    return import.meta.env.VITE_API_BASE
  }
  // 2. If running on Vercel (production), use Render backend
  if (typeof window !== 'undefined' && window.location.hostname.includes('vercel.app')) {
    return RENDER_BACKEND
  }
  // 3. Local development fallback
  return 'http://localhost:8000'
}

const API_BASE = getApiBase()

export async function uploadCSV(file) {
  const formData = new FormData()
  formData.append('file', file)

  let response
  try {
    response = await fetch(`${API_BASE}/upload`, {
      method: 'POST',
      body: formData,
    })
  } catch (networkError) {
    throw new Error(
      `Cannot reach backend server. Make sure the backend is running at ${API_BASE}`
    )
  }

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}))
    throw new Error(errorData.detail || `CSV upload failed (status ${response.status})`)
  }

  return response.json()
}

export async function trainModel(payload) {
  let response
  try {
    response = await fetch(`${API_BASE}/train`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    })
  } catch (networkError) {
    throw new Error(
      `Cannot reach backend server. Make sure the backend is running at ${API_BASE}`
    )
  }

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}))
    throw new Error(errorData.detail || `Training failed (status ${response.status})`)
  }

  return response.json()
}
