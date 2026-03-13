const API_BASE = 'https://ml-training-diagnostics-dashboard.onrender.com/'

export async function uploadCSV(file) {
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch(`${API_BASE}/upload`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    throw new Error('CSV upload failed')
  }

  return response.json()
}

export async function trainModel(payload) {
  const response = await fetch(`${API_BASE}/train`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  })

  if (!response.ok) {
    throw new Error('Training failed')
  }

  return response.json()
}
