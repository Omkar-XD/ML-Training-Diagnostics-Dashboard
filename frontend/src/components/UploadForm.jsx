import { useState } from 'react'
import { uploadCSV } from '../api'

function UploadForm({ onUploadSuccess }) {
  const [file, setFile] = useState(null)
  const [uploadResult, setUploadResult] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a CSV file first.')
      return
    }

    setError(null)
    setLoading(true)

    try {
      const result = await uploadCSV(file)
      setUploadResult(result)
      onUploadSuccess(result.eda || result)
    } catch (err) {
      setError(err.message || 'Upload failed')
    } finally {
      setLoading(false)
    }
  }

  // The API returns { message, rows, columns, eda: { shape, dtypes, missing, head } }
  const edaData = uploadResult?.eda

  return (
    <div className="bg-white p-4 rounded shadow mb-6">
      <h2 className="font-semibold mb-2">Upload CSV</h2>

      <input
        type="file"
        accept=".csv"
        onChange={(e) => setFile(e.target.files[0])}
      />

      <button
        onClick={handleUpload}
        disabled={loading}
        className="ml-2 px-4 py-1 bg-blue-600 text-white rounded disabled:opacity-50"
      >
        {loading ? 'Uploading...' : 'Upload'}
      </button>

      {error && <p className="text-red-600 mt-2">{error}</p>}

      {uploadResult && (
        <div className="mt-4 text-sm">
          <p><b>Rows:</b> {uploadResult.rows}</p>
          <p><b>Columns:</b> {uploadResult.columns?.join(', ')}</p>
          {edaData?.shape && (
            <p><b>Shape:</b> {Array.isArray(edaData.shape) ? edaData.shape.join(' × ') : JSON.stringify(edaData.shape)}</p>
          )}
          {edaData?.missing && (
            <p><b>Missing values:</b> {JSON.stringify(edaData.missing)}</p>
          )}
        </div>
      )}
    </div>
  )
}

export default UploadForm
