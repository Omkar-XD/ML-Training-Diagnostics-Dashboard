import { useState } from 'react'
import { trainModel } from '../api'

function ControlsPanel({ onTrainingComplete }) {
  const [modelType, setModelType] = useState('linear')
  const [learningRate, setLearningRate] = useState(0.01)
  const [epochs, setEpochs] = useState(20)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleTrain = async () => {
    setError(null)
    setLoading(true)

    const payload = {
      model_type: modelType,
    }

    if (modelType === 'mlp') {
      payload.learning_rate = learningRate
      payload.epochs = epochs
    }

    try {
      const result = await trainModel(payload)
      onTrainingComplete(result)
    } catch (err) {
      setError(err.message || 'Training failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="bg-white p-4 rounded shadow mb-6">
      <h2 className="font-semibold mb-3">Training Controls</h2>

      <div className="mb-2">
        <label>Model:</label>
        <select
          className="ml-2 border"
          value={modelType}
          onChange={(e) => setModelType(e.target.value)}
        >
          <option value="linear">Linear Regression</option>
          <option value="tree">Decision Tree</option>
          <option value="mlp">MLP</option>
        </select>
      </div>

      {modelType === 'mlp' && (
        <>
          <div className="mb-2">
            <label>Learning Rate:</label>
            <input
              type="number"
              step="0.001"
              className="ml-2 border"
              value={learningRate}
              onChange={(e) => setLearningRate(Number(e.target.value))}
            />
          </div>

          <div className="mb-3">
            <label>Epochs:</label>
            <input
              type="number"
              className="ml-2 border"
              value={epochs}
              onChange={(e) => setEpochs(Number(e.target.value))}
            />
          </div>
        </>
      )}

      <button
        onClick={handleTrain}
        disabled={loading}
        className="px-4 py-1 bg-green-600 text-white rounded disabled:opacity-50"
      >
        {loading ? 'Training...' : 'Train'}
      </button>

      {error && <p className="text-red-600 mt-2">{error}</p>}
    </div>
  )
}

export default ControlsPanel
