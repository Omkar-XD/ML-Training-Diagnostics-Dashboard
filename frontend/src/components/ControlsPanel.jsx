import { useState } from 'react'
import { trainModel } from '../api'

function ControlsPanel({ onTrainingComplete }) {

  const [modelType, setModelType] = useState('linear_regression')

  const [targetColumn, setTargetColumn] = useState('Credit_Score')

  const [learningRate, setLearningRate] = useState(0.01)

  const [epochs, setEpochs] = useState(20)

  const [loading, setLoading] = useState(false)

  const [error, setError] = useState(null)

  const handleTrain = async () => {

    setError(null)

    setLoading(true)

    const payload = {
      model_type: modelType,
      target_column: targetColumn,
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

      <h2 className="font-semibold mb-3">
        Training Controls
      </h2>

      {/* MODEL SELECTION */}

      <div className="mb-3">

        <label className="font-medium">
          Model:
        </label>

        <select
          className="ml-2 border px-2 py-1"
          value={modelType}
          onChange={(e) => setModelType(e.target.value)}
        >

          <option value="linear_regression">
            Linear Regression
          </option>

          <option value="decision_tree_regressor">
            Decision Tree Regressor
          </option>

          <option value="logistic_regression">
            Logistic Regression
          </option>

          <option value="decision_tree_classifier">
            Decision Tree Classifier
          </option>

          <option value="mlp">
            MLP
          </option>

        </select>

      </div>

      {/* TARGET COLUMN */}

      <div className="mb-3">

        <label className="font-medium">
          Target Column:
        </label>

        <input
          type="text"
          className="ml-2 border px-2 py-1"
          value={targetColumn}
          onChange={(e) => setTargetColumn(e.target.value)}
          placeholder="Enter target column"
        />

      </div>

      {/* MLP SETTINGS */}

      {modelType === 'mlp' && (

        <>

          <div className="mb-2">

            <label>
              Learning Rate:
            </label>

            <input
              type="number"
              step="0.001"
              className="ml-2 border px-2 py-1"
              value={learningRate}
              onChange={(e) =>
                setLearningRate(Number(e.target.value))
              }
            />

          </div>

          <div className="mb-3">

            <label>
              Epochs:
            </label>

            <input
              type="number"
              className="ml-2 border px-2 py-1"
              value={epochs}
              onChange={(e) =>
                setEpochs(Number(e.target.value))
              }
            />

          </div>

        </>

      )}

      {/* TRAIN BUTTON */}

      <button
        onClick={handleTrain}
        disabled={loading}
        className="px-4 py-2 bg-green-600 text-white rounded disabled:opacity-50"
      >

        {loading ? 'Training...' : 'Train'}

      </button>

      {/* ERROR */}

      {error && (
        <p className="text-red-600 mt-3">
          {error}
        </p>
      )}

    </div>
  )
}

export default ControlsPanel