import { useState, Component } from 'react'
import UploadForm from './components/UploadForm'
import ControlsPanel from './components/ControlsPanel'
import LossChart from './components/LossChart'
import Diagnostics from './components/Diagnostics'

class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error }
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="bg-red-50 border border-red-300 p-4 rounded m-6">
          <h2 className="text-red-700 font-bold mb-2">Something went wrong</h2>
          <p className="text-red-600 text-sm">{this.state.error?.message}</p>
          <button
            className="mt-3 px-4 py-1 bg-red-600 text-white rounded"
            onClick={() => this.setState({ hasError: false, error: null })}
          >
            Try Again
          </button>
        </div>
      )
    }
    return this.props.children
  }
}

function App() {
  const [eda, setEda] = useState(null)
  const [history, setHistory] = useState(null)
  const [diagnostics, setDiagnostics] = useState([])

  return (
    <ErrorBoundary>
      <div className="min-h-screen p-6 bg-gray-100">
        <h1 className="text-2xl font-bold mb-6">
          ML Training Diagnostics Dashboard
        </h1>

        <UploadForm
          onUploadSuccess={(edaResult) => {
            setEda(edaResult)
            setHistory(null)
            setDiagnostics([])
          }}
        />

        {eda && (
          <ControlsPanel
            onTrainingComplete={(result) => {
              setHistory(result.history)
              setDiagnostics(result.diagnostics || [])
            }}
          />
        )}

        {history && <LossChart history={history} />}

        {diagnostics && diagnostics.length > 0 && (
          <Diagnostics diagnostics={diagnostics} />
        )}
      </div>
    </ErrorBoundary>
  )
}

export default App

