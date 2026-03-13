import { useState } from 'react'
import UploadForm from './components/UploadForm'
import ControlsPanel from './components/ControlsPanel'
import LossChart from './components/LossChart'
import Diagnostics from './components/Diagnostics'

function App() {
  const [eda, setEda] = useState(null)
  const [history, setHistory] = useState(null)
  const [diagnostics, setDiagnostics] = useState([])

  return (
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
            setDiagnostics(result.diagnostics)
          }}
        />
      )}

      {history && <LossChart history={history} />}

      {diagnostics.length > 0 && (
        <Diagnostics diagnostics={diagnostics} />
      )}
    </div>
  )
}

export default App
