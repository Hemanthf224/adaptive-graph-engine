import { useState } from 'react'

function App() {
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)

  const runBenchmark = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch('http://localhost:8000/api/benchmark')
      if (!response.ok) {
        throw new Error('Failed to fetch from backend')
      }
      const data = await response.json()
      setResults(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="dashboard-container">
      <div className="header">
        <h1>Adaptive Graph Engine</h1>
        <p>CUDA Unified Memory (UVM) Benchmarking Dashboard</p>
      </div>

      <div className="controls">
        <button 
          className="run-btn" 
          onClick={runBenchmark} 
          disabled={loading}
        >
          {loading ? (
            <>
              <div className="spinner"></div>
              Running Benchmarks...
            </>
          ) : (
            '▶ Execute Hardware Benchmark'
          )}
        </button>
      </div>

      {error && (
        <div style={{ color: '#ef4444', textAlign: 'center', background: 'rgba(239,68,68,0.1)', padding: '10px', borderRadius: '8px' }}>
          Error: {error}
        </div>
      )}

      {results && (
        <>
          <div className="dataset-info">
            <div>Dataset: <strong>{results.dataset}</strong></div>
            <div>Vertices: <strong>{results.vertices.toLocaleString()}</strong></div>
            <div>Edges: <strong>{results.edges.toLocaleString()}</strong></div>
          </div>

          <div className="results-grid">
            <div className="metric-card cpu-seq">
              <h3>CPU Sequential</h3>
              <div className="metric-value">
                {results.results.sequential.toFixed(2)} <span>ms</span>
              </div>
            </div>
            
            <div className="metric-card cpu-omp">
              <h3>CPU OpenMP</h3>
              <div className="metric-value">
                {results.results.openmp.toFixed(2)} <span>ms</span>
              </div>
            </div>
            
            <div className="metric-card gpu-cuda">
              <h3>GPU CUDA (UVM)</h3>
              <div className="metric-value" style={{ color: 'var(--cuda-green)' }}>
                {results.results.cuda.toFixed(2)} <span>ms</span>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  )
}

export default App
