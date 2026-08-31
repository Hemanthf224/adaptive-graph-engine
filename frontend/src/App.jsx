import { useState, useRef, useEffect } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

const datasets = [
  { file: 'amazon0302.txt', name: 'Amazon', edges: 1234877 },
  { file: 'web-Google.txt', name: 'Google', edges: 5105039 },
  { file: 'soc-LiveJournal1.txt', name: 'LiveJournal', edges: 68993773 }
]

const TOTAL_VRAM_BYTES = 8 * 1024 * 1024 * 1024; // 8GB for RTX 5060

function App() {
  const [loading, setLoading] = useState(false)
  const [statusText, setStatusText] = useState('')
  const [chartData, setChartData] = useState([])
  const [error, setError] = useState(null)
  
  // Terminal state
  const [logs, setLogs] = useState(["[SYSTEM] Adaptive Graph Engine Initialized", "[SYSTEM] Waiting for execution command..."])
  const terminalRef = useRef(null)

  // VRAM state
  const [activeBlocks, setActiveBlocks] = useState(0)

  // Telemetry States
  const [schedulerData, setSchedulerData] = useState(null)
  const [analyzing, setAnalyzing] = useState(false)
  const [scalingData, setScalingData] = useState([])
  const [scalingLoading, setScalingLoading] = useState(false)

  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight
    }
  }, [logs])

  const appendLog = (msg) => {
    setLogs(prev => [...prev, msg])
  }

  const runTopologyAnalysis = async () => {
    setAnalyzing(true)
    setError(null)
    setSchedulerData(null)
    setLogs(["[SYSTEM] Connecting to Explainable AI Scheduler..."])
    
    try {
      const ds = datasets[2] // LiveJournal
      appendLog(`[SCHEDULER] Analyzing Topology: ${ds.file}...`)
      
      const response = await fetch(`http://localhost:8000/api/scheduler?dataset=${ds.file}`)
      if (!response.ok) {
        throw new Error("Failed to fetch scheduler telemetry")
      }
      
      const data = await response.json()
      setSchedulerData(data)
      
      appendLog(`[SCHEDULER] Topology analyzed successfully.`)
      appendLog(`[STDOUT]\n${data.raw_output}`)
      
    } catch (err) {
      setError(err.message)
      appendLog(`[ERROR] ${err.message}`)
    } finally {
      setAnalyzing(false)
    }
  }

  const runScalingAnalysis = async () => {
    setScalingLoading(true)
    setError(null)
    setScalingData([])
    setLogs(["[SYSTEM] Initiating OpenMP Strong Scaling Analysis (Amdahl's Law)..."])
    
    try {
      const ds = datasets[0] // Amazon (fast to run 6 times)
      appendLog(`[SCALING] Executing multi-threaded benchmarks on ${ds.file}...`)
      
      const response = await fetch(`http://localhost:8000/api/scaling?dataset=${ds.file}`)
      if (!response.ok) {
        throw new Error("Failed to fetch scaling telemetry")
      }
      
      const data = await response.json()
      
      // Format data for Recharts (x=threads, y=time)
      const formattedData = data.scaling.map(s => ({
        threads: `${s.threads} Cores`,
        Time: s.time
      }))
      
      setScalingData(formattedData)
      
      appendLog(`[SCALING] Thread scaling analysis complete.`)
      appendLog(`[STDOUT]\n${data.raw_output}`)
      
    } catch (err) {
      setError(err.message)
      appendLog(`[ERROR] ${err.message}`)
    } finally {
      setScalingLoading(false)
    }
  }

  const runBenchmarkAll = async () => {
    setLoading(true)
    setError(null)
    setChartData([])
    setActiveBlocks(0)
    setLogs(["[SYSTEM] Commencing Large-Scale Benchmark Sequence..."])
    
        const newChartData = []

    try {
      for (const ds of datasets) {
        setStatusText(`[RUNNING] ${ds.file}`)
        appendLog(`[EXEC] Requesting ${ds.file} via Python Subprocess...`)
        
        const bytesUsed = ds.edges * 4
        const percentage = bytesUsed / TOTAL_VRAM_BYTES
        const blocks = Math.ceil(percentage * 256)
        setActiveBlocks(blocks)

        const response = await fetch(`http://localhost:8000/api/benchmark?dataset=${ds.file}`)
        if (!response.ok) {
          throw new Error(`Failed to fetch ${ds.name} from backend`)
        }
        
        const data = await response.json()
        appendLog(`[STDOUT] \n${data.raw_output}`)
        appendLog(`[SUCCESS] ${ds.name} Parsed. CPU: ${data.results.sequential}ms | UVM: ${data.results.cuda_uvm}ms | Explicit: ${data.results.cuda_explicit}ms`)

        newChartData.push({
          name: ds.name,
          edges: data.edges,
          Sequential: data.results.sequential,
          OpenMP: data.results.openmp,
          CUDA_UVM: data.results.cuda_uvm,
          CUDA_Explicit: data.results.cuda_explicit
        })
        
        setChartData([...newChartData])
      }
      setStatusText('')
      appendLog(`[SYSTEM] Sequence Complete.`)
    } catch (err) {
      setError(err.message)
      appendLog(`[ERROR] ${err.message}`)
    } finally {
      setLoading(false)
      setStatusText('')
    }
  }

  const renderVRAMMap = () => {
    const blocks = []
    for (let i = 0; i < 256; i++) {
      blocks.push(
        <div key={i} className={`vram-block ${i < activeBlocks ? 'active' : ''}`}></div>
      )
    }
    return blocks
  }

  return (
    <div className="dashboard-container">
      <div className="header">
        <h1>ADAPTIVE_GRAPH_ENGINE // HPC_DASHBOARD</h1>
        <p>CUDA UVM ALLOCATOR & TELEMETRY MONITOR</p>
      </div>

      <div className="controls">
        <button 
          className="run-btn" 
          onClick={runTopologyAnalysis} 
          disabled={loading || analyzing || scalingLoading}
        >
          {analyzing ? '[██████░░░░] ANALYZING...' : '>_ ANALYZE_TOPOLOGY'}
        </button>

        <button 
          className="run-btn" 
          onClick={runScalingAnalysis} 
          disabled={loading || analyzing || scalingLoading}
        >
          {scalingLoading ? '[██████░░░░] SCALING...' : '>_ AMDAHL_STRONG_SCALING'}
        </button>

        <button 
          className="run-btn" 
          onClick={runBenchmarkAll} 
          disabled={loading || analyzing || scalingLoading}
        >
          {loading ? (
            <>
              <span>[██████░░░░]</span>
              <span>{statusText}</span>
            </>
          ) : (
            '>_ EXECUTE_BENCHMARK_SEQUENCE'
          )}
        </button>
      </div>

      {error && (
        <div style={{ color: '#ff3333', marginTop: '10px' }}>
          FATAL_ERROR: {error}
        </div>
      )}

      {schedulerData && (
        <div className="panel" style={{ borderLeft: '4px solid #76B900', backgroundColor: '#0f170a' }}>
          <div className="panel-title" style={{ color: '#76B900' }}>EXPLAINABLE_AI_SCHEDULER</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '20px', marginTop: '10px' }}>
            <div>
              <p><strong>Dataset:</strong> {schedulerData.dataset}</p>
              <p style={{ color: '#94a3b8', fontSize: '0.9rem', marginTop: '5px' }}>CPU Score: {schedulerData.cpu_score.toFixed(2)}</p>
              <p style={{ color: '#94a3b8', fontSize: '0.9rem' }}>OpenMP Score: {schedulerData.omp_score.toFixed(2)}</p>
              <p style={{ color: '#94a3b8', fontSize: '0.9rem' }}>CUDA Score: {schedulerData.cuda_score.toFixed(2)}</p>
              <p style={{ color: '#76B900', marginTop: '10px', fontWeight: 'bold' }}>SELECTED: {schedulerData.selected}</p>
            </div>
            <div style={{ padding: '15px', backgroundColor: '#050505', border: '1px solid #222', fontSize: '0.95rem', lineHeight: '1.5' }}>
              <span style={{ color: '#76B900' }}>[REASONING_ENGINE]</span><br />
              {schedulerData.reasoning}
            </div>
          </div>
        </div>
      )}

      <div className="grid-layout">
        <div className="panel" style={{ padding: '20px' }}>
          <div className="panel-title">OPENMP_STRONG_SCALING (AMDAHL'S LAW)</div>
          <div style={{ width: '100%', height: '250px' }}>
            {scalingData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={scalingData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                  <XAxis dataKey="threads" stroke="#94a3b8" />
                  <YAxis stroke="#94a3b8" label={{ value: 'Time (ms)', angle: -90, position: 'insideLeft', fill: '#94a3b8' }} />
                  <Tooltip contentStyle={{ backgroundColor: '#0A0A0A', border: '1px solid #0071c5', borderRadius: '0' }} />
                  <Legend verticalAlign="top" height={36} />
                  <Line type="monotone" dataKey="Time" stroke="#0071c5" strokeWidth={3} dot={{ r: 5, fill: '#0A0A0A' }} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div style={{ display: 'flex', height: '100%', alignItems: 'center', justifyContent: 'center', color: '#333' }}>
                AWAITING_SCALING_DATA...
              </div>
            )}
          </div>
        </div>

        <div className="panel" style={{ padding: '20px' }}>
          <div className="panel-title">EXECUTION_TIME_VS_GRAPH_SIZE</div>
          <div style={{ width: '100%', height: '250px' }}>
            {chartData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                  <XAxis dataKey="name" stroke="#94a3b8" />
                  <YAxis stroke="#94a3b8" />
                  <Tooltip contentStyle={{ backgroundColor: '#0A0A0A', border: '1px solid #76B900', borderRadius: '0' }} />
                  <Legend verticalAlign="top" height={36} />
                  <Line type="monotone" dataKey="Sequential" stroke="#94a3b8" strokeWidth={2} dot={{ r: 4, fill: '#0A0A0A' }} />
                  <Line type="monotone" dataKey="OpenMP" stroke="#0071c5" strokeWidth={2} dot={{ r: 4, fill: '#0A0A0A' }} />
                  <Line type="monotone" dataKey="CUDA_UVM" stroke="#76b900" strokeWidth={2} dot={{ r: 4, fill: '#0A0A0A' }} />
                  <Line type="monotone" dataKey="CUDA_Explicit" stroke="#ff3333" strokeWidth={2} dot={{ r: 4, fill: '#0A0A0A' }} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div style={{ display: 'flex', height: '100%', alignItems: 'center', justifyContent: 'center', color: '#333' }}>
                AWAITING_BENCHMARK_DATA...
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="grid-layout" style={{ gridTemplateColumns: '1fr 3fr' }}>
        <div className="panel">
          <div className="panel-title">UVM_VRAM_ALLOCATION</div>
          <div className="vram-grid">
            {renderVRAMMap()}
          </div>
          <div style={{ marginTop: '15px', fontSize: '0.8rem', color: '#94a3b8' }}>
            Active Pages: {activeBlocks} / 256<br/>
            Block Size: 32MB<br/>
            Zero-Copy Mode: ENABLED
          </div>
        </div>
        
        <div className="panel">
          <div className="panel-title">STDOUT_TERMINAL</div>
          <div className="terminal" ref={terminalRef}>
            {logs.map((log, idx) => (
              <div key={idx} className="terminal-line">
                <span className="terminal-prefix">root@hpc:~$</span>
                <span style={{ whiteSpace: 'pre-wrap' }}>{log}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
