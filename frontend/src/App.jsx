import { useState, useRef, useEffect } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import ForceGraph2D from 'react-force-graph-2d'

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

  // Topology State
  const [graphData, setGraphData] = useState({ nodes: [], links: [] })

  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight
    }
  }, [logs])

  const appendLog = (msg) => {
    setLogs(prev => [...prev, msg])
  }

  // WEBSOCKET EXECUTION
  const executeEngineLive = (dataset, mode) => {
    return new Promise((resolve, reject) => {
      appendLog(`[SYSTEM] Initiating Live WebSocket Stream: ws://localhost:8000/ws/stream?dataset=${dataset}&mode=${mode}`)
      const ws = new WebSocket(`ws://localhost:8000/ws/stream?dataset=${dataset}&mode=${mode}`)
      
      ws.onmessage = (event) => {
        appendLog(event.data)
        if (event.data === "[SYSTEM] Execution Completed.") {
          ws.close()
          resolve()
        }
      }
      
      ws.onerror = (err) => {
        appendLog(`[WS_ERROR] Connection Failed`)
        reject(err)
      }
    })
  }

  const fetchTopology = async () => {
    try {
      appendLog(`[SYSTEM] Fetching Sampled Topology for Render...`)
      const ds = datasets[0]
      const response = await fetch(`http://localhost:8000/api/topology?dataset=${ds.file}`)
      if (!response.ok) throw new Error("Failed to fetch topology")
      const data = await response.json()
      setGraphData(data)
    } catch (err) {
      appendLog(`[ERROR] ${err.message}`)
    }
  }

  useEffect(() => {
    fetchTopology(); // Load topology on mount
  }, [])

  const runTopologyAnalysis = async () => {
    setLoading(true)
    setError(null)
    setLogs([])
    try {
      await executeEngineLive(datasets[2].file, 'benchmark') // Run a quick analysis
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const runScalingAnalysis = async () => {
    setLoading(true)
    setError(null)
    setLogs([])
    try {
      await executeEngineLive(datasets[0].file, 'scaling')
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const runTriangleCounting = async () => {
    setLoading(true)
    setError(null)
    setLogs([])
    try {
      await executeEngineLive(datasets[0].file, 'triangles')
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const runContinuousStressTest = async () => {
    setLoading(true)
    setError(null)
    setLogs(["[SYSTEM] INITIATING 10-HOUR CONTINUOUS STRESS TEST..."])
    
    // Simulate infinite loop for UI
    appendLog("[WARNING] Entering infinite loop. Monitoring memory leaks...")
    const loop = async () => {
      while (true) {
        await executeEngineLive(datasets[0].file, 'benchmark')
        appendLog("[SYSTEM] Cycle complete. Cooling down for 2 seconds...")
        await new Promise(r => setTimeout(r, 2000))
      }
    }
    loop()
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
        <h1>ADAPTIVE_GRAPH_ENGINE // HPC_DASHBOARD (V2.0)</h1>
        <p>LIVE WEBSOCKET TELEMETRY & FORCE-DIRECTED TOPOLOGY</p>
      </div>

      <div className="controls">
        <button className="run-btn" onClick={runTopologyAnalysis} disabled={loading}>
          {loading ? '[██████░░░░] RUNNING...' : '>_ RUN_BENCHMARK'}
        </button>

        <button className="run-btn" onClick={runScalingAnalysis} disabled={loading}>
          {loading ? '[██████░░░░] RUNNING...' : '>_ AMDAHL_SCALING'}
        </button>

        <button className="run-btn" onClick={runTriangleCounting} disabled={loading}>
          {loading ? '[██████░░░░] RUNNING...' : '>_ TRIANGLE_COUNTING (NEW)'}
        </button>

        <button className="run-btn" style={{ backgroundColor: '#ff3333' }} onClick={runContinuousStressTest} disabled={loading}>
          >_ CONTINUOUS_STRESS_TEST
        </button>
      </div>

      {error && (
        <div style={{ color: '#ff3333', marginTop: '10px' }}>
          FATAL_ERROR: {error}
        </div>
      )}

      <div className="grid-layout">
        <div className="panel" style={{ padding: '0', height: '400px', overflow: 'hidden', position: 'relative' }}>
          <div className="panel-title" style={{ position: 'absolute', top: 10, left: 10, zIndex: 10, backgroundColor: 'rgba(0,0,0,0.8)' }}>
            LIVE_TOPOLOGY_VISUALIZATION
          </div>
          {graphData.nodes.length > 0 ? (
            <ForceGraph2D
              graphData={graphData}
              width={800}
              height={400}
              nodeColor={() => '#76B900'}
              linkColor={() => '#333'}
              nodeRelSize={4}
              backgroundColor="#0A0A0A"
            />
          ) : (
            <div style={{ display: 'flex', height: '100%', alignItems: 'center', justifyContent: 'center', color: '#333' }}>
              LOADING_TOPOLOGY_DATA...
            </div>
          )}
        </div>

        <div className="panel" style={{ padding: '20px' }}>
          <div className="panel-title">EXECUTION_TIME_VS_GRAPH_SIZE</div>
          <div style={{ width: '100%', height: '300px' }}>
            <div style={{ display: 'flex', height: '100%', alignItems: 'center', justifyContent: 'center', color: '#333' }}>
              [SEE LIVE TERMINAL FOR REAL-TIME BENCHMARKS]
            </div>
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
          <div className="panel-title">STDOUT_TERMINAL (LIVE WEBSOCKET STREAM)</div>
          <div className="terminal" ref={terminalRef} style={{ height: '300px' }}>
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
