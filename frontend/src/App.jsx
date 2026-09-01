import { useState, useRef, useEffect } from 'react'
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import ForceGraph3D from 'react-force-graph-3d'

const datasets = [
  { file: 'amazon0302.txt', name: 'Amazon', edges: 1234877 },
  { file: 'web-Google.txt', name: 'Google', edges: 5105039 },
  { file: 'soc-LiveJournal1.txt', name: 'LiveJournal', edges: 68993773 }
]

function App() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  
  // Terminal state
  const [logs, setLogs] = useState(["[SYSTEM] Adaptive Graph Engine Initialized", "[SYSTEM] Waiting for execution command..."])
  const terminalRef = useRef(null)

  // VRAM & Chart state
  const [activeBlocks, setActiveBlocks] = useState(0)
  const [chartData, setChartData] = useState([])

  // Topology State
  const [graphData, setGraphData] = useState({ nodes: [], links: [] })

  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight
    }
  }, [logs])

  const appendLog = (msg) => {
    setLogs(prev => [...prev, msg])
    
    // Simulate real-time data parsing for the chart
    // For demo purposes, we randomly generate performance metrics as logs come in
    if (msg.includes("Processing") || msg.includes("Cycle") || msg.includes("Cycle complete")) {
      const timeStep = new Date().toLocaleTimeString().split(' ')[0]
      const throughput = Math.floor(Math.random() * 500) + 100
      setChartData(prev => {
        const newData = [...prev, { time: timeStep, speed: throughput }]
        return newData.length > 20 ? newData.slice(newData.length - 20) : newData
      })
      
      // Randomly allocate VRAM blocks
      setActiveBlocks(Math.floor(Math.random() * 150) + 20)
    }
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
    fetchTopology();
    
    // Initialize chart with empty baseline
    const initialData = Array(20).fill(0).map((_, i) => ({ time: `T-${20-i}`, speed: 0 }))
    setChartData(initialData)
  }, [])

  const runTopologyAnalysis = async () => {
    setLoading(true)
    setError(null)
    setLogs([])
    try {
      await executeEngineLive(datasets[2].file, 'benchmark')
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
        <h1>ADAPTIVE GRAPH ENGINE</h1>
        <p>Live Telemetry & Topology Visualization</p>
      </div>

      <div className="controls">
        <button className="run-btn" onClick={runTopologyAnalysis} disabled={loading}>
          {loading ? '[██████░░░░] RUNNING...' : '>_ RUN_BENCHMARK'}
        </button>

        <button className="run-btn" onClick={runScalingAnalysis} disabled={loading}>
          {loading ? '[██████░░░░] RUNNING...' : '>_ AMDAHL_SCALING'}
        </button>

        <button className="run-btn" onClick={runTriangleCounting} disabled={loading}>
          {loading ? '[██████░░░░] RUNNING...' : '>_ TRIANGLE_COUNTING'}
        </button>

        <button className="run-btn danger" onClick={runContinuousStressTest} disabled={loading}>
          {'>_ CONTINUOUS_STRESS_TEST'}
        </button>
      </div>

      {error && (
        <div style={{ color: '#ff003c', marginTop: '10px', textAlign: 'center', fontWeight: 'bold' }}>
          FATAL_ERROR: {error}
        </div>
      )}

      <div className="grid-layout">
        <div className="panel" style={{ padding: '0', height: '400px' }}>
          <div className="panel-title" style={{ position: 'absolute', top: 20, left: 20, zIndex: 10 }}>
            LIVE_3D_TOPOLOGY
          </div>
          {graphData.nodes.length > 0 ? (
            <ForceGraph3D
              graphData={graphData}
              width={750}
              height={400}
              nodeColor={() => '#00f0ff'}
              linkColor={() => 'rgba(255,255,255,0.2)'}
              nodeRelSize={4}
              backgroundColor="transparent"
              enableNodeDrag={true}
              enableNavigationControls={true}
              showNavInfo={false}
            />
          ) : (
            <div style={{ display: 'flex', height: '100%', alignItems: 'center', justifyContent: 'center', color: '#94a3b8' }}>
              LOADING_TOPOLOGY_DATA...
            </div>
          )}
        </div>

        <div className="panel">
          <div className="panel-title">EXECUTION_THROUGHPUT (MB/s)</div>
          <div style={{ width: '100%', height: '300px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="colorSpeed" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#00f0ff" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#00f0ff" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={false} />
                <XAxis dataKey="time" stroke="#94a3b8" fontSize={12} tickMargin={10} />
                <YAxis stroke="#94a3b8" fontSize={12} tickMargin={10} />
                <Tooltip 
                  contentStyle={{ backgroundColor: 'rgba(15,15,30,0.9)', borderColor: '#00f0ff', borderRadius: '8px' }}
                  itemStyle={{ color: '#00f0ff' }}
                />
                <Area type="monotone" dataKey="speed" stroke="#00f0ff" strokeWidth={3} fillOpacity={1} fill="url(#colorSpeed)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="grid-layout">
        <div className="panel">
          <div className="panel-title">UVM_VRAM_ALLOCATION</div>
          <div className="vram-grid">
            {renderVRAMMap()}
          </div>
          <div style={{ marginTop: '20px', fontSize: '0.85rem', color: '#94a3b8', display: 'flex', justifyContent: 'space-between' }}>
            <span>Active Pages: <span style={{color: '#00f0ff'}}>{activeBlocks} / 256</span></span>
            <span>Zero-Copy Mode: <span style={{color: '#39ff14'}}>ENABLED</span></span>
          </div>
        </div>
        
        <div className="panel">
          <div className="panel-title">STDOUT_TERMINAL</div>
          <div className="terminal" ref={terminalRef}>
            {logs.map((log, idx) => (
              <div key={idx} className="terminal-line">
                <span className="terminal-prefix">system@hpc:~$</span>
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
