from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import subprocess
import re
import os

app = FastAPI(title="Adaptive Graph Engine API")

# Enable CORS for the React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "Adaptive Graph Engine Backend is Running"}

@app.get("/api/benchmark")
def run_benchmark(dataset: str = Query("amazon0302.txt", description="Dataset file name inside data/ directory")):
    try:
        # Determine the absolute path to the project root assuming backend is in project_root/backend
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Detect OS: If running in Docker (Linux), execute directly. If Windows, use WSL.
        if os.name == 'posix':
            cmd = f'cd {project_root} && ./build/src/graph_engine \'{dataset_path}\' --benchmark --runs 4'
        else:
            wsl_path = project_root.replace('\\', '/').replace('C:', '/mnt/c').replace('c:', '/mnt/c')
            cmd = f'wsl -d Ubuntu -- bash -c "cd \'{wsl_path}\' && ./build/src/graph_engine \'{dataset_path}\' --benchmark --runs 4"'
        
        # Run the command and capture output
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True
        )
        
        output = result.stdout + result.stderr
        
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"C++ Engine Failed: {output}")

        # Parse the execution times from the output using Regex
        seq_match = re.search(r"CPU Sequential\s+([0-9.]+)", output)
        omp_match = re.search(r"CPU OpenMP\s+([0-9.]+)", output)
        cuda_match = re.search(r"GPU CUDA \(UVM\)\s+([0-9.]+)", output)
        cuda_exp_match = re.search(r"GPU CUDA \(Expl\)\s+([0-9.]+)", output)
        
        # Parse graph metadata
        v_match = re.search(r"V=([0-9]+)", output)
        e_match = re.search(r"E=([0-9]+)", output)

        if not (seq_match and omp_match and cuda_match and cuda_exp_match and v_match and e_match):
            raise HTTPException(status_code=500, detail=f"Failed to parse engine output: {output}")

        return {
            "success": True,
            "dataset": dataset,
            "vertices": int(v_match.group(1)),
            "edges": int(e_match.group(1)),
            "results": {
                "sequential": float(seq_match.group(1)),
                "openmp": float(omp_match.group(1)),
                "cuda_uvm": float(cuda_match.group(1)),
                "cuda_explicit": float(cuda_exp_match.group(1))
            },
            "raw_output": output
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/scheduler")
def explain_scheduler(dataset: str = Query("amazon0302.txt")):
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dataset_path = f"data/{dataset}"
        
        if os.name == 'posix':
            cmd = f'cd {project_root} && ./build/src/graph_engine \'{dataset_path}\''
        else:
            wsl_path = project_root.replace('\\', '/').replace('C:', '/mnt/c').replace('c:', '/mnt/c')
            cmd = f'wsl -d Ubuntu -- bash -c "cd \'{wsl_path}\' && ./build/src/graph_engine \'{dataset_path}\'"'
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        output = result.stdout + result.stderr
        
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"C++ Engine Failed: {output}")

        cpu_score = re.search(r"CPU Score\s+:\s+([0-9.-]+)", output)
        omp_score = re.search(r"OMP Score\s+:\s+([0-9.-]+)", output)
        cuda_score = re.search(r"CUDA Score\s+:\s+([0-9.-]+)", output)
        reasoning = re.search(r"Reasoning\s+:\s+(.*)", output)
        selected = re.search(r"Selected\s+:\s+(.*)", output)

        if not (cpu_score and omp_score and cuda_score and reasoning and selected):
            raise HTTPException(status_code=500, detail=f"Failed to parse scheduler output: {output}")

        return {
            "success": True,
            "dataset": dataset,
            "cpu_score": float(cpu_score.group(1)),
            "omp_score": float(omp_score.group(1)),
            "cuda_score": float(cuda_score.group(1)),
            "reasoning": reasoning.group(1).strip(),
            "selected": selected.group(1).strip(),
            "raw_output": output
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/scaling")
def run_scaling_analysis(dataset: str = Query("amazon0302.txt")):
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dataset_path = f"data/{dataset}"
        
        if os.name == 'posix':
            cmd = f'cd {project_root} && ./build/src/graph_engine \'{dataset_path}\' --scaling'
        else:
            wsl_path = project_root.replace('\\', '/').replace('C:', '/mnt/c').replace('c:', '/mnt/c')
            cmd = f'wsl -d Ubuntu -- bash -c "cd \'{wsl_path}\' && ./build/src/graph_engine \'{dataset_path}\' --scaling"'
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        output = result.stdout + result.stderr
        
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"C++ Engine Failed: {output}")

        # Parse scaling results
        # [SCALING_RESULT] Threads: 1 | Time: 109.834 ms
        scaling_data = []
        for match in re.finditer(r"\[SCALING_RESULT\] Threads:\s+(\d+)\s+\|\s+Time:\s+([0-9.]+)\s+ms", output):
            scaling_data.append({
                "threads": int(match.group(1)),
                "time": float(match.group(2))
            })

        if not scaling_data:
            raise HTTPException(status_code=500, detail=f"Failed to parse scaling output: {output}")

        return {
            "success": True,
            "dataset": dataset,
            "scaling": scaling_data,
            "raw_output": output
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/trace")
def download_trace():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    trace_path = os.path.join(project_root, "trace.json")
    if os.path.exists(trace_path):
        return FileResponse(trace_path, filename="trace.json", media_type="application/json")
    raise HTTPException(status_code=404, detail="Trace file not found. Run a benchmark first.")


