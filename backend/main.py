from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
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
        
        # Convert Windows path (e.g. C:\all projects\...) to WSL path (/mnt/c/all projects/...)
        wsl_path = project_root.replace('\\', '/').replace('C:', '/mnt/c').replace('c:', '/mnt/c')
        
        # Build the path to the dataset
        dataset_path = f"data/{dataset}"
        
        # Let's try running it as a native WSL command first, which works inside WSL and from Windows.
        cmd = f'wsl -d Ubuntu -- bash -c "cd \'{wsl_path}\' && ./build/src/graph_engine \'{dataset_path}\' --benchmark"'
        
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
        cuda_match = re.search(r"GPU CUDA\s+([0-9.]+)", output)
        
        # Parse graph metadata
        v_match = re.search(r"V=([0-9]+)", output)
        e_match = re.search(r"E=([0-9]+)", output)

        if not (seq_match and omp_match and cuda_match and v_match and e_match):
            raise HTTPException(status_code=500, detail=f"Failed to parse engine output: {output}")

        return {
            "success": True,
            "dataset": dataset,
            "vertices": int(v_match.group(1)),
            "edges": int(e_match.group(1)),
            "results": {
                "sequential": float(seq_match.group(1)),
                "openmp": float(omp_match.group(1)),
                "cuda": float(cuda_match.group(1))
            },
            "raw_output": output
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/scheduler")
def explain_scheduler(dataset: str = Query("amazon0302.txt")):
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        wsl_path = project_root.replace('\\', '/').replace('C:', '/mnt/c').replace('c:', '/mnt/c')
        dataset_path = f"data/{dataset}"
        
        # Run without --benchmark to trigger AdaptiveScheduler
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
        wsl_path = project_root.replace('\\', '/').replace('C:', '/mnt/c').replace('c:', '/mnt/c')
        dataset_path = f"data/{dataset}"
        
        # Run with --scaling
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


