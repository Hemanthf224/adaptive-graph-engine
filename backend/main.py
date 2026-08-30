from fastapi import FastAPI, HTTPException
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
def run_benchmark():
    try:
        # Determine the absolute path to the project root assuming backend is in project_root/backend
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # We execute the graph engine inside WSL. 
        # Convert Windows path (e.g. C:\all projects\...) to WSL path (/mnt/c/all projects/...)
        wsl_path = project_root.replace('\\', '/').replace('C:', '/mnt/c').replace('c:', '/mnt/c')
        
        cmd = f'wsl -d Ubuntu -- bash -c "cd \'{wsl_path}\' && ./build/src/graph_engine data/amazon0302.txt --benchmark"'
        
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
        # Looking for lines like: "CPU Sequential      16.1201             1.0x"
        seq_match = re.search(r"CPU Sequential\s+([0-9.]+)", output)
        omp_match = re.search(r"CPU OpenMP\s+([0-9.]+)", output)
        cuda_match = re.search(r"GPU CUDA\s+([0-9.]+)", output)

        if not (seq_match and omp_match and cuda_match):
            raise HTTPException(status_code=500, detail=f"Failed to parse engine output: {output}")

        return {
            "success": True,
            "dataset": "Amazon Product Co-purchasing Network",
            "vertices": 262111,
            "edges": 1234877,
            "results": {
                "sequential": float(seq_match.group(1)),
                "openmp": float(omp_match.group(1)),
                "cuda": float(cuda_match.group(1))
            },
            "raw_output": output
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
