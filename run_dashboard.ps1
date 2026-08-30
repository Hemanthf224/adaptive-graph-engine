# Install Python Backend Dependencies
Write-Host "Installing Python Backend Dependencies..."
pip install fastapi uvicorn

# Start the Python Backend in the background
Write-Host "Starting FastAPI Backend..."
Start-Process -NoNewWindow -FilePath "python" -ArgumentList "-m uvicorn backend.main:app --host 127.0.0.1 --port 8000"

# Start the React Frontend
Write-Host "Starting React Frontend..."
Set-Location frontend
npm run dev
