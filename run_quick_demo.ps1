$ErrorActionPreference = "Stop"
$script:Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $script:Root

$py = Join-Path $script:Root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "Virtual environment not found at .venv. Please run: py -3 -m venv .venv; .venv\Scripts\python.exe -m pip install -r requirements.txt" }

Write-Host "Running ETL..." -ForegroundColor Cyan
& $py -c "from src import etl; print(etl.run_etl(limit=10000, detect_lang=False))"

Write-Host "Running NLP (fast sample 3000)..." -ForegroundColor Cyan
& $py -c "from src import nlp; print(nlp.run_nlp(batch_size=64, sample=3000))"

Write-Host "Starting Streamlit on http://127.0.0.1:8509 ..." -ForegroundColor Green
Start-Process -FilePath $py -ArgumentList "-m streamlit run app\streamlit_app.py --server.address 127.0.0.1 --server.port 8509 --server.headless false --server.enableCORS false --server.enableXsrfProtection false" -WindowStyle Normal
