param(
    [ValidateSet('start', 'stop', 'status')]
    [string]$Action = 'start'
)

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
$streamlitExe = Join-Path $projectRoot '.venv\Scripts\streamlit.exe'
$pidFile = Join-Path $projectRoot '.app-pids.json'

function Get-AppState {
    if (-not (Test-Path $pidFile)) {
        return $null
    }

    try {
        return Get-Content $pidFile -Raw | ConvertFrom-Json
    }
    catch {
        return $null
    }
}

function Save-AppState {
    param(
        [int]$BackendPid,
        [int]$FrontendPid
    )

    @{ backend = $BackendPid; frontend = $FrontendPid } | ConvertTo-Json | Set-Content $pidFile
}

function Stop-AppProcess {
    param(
        [int]$ProcessId
    )

    if ($ProcessId -and (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)) {
        Stop-Process -Id $ProcessId -Force
    }
}

switch ($Action) {
    'start' {
        if (-not (Test-Path $pythonExe)) {
            throw "Python not found at $pythonExe. Create the local virtual environment first."
        }

        if (-not (Test-Path $streamlitExe)) {
            throw "Streamlit not found at $streamlitExe. Install dependencies into .venv first."
        }

        $existing = Get-AppState
        if ($existing) {
            $backendAlive = $existing.backend -and (Get-Process -Id $existing.backend -ErrorAction SilentlyContinue)
            $frontendAlive = $existing.frontend -and (Get-Process -Id $existing.frontend -ErrorAction SilentlyContinue)
            if ($backendAlive -or $frontendAlive) {
                Write-Host 'App already appears to be running. Use .\manage-app.ps1 stop first if needed.'
                break
            }
        }

        $backend = Start-Process -FilePath $pythonExe -ArgumentList @('-m', 'uvicorn', 'backend:app', '--host', '127.0.0.1', '--port', '8000') -WorkingDirectory $projectRoot -PassThru
        Start-Sleep -Seconds 2
        $frontend = Start-Process -FilePath $streamlitExe -ArgumentList @('run', 'app.py', '--server.address', '127.0.0.1', '--server.port', '8501') -WorkingDirectory $projectRoot -PassThru

        Save-AppState -BackendPid $backend.Id -FrontendPid $frontend.Id

        Write-Host 'Backend:  http://127.0.0.1:8000/'
        Write-Host 'Frontend: http://127.0.0.1:8501/'
        Write-Host "Saved process ids to $pidFile"
    }

    'stop' {
        $state = Get-AppState
        if ($state) {
            Stop-AppProcess -ProcessId $state.frontend
            Stop-AppProcess -ProcessId $state.backend
        }

        foreach ($port in 8501, 8000) {
            Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty OwningProcess |
            Sort-Object -Unique |
            ForEach-Object { Stop-AppProcess -ProcessId $_ }
        }

        if (Test-Path $pidFile) {
            Remove-Item $pidFile -Force
        }

        Write-Host 'Local app stopped.'
    }

    'status' {
        $state = Get-AppState
        if (-not $state) {
            Write-Host 'No saved app state found.'
            exit 0
        }

        Write-Host "Backend PID:  $($state.backend)"
        Write-Host "Frontend PID: $($state.frontend)"
    }
}