# Grid watcher: when the current run_grade.py finishes, launches catch-up
# (suites spr+aux+gnn: retry of failed SPR + new CURL/CPC/ACL + rest
# of GNN), with resume — nothing is overwritten. Usage:
#   powershell -ExecutionPolicy Bypass -File jax_port/continue_grade.ps1
# Runs detached; track via <repo>/jax_port/results_grade/watcher.log
#
# Defaults derive from this script's location. Override with:
#   REPO               checkout root as a Windows path (default: parent of jax_port/)
#   PROCGEN_JAX_VENV   WSL-side path of the JAX venv (default: /root/procgen-jax)
$ErrorActionPreference = "Stop"

$repo = if ($env:REPO) { $env:REPO } else { Split-Path -Parent $PSScriptRoot }
$venv = if ($env:PROCGEN_JAX_VENV) { $env:PROCGEN_JAX_VENV } else { "/root/procgen-jax" }
# run_grade.py executes inside WSL, so it needs the DrvFS form of the checkout path
$repoWsl = (& wsl wslpath -u ($repo -replace '\\', '/')).Trim()
if (-not $repoWsl) { throw "could not translate '$repo' to a WSL path (is WSL available?)" }

$gradeDir = Join-Path $repo "jax_port\results_grade"
$log = Join-Path $gradeDir "watcher.log"
"watcher started $(Get-Date -Format 'HH:mm:ss')" | Out-File $log -Append
$calm = 0
while ($true) {
  Start-Sleep -Seconds 180
  $run = wsl -e pgrep -f run_grade.py 2>$null
  if ([string]::IsNullOrWhiteSpace("$run")) { $calm++ } else { $calm = 0 }
  "check $(Get-Date -Format 'HH:mm:ss') calm=$calm" | Out-File $log -Append
  if ($calm -ge 3) {
    "launching catch-up $(Get-Date -Format 'HH:mm:ss')" | Out-File $log -Append
    $argv = @('-e', 'env', "PYTHONPATH=$repoWsl", 'XLA_PYTHON_CLIENT_MEM_FRACTION=0.9',
              "$venv/bin/python", "$repoWsl/jax_port/run_grade.py",
              '--suite', 'spr', 'aux', 'gnn', '--seeds', '42', '43', '44', '45', '46',
              '--timesteps', '100000', '--eval-full',
              '--out-dir', "$repoWsl/jax_port/results_grade",
              '--master', "$repoWsl/jax_port/results_grade/master_full.json")
    Start-Process -FilePath "wsl" -ArgumentList $argv `
      -RedirectStandardOutput (Join-Path $gradeDir "grade_aux.log") `
      -RedirectStandardError (Join-Path $gradeDir "grade_aux.err.log") `
      -WorkingDirectory $repo
    "catch-up launched" | Out-File $log -Append
    break
  }
}
