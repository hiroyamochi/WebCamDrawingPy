# 配布用の exe を作る（Windows）
#   Python 公式の embeddable Python + 依存パッケージ + アプリ本体を zip にまとめ、
#   起動用 exe (launcher\Launcher.cs) に埋め込む。PyInstaller は使わない
#   （PyInstaller の exe はウイルス対策ソフトに誤検知されやすいため）
# 使い方: .\build.ps1 -Version v1.0.0   （PATH 上の python と同じバージョンの embeddable Python を使う）
param([string]$Version = "dev")
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"  # 進捗表示があるとダウンロードが極端に遅い

$root = $PSScriptRoot
$dist = Join-Path $root "dist"
$stage = Join-Path $dist "stage"
$model = Join-Path $root "hand_landmarker.task"
$pyver = python -c "import platform; print(platform.python_version())"

if (Test-Path $dist) { Remove-Item -Recurse -Force $dist }
New-Item -ItemType Directory -Force "$stage\python", "$stage\app" | Out-Null

# embeddable Python（python.org の署名付き pythonw.exe）
$embed = Join-Path $dist "python-embed.zip"
Invoke-WebRequest -UseBasicParsing "https://www.python.org/ftp/python/$pyver/python-$pyver-embed-amd64.zip" -OutFile $embed
Expand-Archive $embed "$stage\python"
Remove-Item $embed
# ._pth の import site を有効にして Lib\site-packages を読めるようにする
$pth = Get-ChildItem "$stage\python\python*._pth"
(Get-Content $pth) -replace '^#\s*import site', 'import site' | Set-Content $pth

python -m pip install --disable-pip-version-check --target "$stage\python\Lib\site-packages" -r (Join-Path $root "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

if (-not (Test-Path $model)) {
    Invoke-WebRequest -UseBasicParsing "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" -OutFile $model
}
Copy-Item (Join-Path $root "webcamdrawing.py"), $model "$stage\app"

# 起動用 exe に埋め込む zip
$payload = Join-Path $dist "payload.zip"
python -c "import shutil, sys; shutil.make_archive(sys.argv[1][:-4], 'zip', sys.argv[2])" $payload $stage
if ($LASTEXITCODE -ne 0) { throw "zip failed" }

# exe のバージョン情報（タグ v1.2.3 -> 1.2.3）
$num = if ($Version -match '^v?(\d+\.\d+\.\d+)') { $Matches[1] } else { "0.0.0" }
$info = Join-Path $dist "AssemblyInfo.cs"
Set-Content $info -Encoding utf8 -Value @"
using System.Reflection;
[assembly: AssemblyTitle("WebCam Drawing")]
[assembly: AssemblyProduct("WebCam Drawing")]
[assembly: AssemblyDescription("Draw in the air with your fingertip")]
[assembly: AssemblyVersion("$num.0")]
[assembly: AssemblyFileVersion("$num.0")]
[assembly: AssemblyInformationalVersion("$Version")]
"@

$csc = Join-Path $env:WINDIR "Microsoft.NET\Framework64\v4.0.30319\csc.exe"
& $csc /nologo /optimize /target:winexe `
    "/out:$(Join-Path $dist "WebCamDrawing-$Version-windows.exe")" `
    "/resource:$payload,payload.zip" `
    /r:System.IO.Compression.dll /r:System.IO.Compression.FileSystem.dll /r:System.Windows.Forms.dll `
    (Join-Path $root "launcher\Launcher.cs") $info
if ($LASTEXITCODE -ne 0) { throw "csc failed" }

Remove-Item -Recurse -Force $stage, $payload, $info
