# WebCam Drawing

Webカメラに映した指先で空中に絵を描く。

## ダウンロード (Windows)
[Releases](https://github.com/hiroyamochi/WebCamDrawingPy/releases) から `WebCamDrawing-<バージョン>-windows.exe` を取得して起動する。
初回だけ `%LOCALAPPDATA%\WebCamDrawing` への展開に数十秒かかる。

## 操作
| 操作 | 動作 |
|---|---|
| 親指と人差し指をつまむ | 描く |
| 指先を色 / ✕ の上にとどめる | 色を選ぶ / 全消去 |
| `1`〜`4` | 色を選ぶ |
| `Space` | 全消去 |
| `↑` / `↓` | 操作検出感度を上げる / 下げる |
| `T` | 次のカメラに切り替え |

## ソースから実行 (Python 3.10 以降)
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python webcamdrawing.py
```
初回起動時に手検出モデル (`hand_landmarker.task`) を自動でダウンロードする。

## リリース
バージョンのタグを push すると、GitHub Actions が exe を作って Releases に置く。
exe は PyInstaller ではなく [launcher/Launcher.cs](launcher/Launcher.cs) の小さな起動用プログラムで、埋め込んだ Python 公式の embeddable Python を展開し、署名付きの pythonw.exe でアプリを起動する (PyInstaller の exe はウイルス対策ソフトに誤検知されやすいため)。
```
git tag v1.0.0
git push origin v1.0.0
```
手元で同じ exe を作る場合は (`dist/` に出力):
```
powershell -ExecutionPolicy Bypass -File build.ps1 -Version v1.0.0
```
