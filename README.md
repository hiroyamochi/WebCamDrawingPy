# WebCam Drawing

Webカメラに映した指先で空中に絵を描く。

## ダウンロード (Windows)
[Releases](https://github.com/hiroyamochi/WebCamDrawingPy/releases) から `WebCamDrawing-<バージョン>-windows.zip` を取得し、展開した中の `WebCamDrawing.exe` を起動する。

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
バージョンのタグを push すると、GitHub Actions が exe をビルドして Releases に置く。
```
git tag v1.0.0
git push origin v1.0.0
```
手元でビルドする場合は `hand_landmarker.task` を置いた状態で:
```
pip install pyinstaller
pyinstaller WebCamDrawing.spec
```
