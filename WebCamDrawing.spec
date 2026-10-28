# -*- mode: python ; coding: utf-8 -*-
# ビルド: pyinstaller WebCamDrawing.spec  （事前に hand_landmarker.task をこのフォルダに置く）
from PyInstaller.utils.hooks import collect_all

# libmediapipe.dll を ctypes で読み込むので、パッケージ一式を同梱する
datas, binaries, hiddenimports = collect_all("mediapipe")
datas += [("hand_landmarker.task", ".")]

a = Analysis(
    ["webcamdrawing.py"],
    datas=datas,
    binaries=binaries,
    hiddenimports=hiddenimports,
)
# カメラ入力には使わない動画ファイル用の ffmpeg (約30MB) を外す
a.binaries = [b for b in a.binaries if "opencv_videoio_ffmpeg" not in b[0]]
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="WebCamDrawing",
    console=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="WebCamDrawing")
