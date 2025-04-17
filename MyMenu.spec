# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['MyMenu.py'],
    pathex=[],
    binaries=[],
    datas=[('airport_2D.png', '.'), ('airport_3D.png', '.'), ('airport-comments.csv', '.'), ('airport-frequencies.csv', '.'), ('airports.csv', '.'), ('cleaned_data.pkl', '.'), ('cluster_distribution.txt', '.'), ('clusters_boxplot.png', '.'), ('countries.csv', '.'), ('DEM_topo.png', '.'), ('elbow_image.png', '.'), ('navaids.csv', '.'), ('regions.csv', '.'), ('runways.csv', '.')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='MyMenu',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
