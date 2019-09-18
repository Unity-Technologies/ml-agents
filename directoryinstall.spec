# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

from PyInstaller.utils.hooks import collect_submodules

hidden_imports = collect_submodules('tensorflow.contrib')

added_files = [
         ( 'config', 'config' ),
         ( 'demos', 'demos' ),
         ]

a = Analysis(['ml-agents/mlagents/trainers/learn.py'],
             pathex=['/Users/ervin/Development/ml-agents-develop'],
             binaries=[],
             datas=added_files,
             hiddenimports=hidden_imports,
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='mlagents-learn',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='testonedir')
