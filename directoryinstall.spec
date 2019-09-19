# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

from PyInstaller.utils.hooks import collect_submodules
import os, importlib

package_imports = [['tensorboard', ['webfiles.zip']]]

added_files = [( 'config', 'config' ),( 'demos', 'demos' )]

for package, files in package_imports:
    proot = os.path.dirname(importlib.import_module(package).__file__)
    added_files.extend((os.path.join(proot, f), package) for f in files)

hidden_imports = collect_submodules('tensorflow.contrib')

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
               name='mlagents-dir')
