# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

from PyInstaller.utils.hooks import collect_submodules
import os, importlib

hidden_imports = collect_submodules('tensorflow.contrib')

package_imports = [['tensorboard', ['webfiles.zip']]]

added_files = []
for package, files in package_imports:
    proot = os.path.dirname(importlib.import_module(package).__file__)
    added_files.extend((os.path.join(proot, f), package) for f in files)

a = Analysis(['ml-agents/mlagents/trainers/learn.py'],
             pathex=['./'],
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
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='mlagents-learn',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )
