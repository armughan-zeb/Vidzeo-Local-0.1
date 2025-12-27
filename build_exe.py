# =============================================================================
# VIDZEO LOCAL - PyInstaller Build Script
# =============================================================================
# Run this script to build the desktop application EXE
#
# Usage: python build_exe.py
# =============================================================================

import PyInstaller.__main__
import os
import sys
import shutil

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIST_DIR = os.path.join(BASE_DIR, 'dist')
BUILD_DIR = os.path.join(BASE_DIR, 'build')

def clean_build():
    """Clean previous build artifacts"""
    for d in [DIST_DIR, BUILD_DIR]:
        if os.path.exists(d):
            print(f"Cleaning {d}...")
            shutil.rmtree(d)

def build_exe():
    """Build the single-file EXE"""
    print("=" * 60)
    print("  VIDZEO LOCAL - Building Desktop Application")
    print("=" * 60)
    print()
    
    # PyInstaller arguments
    args = [
        'desktop_app.py',                    # Entry point
        '--name=Vidzeo Local',               # EXE name
        '--onefile',                         # Single EXE
        '--windowed',                        # No console window
        '--noconfirm',                       # Overwrite without asking
        
        # Add data files
        '--add-data=server;server',          # Flask app
        '--add-data=public;public',          # Frontend files
        '--add-data=fonts;fonts',            # Font files
        
        # Hidden imports for Flask and dependencies
        '--hidden-import=flask',
        '--hidden-import=flask_cors',
        '--hidden-import=werkzeug',
        '--hidden-import=jinja2',
        '--hidden-import=markupsafe',
        '--hidden-import=itsdangerous',
        '--hidden-import=click',
        '--hidden-import=blinker',
        
        # Hidden imports for services
        '--hidden-import=numpy',
        '--hidden-import=soundfile',
        '--hidden-import=PIL',
        '--hidden-import=pydub',
        '--hidden-import=requests',
        '--hidden-import=groq',
        '--hidden-import=openai',
        
        # PyWebView
        '--hidden-import=webview',
        '--hidden-import=webview.platforms.winforms',
        
        # Exclude large unnecessary packages
        '--exclude-module=torch',            # We'll handle TTS separately
        '--exclude-module=whisper',          # We'll handle Whisper separately
        '--exclude-module=kokoro',           # We'll handle Kokoro separately
        
        # Paths
        f'--distpath={DIST_DIR}',
        f'--workpath={BUILD_DIR}',
    ]
    
    print("Building EXE with PyInstaller...")
    print()
    
    PyInstaller.__main__.run(args)
    
    exe_path = os.path.join(DIST_DIR, 'Vidzeo Local.exe')
    
    if os.path.exists(exe_path):
        size_mb = os.path.getsize(exe_path) / (1024 * 1024)
        print()
        print("=" * 60)
        print(f"  SUCCESS! Built: {exe_path}")
        print(f"  Size: {size_mb:.1f} MB")
        print("=" * 60)
        return True
    else:
        print("ERROR: Build failed!")
        return False


if __name__ == '__main__':
    clean_build()
    success = build_exe()
    sys.exit(0 if success else 1)
