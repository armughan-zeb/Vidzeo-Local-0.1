# =============================================================================
# VIDZEO LOCAL - Simple Launcher (for PyInstaller)
# =============================================================================
# This is a minimal launcher that runs the installer script
# It can be compiled to EXE without heavy AI dependencies

import os
import sys
import subprocess
import ctypes

def is_admin():
    """Check if running as administrator"""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def get_base_dir():
    """Get the base directory (where EXE is located)"""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

def show_message(title, message, icon=0x40):
    """Show Windows message box"""
    ctypes.windll.user32.MessageBoxW(0, message, title, icon)

def main():
    base_dir = get_base_dir()
    
    # Check if INSTALL_AND_RUN.bat exists
    install_bat = os.path.join(base_dir, 'INSTALL_AND_RUN.bat')
    
    if not os.path.exists(install_bat):
        show_message(
            "Vidzeo Local - Error",
            f"INSTALL_AND_RUN.bat not found in:\n{base_dir}\n\nPlease extract all files from the ZIP.",
            0x10  # Error icon
        )
        return 1
    
    # Run the batch file
    try:
        subprocess.Popen(
            ['cmd', '/c', 'start', '', install_bat],
            cwd=base_dir,
            shell=True
        )
    except Exception as e:
        show_message(
            "Vidzeo Local - Error",
            f"Failed to start: {e}",
            0x10
        )
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
