# =============================================================================
# VIDZEO LOCAL - Desktop Application Entry Point
# =============================================================================
# This file creates a native desktop window for the Flask app using PyWebView

import sys
import os
import threading
import time

# Ensure we can import from server directory
if getattr(sys, 'frozen', False):
    # Running as compiled EXE
    BASE_DIR = os.path.dirname(sys.executable)
else:
    # Running as script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Add server to path
sys.path.insert(0, os.path.join(BASE_DIR, 'server'))

# Set environment variable to indicate desktop mode
os.environ['VIDZEO_DESKTOP_MODE'] = '1'


def start_flask_server():
    """Start Flask server in background thread"""
    from app import app
    
    # Disable Flask's reloader and debugger for production
    app.run(
        host='127.0.0.1',
        port=5000,
        debug=False,
        use_reloader=False,
        threaded=True
    )


def wait_for_server(timeout=30):
    """Wait for Flask server to be ready"""
    import urllib.request
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            urllib.request.urlopen('http://127.0.0.1:5000/api/status', timeout=1)
            return True
        except:
            time.sleep(0.5)
    return False


def main():
    """Main entry point for desktop app"""
    import webview
    
    # Start Flask server in background thread
    server_thread = threading.Thread(target=start_flask_server, daemon=True)
    server_thread.start()
    
    print("Starting Vidzeo Local...")
    print("Waiting for server...")
    
    # Wait for server to be ready
    if not wait_for_server():
        print("ERROR: Server failed to start!")
        return 1
    
    print("Server ready! Opening window...")
    
    # Create native window
    window = webview.create_window(
        title='Vidzeo Local - AI Video Generator',
        url='http://127.0.0.1:5000',
        width=1400,
        height=900,
        min_size=(1200, 700),
        resizable=True,
        fullscreen=False,
        text_select=True,
        confirm_close=True
    )
    
    # Start the GUI (this blocks until window is closed)
    webview.start(
        debug=False,
        http_server=False
    )
    
    print("Window closed. Goodbye!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
