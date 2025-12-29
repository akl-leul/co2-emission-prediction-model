import webview
import threading
import os
import sys
import subprocess
import webbrowser
from pathlib import Path

# Path to the Streamlit app
STREAMLIT_APP = "app.py"

class App:
    def __init__(self):
        self.server_process = None
        self.window = None

    def run_server(self):
        # Run Streamlit in server mode
        cmd = ["streamlit", "run", STREAMLIT_APP, "--server.headless", "true", "--server.port", "8501"]
        self.server_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start
        import time
        time.sleep(2)  # Give the server a moment to start

    def on_closed(self):
        # Clean up the server process when the window is closed
        if self.server_process:
            self.server_process.terminate()
            self.server_process = None

def start_desktop_app():
    # Set up the application
    app = App()
    
    # Start the Streamlit server in a separate thread
    server_thread = threading.Thread(target=app.run_server)
    server_thread.daemon = True
    server_thread.start()
    
    # Create the webview window
    app.window = webview.create_window(
        'CO2 Charcoal Emission Analyzer',
        'https://co2-prediction-addis-ababa.streamlit.app/',
        width=1200,
        height=800,
        min_size=(800, 600),
        on_top=True
    )
    
    # Set up cleanup on window close
    app.window.events.closed += app.on_closed
    
    # Start the webview
    webview.start()

if __name__ == '__main__':
    start_desktop_app()