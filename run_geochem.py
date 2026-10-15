import os
import subprocess
import webbrowser
import threading
import time

def open_browser():
    time.sleep(5)
    webbrowser.open("http://localhost:8501")

threading.Thread(target=open_browser).start()

app = os.path.join(
    os.path.dirname(__file__),
    "geochem_app.py"
)

subprocess.run([
    "streamlit",
    "run",
    app,
    "--server.headless=true"
])
