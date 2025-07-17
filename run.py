import subprocess
import sys

def main():
    """Ejecuta la aplicación Streamlit"""
    cmd = [
        sys.executable, 
        "-m", 
        "streamlit", 
        "run", 
        "dashboard.py",
        "--server.port=8501",
        "--server.address=0.0.0.0"
    ]
    subprocess.run(cmd)

if __name__ == "__main__":
    main()