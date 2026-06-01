import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit.web.cli as stcli

sys.argv = [
    "streamlit",
    "run",
    "app/streamlit_app.py",
    "--server.port=7860",
    "--server.address=0.0.0.0"
]

if __name__ == "__main__":
    sys.exit(stcli.main())