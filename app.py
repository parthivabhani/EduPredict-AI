#!/usr/bin/env python3
"""
EduPredict AI: Root Flask Entrypoint for Vercel Deployment
Author: Parthiv Abhani (PRN: 23070521106)
Semester 7 Data Science Mini Project
Under the Guidance of: Dr. Smita Singh
"""

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from web_app.server import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)
