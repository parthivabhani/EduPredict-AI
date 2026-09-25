#!/usr/bin/env python3
"""
EduPredict AI: Serverless Function Entrypoint for Vercel
Author: Parthiv Abhani (PRN: 23070521106)
Semester 7 Data Science Mini Project
Under the Guidance of: Dr. Smita Singh
"""

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from web_app.server import app
