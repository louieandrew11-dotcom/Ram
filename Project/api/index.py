"""
api/index.py
Serverless function entry point for Vercel deployment.
"""

import sys
import os

# Set Matplotlib configuration directory to writable /tmp for serverless environments
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')

# Add project root directory to sys.path so app and ml_model can be imported
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Import the Flask application instance
from app import app
