"""
Run script for Cheating Detection Application
"""

import os
import sys

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the main module
from src.main import main

if __name__ == "__main__":
    # Run the main function
    main()
