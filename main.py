"""
Main entry point for Agentic Memory Systems: Progressive Examples & Inspector.
"""
import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.cli import main_menu

if __name__ == "__main__":
    main_menu()
