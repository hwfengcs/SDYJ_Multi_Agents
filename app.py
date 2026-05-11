"""Hugging Face Spaces entry point.

Spaces commonly look for ``app.py`` at the repository root. Keep this wrapper
thin so local Streamlit, package Streamlit, and hosted Spaces all run the same
UI implementation.
"""

from SDYJ_Agents.web.app import main


if __name__ == "__main__":
    main()
