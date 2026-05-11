"""Top-level Streamlit entry point.

Hugging Face Spaces (and most hosted Streamlit runners) look for an
``app.py`` or ``streamlit_app.py`` at the repository root. This file is the
thin wrapper that delegates to ``SDYJ_Agents.web.app.main``.

For local development, you can also point Streamlit at the package module
directly::

    streamlit run SDYJ_Agents/web/app.py

Both paths run the exact same UI.
"""

from SDYJ_Agents.web.app import main


if __name__ == "__main__":
    main()
