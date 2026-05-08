"""Run SDYJ local release-readiness gates.

This wrapper mirrors ``sdyj release-check`` for contributors who prefer a
script path.
"""

from SDYJ_Agents.release_check import main


if __name__ == "__main__":
    raise SystemExit(main())
