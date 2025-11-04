import os
import sys
import runpy
from pathlib import Path


def main() -> int:
    """Change CWD to the `webapp` directory and run `app.py` as a script.

    This ensures `webapp` is the current directory (so relative static/templates
    paths in `webapp/app.py` work correctly) and then executes the script.
    """
    repo_root = Path(__file__).resolve().parent
    webapp_dir = repo_root / "webapp"

    if not webapp_dir.exists():
        print(f"Error: webapp directory not found at {webapp_dir}", file=sys.stderr)
        return 2

    # Change working directory so webapp/app.py can find its static/templates
    os.chdir(str(webapp_dir))

    # Ensure the `webapp` directory is first on sys.path so top-level imports
    # inside `app.py` (like `from session_manager import ...`) resolve correctly
    # even when this script was started from the repository root.
    webapp_path = str(webapp_dir.resolve())
    if sys.path[0] != webapp_path:
        sys.path.insert(0, webapp_path)

    # Execute app.py as __main__ (this will start uvicorn if app.py does so)
    try:
        runpy.run_path(str(webapp_dir / "app.py"), run_name="__main__")
    except Exception as e:
        # Re-raise keyboard interrupts so the process can be stopped normally
        if isinstance(e, KeyboardInterrupt):
            raise
        print(f"Failed to run webapp/app.py: {e}", file=sys.stderr)
        return 3

    return 0


if __name__ == "__main__":
    sys.exit(main())
