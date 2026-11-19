import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SITE_PACKAGES = _PROJECT_ROOT / ".venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if _SITE_PACKAGES.is_dir() and str(_SITE_PACKAGES) not in sys.path:
    sys.path.insert(0, str(_SITE_PACKAGES))
