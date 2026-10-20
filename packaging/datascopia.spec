# PyInstaller spec for a standalone Windows build of DataScopia.
#
# Build with (from the project root):
#   pyinstaller packaging/datascopia.spec --clean
# Output lands in dist_desktop/DataScopia/ (DataScopia.exe plus its _internal support files) -
# a different folder than dist/, which tools/build_static_site.py uses for the GitHub Pages
# frontend-only build; the two are unrelated and never collide.
#
# Why this looks more involved than a typical PyInstaller spec: run.py and wsgi.py load
# apps/api/app/main.py by reading its actual file from disk at startup (importlib's
# spec_from_file_location), not via a plain `import main` - a deliberate project convention so
# apps/ doesn't need __init__.py files to work as an importable package. PyInstaller's static
# analyzer can only discover the *compiled entry script's* imports, so it never sees main.py's
# dependencies at all unless told explicitly - hence the hiddenimports list below, which tells
# PyInstaller to analyze those modules too (as if the entry script had imported them), and the
# `datas` entries, which copy the real .py source for apps/api/app and the real static apps/web
# files, since both are read straight off disk at runtime (one via spec_from_file_location, the
# other via Flask's send_file/static_folder) rather than through the frozen import mechanism.

from __future__ import annotations

from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

# SPECPATH is injected by PyInstaller into the spec file's exec namespace (there's no __file__
# here - the spec isn't run as a normal module import).
PROJECT_ROOT = Path(SPECPATH).resolve().parent

# Local packages main.py imports normally (`from analysis.x import y`, `from dataset.x import y`,
# `import store`, etc.) - these ARE discoverable once their directories are on pathex, but only if
# something tells PyInstaller's analyzer to look at them in the first place. Explicitly analyzing
# "main" (and letting it pull in its own imports) covers the apps/api/app side; collect_submodules
# covers every analysis/dataset submodule even if one is only imported conditionally and so isn't
# reachable from main.py's top-level import statements alone.
#
# Third-party libraries this app actually imports (sklearn/scipy via analysis/report*.py,
# pyarrow via dataset/streaming.py, reportlab via analysis/report_pdf.py, google.auth/cryptography
# via apps/api/app/firebase.py) are deliberately NOT force-collected here with collect_all: it
# pulls in each library's entire test suite and demo/sample subpackages (pyarrow.tests alone is
# enormous), bloating the build for nothing actually used at runtime. Once "main" is a hidden
# import, PyInstaller's analyzer reaches every one of those real import statements on its own and
# applies pyinstaller-hooks-contrib's existing, already-tuned hooks for each library - no manual
# override needed. (kaggle/kagglehub are in requirements.txt but never actually imported as Python
# modules anywhere in this app - dataset/kaggle_search.py talks to Kaggle's REST API directly via
# requests - so they're correctly left out entirely rather than bundled for nothing.)
hiddenimports = ["main"] + collect_submodules("analysis") + collect_submodules("dataset")

datas = [
    (str(PROJECT_ROOT / "apps" / "web"), "apps/web"),
    (str(PROJECT_ROOT / "apps" / "api" / "app"), "apps/api/app"),
]

a = Analysis(
    [str(PROJECT_ROOT / "run.py")],
    pathex=[
        str(PROJECT_ROOT),
        str(PROJECT_ROOT / "apps" / "api" / "app"),
        str(PROJECT_ROOT / "packages" / "analysis" / "src"),
        str(PROJECT_ROOT / "packages" / "dataset" / "src"),
    ],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["kaggle", "kagglehub"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DataScopia",
    debug=False,
    strip=False,
    upx=False,
    console=True,  # shows "DataScopia is running at http://..." and any startup errors
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="DataScopia",
)
