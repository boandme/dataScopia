"""Build a static, GitHub-Pages-ready copy of the frontend into dist/.

GitHub Pages only serves static files - it can't run this project's Flask backend - so this script
doesn't build "the app"; it builds the *frontend*, wired to call a separately hosted backend (see
apps/web/gh-pages-config.js, the one file you edit to point it at your backend's URL).

What it does, concretely:
  1. Copies apps/web/index.html into dist/, rewriting its root-relative asset paths
     ("/assets/...") to relative ones ("assets/..."), since a GitHub Pages *project* site (the
     default, e.g. https://you.github.io/your-repo/) is served under a path prefix, not domain
     root - a root-relative path would resolve to the wrong place.
  2. Injects a <script> tag loading gh-pages-config.js before app.js, so window.KAGGLE_API_BASE
     (and nothing else) is set before app.js reads it.
  3. Copies every other apps/web/*.js and *.css file, plus gh-pages-config.js, into dist/assets/.
  4. Writes dist/.nojekyll so GitHub Pages serves the files as-is (Jekyll's default processing
     ignores files/folders starting with "_", which this project doesn't have, but it also adds
     needless build time for a site that's already fully built).

Run it locally to preview (`python -m http.server 8080 -d dist`, since the relative asset paths
need an actual HTTP server - opening dist/index.html directly as a file:// URL won't load them).
The GitHub Actions workflow (.github/workflows/gh-pages.yml) runs this same script automatically
on every push to main and deploys dist/ to GitHub Pages - no separate build step to remember.
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "apps" / "web"
DIST_DIR = ROOT / "dist"


def build() -> None:
    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)
    assets_dir = DIST_DIR / "assets"
    assets_dir.mkdir(parents=True)

    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    html = html.replace('href="/assets/', 'href="assets/').replace('src="/assets/', 'src="assets/')
    config_tag = '    <script src="assets/gh-pages-config.js"></script>\n'
    marker = '    <script src="assets/app.js"></script>'
    if marker not in html:
        raise RuntimeError("Could not find the app.js <script> tag to anchor the config script before - "
                            "apps/web/index.html's asset tags may have changed; update this script to match.")
    html = html.replace(marker, config_tag + marker)
    (DIST_DIR / "index.html").write_text(html, encoding="utf-8")

    for name in ("app.js", "social.js", "firebase-client.js", "styles.css", "gh-pages-config.js"):
        shutil.copyfile(WEB_DIR / name, assets_dir / name)

    (DIST_DIR / ".nojekyll").write_text("", encoding="utf-8")
    print(f"Built {DIST_DIR} ({sum(1 for _ in DIST_DIR.rglob('*') if _.is_file())} files).")


if __name__ == "__main__":
    build()
