#!/usr/bin/env python3
"""Native desktop entry point for DataScopia.

This is what the packaged .exe (packaging/datascopia.spec) actually runs - run.py's own
`python run.py` still opens the app in your default browser, which stays the right choice for
anyone running from source who'd rather not pull in pywebview. This file instead shows the app
in its own window (via pywebview, using the WebView2 runtime Windows 10/11 already ships with -
no bundled Chromium, no extra install step): a real "DataScopia" entry in the taskbar and
Alt-Tab, its own icon, and closing that window actually quits the app, instead of a browser tab
sitting on top of a separate always-running console process.

Usage:
    python desktop_app.py
"""

from __future__ import annotations

import sys
import threading

import webview

import run

WINDOW_TITLE = "DataScopia"


def main() -> None:
    server, url = run.start_server()

    # daemon=True: if anything goes wrong with the window below, this thread dying with the
    # process is correct - there's no "server outlives the window" case for a desktop app.
    threading.Thread(target=server.serve_forever, daemon=True).start()

    window = webview.create_window(WINDOW_TITLE, url, width=1360, height=860, min_size=(960, 640))

    def _on_closed() -> None:
        # pywebview's own event loop owns the process from here; shutting the server down
        # explicitly (rather than just letting the daemon thread die with the process) avoids a
        # moment where the OS still shows the port as in use right as the window closes.
        server.shutdown()

    window.events.closed += _on_closed
    webview.start()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        # Console=False in the packaged build means there's no terminal to show this in - a
        # dialog is the only way a non-technical user would ever see why the app didn't start
        # (most likely cause in practice: the WebView2 runtime genuinely missing, not something
        # the app can fix for them - see README for the one-time runtime installer link).
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(0, str(exc), f"{WINDOW_TITLE} failed to start", 0x10)
        except Exception:  # noqa: BLE001
            pass
        print(exc, file=sys.stderr)
        raise SystemExit(1)
