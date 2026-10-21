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

import ctypes
import sys
import threading
import winreg

import webview

import run

WINDOW_TITLE = "DataScopia"
_MUTEX_NAME = "Global\\DataScopiaDesktopSingleInstance"
ERROR_ALREADY_EXISTS = 183

# The WebView2 Runtime is what actually renders the app window (see module docstring). It's a
# standard Windows component - built into Windows 11 from release, and pushed to Windows 10 via
# Windows Update since 2022 - so this is normally a no-op, but an old/unpatched or locked-down
# Windows 10 machine could genuinely lack it. Microsoft's own documented way to detect it is this
# registry key, under any of these three locations depending on whether it was installed
# per-machine (32- or 64-bit registry view) or per-user. Checked explicitly, before webview even
# tries to start, purely so a machine that's genuinely missing it gets a message that tells it
# what to do, instead of a pythonnet/COM error that doesn't.
_WEBVIEW2_CLIENT_KEY = r"SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"
_WEBVIEW2_DOWNLOAD_URL = "https://developer.microsoft.com/microsoft-edge/webview2/"


def _webview2_installed() -> bool:
    for hive, subkey in (
        (winreg.HKEY_LOCAL_MACHINE, _WEBVIEW2_CLIENT_KEY),
        (winreg.HKEY_LOCAL_MACHINE, _WEBVIEW2_CLIENT_KEY.replace("Microsoft\\", "WOW6432Node\\Microsoft\\", 1)),
        (winreg.HKEY_CURRENT_USER, _WEBVIEW2_CLIENT_KEY),
    ):
        try:
            with winreg.OpenKey(hive, subkey):
                return True
        except OSError:
            continue
    return False


def _focus_existing_window() -> bool:
    """True if another DataScopia window was found and brought to the front.

    Not a plain FindWindowW(None, "DataScopia") exact-title lookup: the app changes the page
    (and so the native window's) title on navigation - e.g. "Visualization · DataScopia" -
    so only the initial launch screen is ever titled exactly "DataScopia". Enumerating windows
    and matching the suffix instead finds the real window regardless of which page it's on."""
    found = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def _enum_proc(hwnd, _lparam):
        length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return True
        buffer = ctypes.create_unicode_buffer(length + 1)
        ctypes.windll.user32.GetWindowTextW(hwnd, buffer, length + 1)
        if buffer.value == WINDOW_TITLE or buffer.value.endswith(f"· {WINDOW_TITLE}"):
            found.append(hwnd)
            return False
        return True

    ctypes.windll.user32.EnumWindows(_enum_proc, 0)
    if not found:
        return False
    SW_RESTORE = 9
    ctypes.windll.user32.ShowWindow(found[0], SW_RESTORE)
    ctypes.windll.user32.SetForegroundWindow(found[0])
    return True


def _acquire_single_instance_lock():
    """A named kernel mutex, not a lock file: Windows releases it automatically the instant the
    owning process exits for any reason (including a crash), so there's no stale-lock case to
    clean up - unlike a lock file, which a crash would leave behind, permanently blocking every
    future launch until a user manually deletes it. Returns the mutex handle (keep it referenced
    for the process lifetime - a GC'd handle releases the mutex) or None if another instance
    already holds it."""
    mutex = ctypes.windll.kernel32.CreateMutexW(None, False, _MUTEX_NAME)
    if ctypes.windll.kernel32.GetLastError() == ERROR_ALREADY_EXISTS:
        return None
    return mutex


def main() -> None:
    lock = _acquire_single_instance_lock()
    if lock is None:
        # Double-launching (e.g. double-clicking the exe twice) would otherwise start a second,
        # fully independent server + window rather than reusing the running one - confusing, and
        # wastes real memory (this app's own dependencies - pandas/numpy/scipy - aren't light).
        if not _focus_existing_window():
            ctypes.windll.user32.MessageBoxW(0, "DataScopia is already running.", WINDOW_TITLE, 0x40)
        return

    if not _webview2_installed():
        ctypes.windll.user32.MessageBoxW(
            0,
            "DataScopia needs the Microsoft Edge WebView2 Runtime to show its window, and this "
            "PC doesn't have it installed. Most Windows 10/11 PCs already have it - this is "
            "unusual.\n\nInstall it (one-time, free) from:\n" + _WEBVIEW2_DOWNLOAD_URL
            + "\n\nThen run DataScopia again.",
            WINDOW_TITLE,
            0x30,
        )
        return

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
            ctypes.windll.user32.MessageBoxW(0, str(exc), f"{WINDOW_TITLE} failed to start", 0x10)
        except Exception:  # noqa: BLE001
            pass
        print(exc, file=sys.stderr)
        raise SystemExit(1)
