"""
run_app.py  --  start the SD-XTG GUI.

    python run_app.py            (terminal / Anaconda Prompt)
    Spyder: open run_app.py OR sdxtg_app.py and press Run (F5)
    streamlit run sdxtg_app.py   (the classic way, still works)

A Streamlit app cannot be executed by Spyder's Run button directly: that runs it in "bare mode",
prints "missing ScriptRunContext" and shows nothing.  This launcher starts the real Streamlit
server in a separate process, waits until it answers, and opens the GUI in your browser.
Everything the server prints goes to sdxtg_app.log next to this file.
"""
import builtins
import os
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "sdxtg_app.log")


def _free_port(start=8501, tries=50):
    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("no free port between 8501 and 8550")


def _tail(n=25):
    try:
        with open(LOG, encoding="utf-8", errors="replace") as fh:
            return "".join(fh.readlines()[-n:])
    except OSError:
        return "(no log)"


def _check_deps():
    missing = []
    for mod in ("streamlit", "plotly", "numpy", "matplotlib"):
        try:
            __import__(mod)
        except Exception as exc:
            missing.append(mod)
            print(f"  import {mod} failed: {type(exc).__name__}: {exc}")
    if missing:
        print("\nThe Python running this script cannot import: " + ", ".join(missing))
        print("Python used:", sys.executable)
        print("Install them into THIS Python with:\n")
        print(f'    "{sys.executable}" -m pip install ' + " ".join(missing) + "\n")
        print("(In Spyder: Tools > Preferences > Python interpreter shows which Python is used.)")
        return False
    import streamlit
    print(f"Python {sys.version.split()[0]} at {sys.executable}, Streamlit {streamlit.__version__}")
    return True


def _in_spyder():
    return any(k.startswith("SPYDER") for k in os.environ) or "spyder_kernels" in sys.modules


def main(block=None):
    if not _check_deps():
        return None
    # a previous launch from the same Spyder console: stop it first
    old = getattr(builtins, "_sdxtg_server", None)
    if old is not None and old.poll() is None:
        old.terminate()
        time.sleep(1.0)

    port = _free_port()
    url = f"http://localhost:{port}"
    cmd = [sys.executable, "-m", "streamlit", "run", os.path.join(HERE, "sdxtg_app.py"),
           "--server.headless", "true", "--server.port", str(port),
           "--browser.gatherUsageStats", "false"]
    log = open(LOG, "w", encoding="utf-8")
    kw = {}
    if os.name == "nt":
        kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.Popen(cmd, cwd=HERE, stdin=subprocess.DEVNULL, stdout=log,
                            stderr=subprocess.STDOUT, **kw)
    builtins._sdxtg_server = proc

    print(f"Starting the SD-XTG GUI on {url} ...")
    t0 = time.time()
    while True:
        if proc.poll() is not None:
            print("\nStreamlit stopped before the GUI came up. Last lines of", LOG, ":\n")
            print(_tail())
            return None
        try:
            urllib.request.urlopen(url + "/_stcore/health", timeout=1).read()
            break
        except Exception:
            if time.time() - t0 > 60:
                print("\nNo answer after 60 s. Last lines of", LOG, ":\n")
                print(_tail())
                proc.terminate()
                return None
            time.sleep(0.5)

    print(f"GUI is running: {url}  (opening your browser)")
    webbrowser.open(url)

    if block is None:
        block = not _in_spyder()
    if not block:
        print("Spyder: the GUI keeps running in the background. Press Run again to restart it,")
        print("or restart the Spyder console (Consoles > Restart kernel) to stop it.")
        return proc
    print("Press Ctrl+C here to stop it.")
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
    return proc


if __name__ == "__main__":
    main()
