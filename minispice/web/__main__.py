"""`python -m minispice.web`: run the local development UI server.

Runs with Flask's debug reloader enabled, so editing any MiniSPICE
source file restarts the server automatically -- the next time you hit
"Simulate" in the browser, it's running your latest code. Binds to
localhost only; this is a local dev tool, not meant to be exposed on
the network.
"""

from .app import create_app

if __name__ == "__main__":
    app = create_app()
    print("MiniSPICE web UI: http://127.0.0.1:5000  (Ctrl+C to stop)")
    app.run(debug=True, host="127.0.0.1", port=5000)
