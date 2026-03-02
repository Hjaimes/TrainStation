"""Web UI entry point. Starts FastAPI server and opens browser (or desktop window)."""
import argparse
import sys
import webbrowser


def main():
    parser = argparse.ArgumentParser(description="AI Model Training UI")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8675, help="Port (default: 8675)")
    parser.add_argument("--desktop", action="store_true",
                        help="Open in a native desktop window via pywebview instead of browser")
    parser.add_argument("--no-open", action="store_true", help="Don't auto-open browser")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("uvicorn not installed. Run: pip install uvicorn[standard]", file=sys.stderr)
        sys.exit(1)

    url = f"http://{args.host}:{args.port}"

    if args.desktop:
        try:
            import webview
        except ImportError:
            print("pywebview not installed. Run: pip install pywebview", file=sys.stderr)
            sys.exit(1)
        import threading
        server_thread = threading.Thread(
            target=uvicorn.run, args=("ui.server:app",),
            kwargs={"host": args.host, "port": args.port, "log_level": "warning"},
            daemon=True)
        server_thread.start()
        webview.create_window("AI Trainer", url, width=1280, height=800)
        webview.start()
    else:
        if not args.no_open:
            import threading
            threading.Timer(1.5, webbrowser.open, args=[url]).start()
        uvicorn.run("ui.server:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
