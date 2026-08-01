#!/usr/bin/env python3
"""Serve the live-session page and proxy its websocket to the vLLM-Omni server.

Two jobs, and the second is the reason this exists rather than pointing the
browser straight at the engine:

* serve ``app/`` over HTTP;
* proxy ``/ws`` on this origin to ``/v1/video/chat/stream`` on the engine.

Same-origin proxying means a remote Mac forwards **one** port:

    ssh -N -L 7870:127.0.0.1:7870 <server>
    open http://localhost:7870/

and ``http://localhost`` is a secure context, so ``getUserMedia`` grants camera
and microphone without any certificate. Pointing the browser at the engine
directly would need a second forward and would put the engine's websocket on a
different origin.

The proxy is byte-transparent: it does not parse or rewrite the protocol. Both
directions are pumped concurrently, because this is a duplex stream -- frames and
audio go up while audio comes down, and a half-duplex pump would deadlock the
moment the model started speaking while the user was still talking.

    python server.py --port 7870 --ws-backend ws://127.0.0.1:8091
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys

# MODULE level, and it has to be. With `from __future__ import annotations` above,
# every annotation becomes a string, and FastAPI resolves `client: WebSocket` by
# looking "WebSocket" up in this module's globals. These imports used to live inside
# build_app, where they are locals -- so the lookup failed, FastAPI decided `client`
# must be a query parameter, found it missing, and closed the websocket with **HTTP
# 403 and an empty body**. The handler was never entered. Nothing was logged, the
# page served fine, and only the websocket died: a failure that reads as a proxy or
# a network problem and is neither.
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

APP_DIR = pathlib.Path(__file__).resolve().parent / "app"
UPSTREAM_PATH = "/v1/video/chat/stream"


def asset_version() -> str:
    """Short stamp over the client files, used to bust caches and to be checkable.

    A stale cached `app.js` or -- worse -- a stale AudioWorklet module makes a fixed
    playback bug look unfixed, and there is nothing in the UI to tell the two apart.
    `addModule()` in particular is cached hard, and a normal reload does not always
    replace it. So every asset URL carries this stamp, responses are `no-store`, and
    the page prints the stamp in its event log: if a fix seems absent, compare it with
    what the server reports at startup before looking anywhere else.
    """
    parts = []
    for name in ("index.html", "static/app.js", "static/styles.css",
                 "static/playback_worklet.js", "static/capture_worklet.js"):
        path = APP_DIR / name
        if path.exists():
            stat = path.stat()
            parts.append(f"{name}:{int(stat.st_mtime)}:{stat.st_size}")
    import hashlib

    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:8]


def build_app(ws_backend: str, ws_url_override: str | None):
    app = FastAPI(title="Qwen3-Omni live session")
    app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")

    version = asset_version()

    @app.middleware("http")
    async def no_store(request, call_next):
        response = await call_next(request)
        # This is a development tool served over an ssh tunnel; correctness beats a
        # cache hit on a 25 KB file every time.
        response.headers["Cache-Control"] = "no-store, must-revalidate"
        return response

    index_template = (APP_DIR / "index.html").read_text(encoding="utf-8")
    # Empty wsUrl makes the page derive ws://<this origin>/ws, which is what the
    # same-origin proxy is for. An override exists for the case where a reverse
    # proxy in front of this server does not forward websocket upgrades.
    page_config = json.dumps({"wsUrl": ws_url_override or "", "assetVersion": version})
    index_html = (index_template
                  .replace("{{CONFIG_JSON}}", page_config)
                  .replace('href="static/styles.css"', f'href="static/styles.css?v={version}"')
                  .replace('src="static/app.js"', f'src="static/app.js?v={version}"'))

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(index_html)

    @app.get("/healthz", response_class=PlainTextResponse)
    async def healthz() -> PlainTextResponse:
        return PlainTextResponse("ok")

    @app.websocket("/ws")
    async def proxy(client: WebSocket) -> None:
        import websockets

        await client.accept()
        target = f"{ws_backend.rstrip('/')}{UPSTREAM_PATH}"
        try:
            upstream = await websockets.connect(target, max_size=None, ping_interval=20)
        except Exception as exc:
            # Tell the browser in its own protocol's shape, so the page surfaces
            # it in the event log instead of showing a bare socket close that
            # looks identical to the model going quiet.
            await client.send_text(json.dumps({
                "type": "error",
                "message": f"cannot reach the engine at {target}: {exc}",
            }))
            await client.close()
            return

        async def up() -> None:
            try:
                while True:
                    await upstream.send(await client.receive_text())
            except (WebSocketDisconnect, Exception):
                pass

        async def down() -> None:
            try:
                async for message in upstream:
                    await client.send_text(message if isinstance(message, str) else message.decode())
            except Exception:
                pass

        pump_up = asyncio.create_task(up())
        pump_down = asyncio.create_task(down())
        try:
            # Whichever side ends first ends the session; the other is cancelled
            # rather than left pumping into a closed socket.
            await asyncio.wait({pump_up, pump_down}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in (pump_up, pump_down):
                task.cancel()
            await asyncio.gather(pump_up, pump_down, return_exceptions=True)
            with contextlib_suppress():
                await upstream.close()
            with contextlib_suppress():
                await client.close()

    return app


class contextlib_suppress:
    """Tiny local suppressor so teardown never raises over a real error."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7870)
    parser.add_argument("--ws-backend", default="ws://127.0.0.1:8091",
                        help="the vLLM-Omni server; /v1/video/chat/stream is appended")
    parser.add_argument("--public-ws-url", default=None,
                        help="only if a front proxy will not forward websocket upgrades")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("needs uvicorn and fastapi: pip install uvicorn fastapi websockets", file=sys.stderr)
        return 2

    print(f"page      http://{args.host}:{args.port}/", file=sys.stderr)
    print(f"proxying  /ws  ->  {args.ws_backend.rstrip('/')}{UPSTREAM_PATH}", file=sys.stderr)
    uvicorn.run(build_app(args.ws_backend, args.public_ws_url), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
