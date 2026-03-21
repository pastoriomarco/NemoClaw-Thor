#!/usr/bin/env python3
"""Small TCP proxy for exposing the sandbox dashboard on localhost."""

from __future__ import annotations

import argparse
import signal
import socket
import threading
from contextlib import suppress


def relay(src: socket.socket, dst: socket.socket) -> None:
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        with suppress(OSError):
            dst.shutdown(socket.SHUT_WR)
        with suppress(OSError):
            src.close()
        with suppress(OSError):
            dst.close()


def handle_client(client: socket.socket, target_host: str, target_port: int) -> None:
    upstream = socket.create_connection((target_host, target_port))
    threading.Thread(target=relay, args=(client, upstream), daemon=True).start()
    threading.Thread(target=relay, args=(upstream, client), daemon=True).start()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--target-host", required=True)
    parser.add_argument("--target-port", type=int, required=True)
    args = parser.parse_args()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.listen_host, args.listen_port))
    server.listen()

    stop_event = threading.Event()

    def stop_handler(_signum: int, _frame: object) -> None:
        stop_event.set()
        with suppress(OSError):
            server.close()

    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)

    while not stop_event.is_set():
        try:
            client, _addr = server.accept()
        except OSError:
            break
        threading.Thread(
            target=handle_client,
            args=(client, args.target_host, args.target_port),
            daemon=True,
        ).start()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
