# smoke testing 

import asyncio
from pathlib import Path
import os
import aiohttp
import time

endpoint = "ws://localhost:8000/"

shutdown_flag = asyncio.Event()

WARMUP_TIMEOUT = 60  # give server 60 seconds to warm up


async def ensure_server_ready():
    """Ensure the server is ready to accept connections."""
    deadline = time.time() + WARMUP_TIMEOUT
    async with aiohttp.ClientSession() as session:
        while time.time() < deadline:
            try:
                print("Checking server status...")
                resp = await session.get(endpoint + "status")
                if resp.status == 200:
                    print("Server is ready.")
                    return
                else:
                    print(f"Unexpected server status: {resp.status}")
            except Exception as e:
                print(f"Error while checking server status: {e}")
                await asyncio.sleep(5)
        raise RuntimeError("Server did not become ready within the timeout period.")


async def run():
    print("Starting smoke test...")
    print("Ensuring server is ready...")
    await ensure_server_ready()

    send_chunks = []
    recv_chunks = []

    print("Loading input chunks...")
    files = os.listdir(Path(__file__).parent / "e2e_in")
    files.sort()
    for chunk_file in files:
        filepath = Path(__file__).parent / "e2e_in" / chunk_file
        with open(filepath, "rb") as f:
            data = f.read()
            send_chunks.append(data)
        print(f"Loaded chunk: {chunk_file}, size: {len(data)} bytes")
    print(f"Total chunks loaded: {len(send_chunks)}")

    print(f"Connecting to WebSocket endpoint: {endpoint + 'ws'}")
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(endpoint + "ws") as ws:
            print("Connection established.")

            async def send_loop():
                for i, chunk in enumerate(send_chunks):
                    await asyncio.sleep(0.1)
                    await ws.send_bytes(chunk)
                    print(f"Sent chunk {i + 1}/{len(send_chunks)}, size: {len(chunk)} bytes")

            async def recv_loop():
                async for msg in ws:
                    if shutdown_flag.is_set():
                        print("Shutdown flag detected, stopping receive loop.")
                        break
                    data = msg.data
                    if not isinstance(data, bytes):
                        print("Received non-bytes message, ignoring.")
                        continue
                    if len(data) == 0:
                        print("Received empty message, ignoring.")
                        continue
                    print(f"Received chunk, size: {len(data)} bytes")
                    recv_chunks.append(data)

            async def timeout_loop():
                print("Timeout loop started, will close WebSocket after 10 seconds.")
                await asyncio.sleep(10)
                shutdown_flag.set()
                print("Timeout reached, closing WebSocket.")
                await ws.close()

            print("Starting send, receive, and timeout loops.")
            await asyncio.gather(send_loop(), recv_loop(), timeout_loop())
            print("All loops completed.")

    print(f"Total received chunks: {len(recv_chunks)}")
    if len(recv_chunks) < 3:
        print("Error: Not enough chunks received for a healthy inference.")
        raise AssertionError("Received fewer than 3 chunks.")

    print("Smoke test completed successfully.")


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except Exception as e:
        print(f"Smoke test failed with exception: {e}")
    else:
        print("Smoke test passed successfully.")
