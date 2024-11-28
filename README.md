## Moshi WebSocket Server and Frontend Integration


### Moshi WebSocket Server

```bash
uvicorn moshi_service:app --host 0.0.0.0 --port 8000
```

The server will start on `http://localhost:8000`. Use the `/ws` endpoint for WebSocket communication and `/status` to check the server's health.

---

### Frontend Server

The frontend server serves the web-based interface and interacts with the WebSocket server.

```bash
uvicorn frontend:app --host 0.0.0.0 --port 3000
```

The frontend will be served at `http://localhost:3000` (or as configured).

---

