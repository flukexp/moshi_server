import asyncio
import torch
from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
from moshi.models import loaders, LMGen
import sentencepiece
import sphn
import numpy as np

# Initialize the FastAPI app
app = FastAPI()

class MoshiService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.frame_size = None
        self.mimi = None
        self.lm_gen = None
        self.text_tokenizer = None
        self.opus_stream_outbound = None
        self.opus_stream_inbound = None

    def initialize(self):
        # Download models
        mimi_weight = hf_hub_download("kyutai/moshika-pytorch-bf16", loaders.MIMI_NAME)
        self.mimi = loaders.get_mimi(mimi_weight, device=self.device)
        self.mimi.set_num_codebooks(8)
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)

        moshi_weight = hf_hub_download("kyutai/moshika-pytorch-bf16", loaders.MOSHI_NAME)
        self.moshi = loaders.get_moshi_lm(moshi_weight, device=self.device)
        self.lm_gen = LMGen(
            self.moshi,
            temp=0.8,
            temp_text=0.8,
            top_k=250,
            top_k_text=25,
        )
        
        self.mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)

        tokenizer_config = hf_hub_download(
            "kyutai/moshika-pytorch-bf16", loaders.TEXT_TOKENIZER_NAME
        )
        self.text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_config)

        # Warm up the models
        for _ in range(4):
            chunk = torch.zeros(
                1, 1, self.frame_size, dtype=torch.float32, device=self.device
            )
            codes = self.mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c : c + 1])
                if tokens is None:
                    continue
                _ = self.mimi.decode(tokens[:, 1:])
        if self.device == "cuda":
            torch.cuda.synchronize()

    def reset_state(self):
        self.opus_stream_outbound = sphn.OpusStreamWriter(self.mimi.sample_rate)
        self.opus_stream_inbound = sphn.OpusStreamReader(self.mimi.sample_rate)
        self.mimi.reset_streaming()
        self.lm_gen.reset_streaming()


# Initialize the MoshiService instance
moshi = MoshiService()
moshi.initialize()

@app.get("/status")
async def status():
    return Response(status_code=200)

@app.websocket("/ws")
async def websocket(ws: WebSocket):
    with torch.no_grad():
        await ws.accept()
        moshi.reset_state()
        print("Session started")
        tasks = []
        websocket_closed = False

        async def recv_loop():
            nonlocal websocket_closed
            try:
                while not websocket_closed:
                    data = await ws.receive_bytes()
                    if not data:
                        print("Received empty message")
                        continue
                    moshi.opus_stream_inbound.append_bytes(data)
            except WebSocketDisconnect:
                websocket_closed = True
                print("Client disconnected in recv_loop")

        async def inference_loop():
            nonlocal websocket_closed
            all_pcm_data = None
            try:
                while not websocket_closed:
                    await asyncio.sleep(0.001)
                    pcm = moshi.opus_stream_inbound.read_pcm()
                    if pcm is None or pcm.size == 0:
                        continue

                    if all_pcm_data is None:
                        all_pcm_data = pcm
                    else:
                        all_pcm_data = np.concatenate((all_pcm_data, pcm))

                    while all_pcm_data.shape[-1] >= moshi.frame_size:
                        chunk = all_pcm_data[: moshi.frame_size]
                        all_pcm_data = all_pcm_data[moshi.frame_size:]
                        chunk = torch.from_numpy(chunk).to(device=moshi.device)[None, None]

                        codes = moshi.mimi.encode(chunk)
                        for c in range(codes.shape[-1]):
                            tokens = moshi.lm_gen.step(codes[:, :, c:c+1])
                            if tokens is None:
                                continue

                            main_pcm = moshi.mimi.decode(tokens[:, 1:]).cpu()
                            moshi.opus_stream_outbound.append_pcm(main_pcm[0, 0].detach().numpy())

                            text_token = tokens[0, 0, 0].item()
                            if text_token not in (0, 3):
                                text = moshi.text_tokenizer.id_to_piece(text_token).replace("▁", " ")
                                await ws.send_bytes(b"\x02" + bytes(text, "utf-8"))
            except Exception as e:
                websocket_closed = True
                print(f"Inference loop error: {e}")

        async def send_loop():
            nonlocal websocket_closed
            try:
                while not websocket_closed:
                    await asyncio.sleep(0.001)
                    msg = moshi.opus_stream_outbound.read_bytes()
                    if msg is None or len(msg) == 0:
                        continue
                    await ws.send_bytes(b"\x01" + msg)
            except WebSocketDisconnect:
                websocket_closed = True
                print("Send loop disconnected")

        try:
            tasks = [
                asyncio.create_task(recv_loop()),
                asyncio.create_task(inference_loop()),
                asyncio.create_task(send_loop()),
            ]
            await asyncio.gather(*tasks)

        except WebSocketDisconnect:
            print("WebSocket disconnected")
        except Exception as e:
            print(f"Unexpected error: {e}")
        finally:
            websocket_closed = True
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            moshi.reset_state()
            print("Session ended")
