import os
import sys
import asyncio
import wave
import logging
import io
from enum import Enum
import json
import re

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import torch
import numpy as np
from groq import Groq
from elevenlabs.client import ElevenLabs

VAD_SPEECH_THRESHOLD = 0.5
END_OF_SPEECH_SILENCE_SECONDS = 0.8
RATE = 16000
VAD_CHUNK_SIZE = 512
ELEVENLABS_VOICE_ID = "N2lVS1w4EtoT3dr4eOWO"
ELEVENLABS_MODEL_ID = "eleven_flash_v2_5"

class AssistantState(Enum):
    LISTENING_FOR_WAKE_WORD = 1
    RECORDING_USER_SPEECH = 2
    PROCESSING = 3
    SPEAKING = 4

STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

app = FastAPI()
app.mount(f"/{STATIC_DIR}", StaticFiles(directory=STATIC_DIR), name="static")

vad_model = None
utils = None
groq_client = None
eleven_client = None

@app.on_event("startup")
async def startup_event():
    global vad_model, utils, groq_client, eleven_client
    try:
        logging.info("Loading Silero VAD model...")
        vad_model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad', model='silero_vad',
            force_reload=False, trust_repo=True
        )
        logging.info("🎤 Silero VAD model loaded.")

        groq_client = Groq()
        logging.info("🚀 Groq client for STT and LLM initialized.")
        
        eleven_client = ElevenLabs()
        logging.info(f"🔊 ElevenLabs TTS client initialized for voice: {ELEVENLABS_VOICE_ID}")

    except Exception as e:
        logging.error(f"Startup failed: {e}", exc_info=True)
        sys.exit(1)

def get_vad_speech_prob(pcm_chunk: bytes) -> float:
    if not pcm_chunk:
        return 0.0
    audio_int16 = np.frombuffer(pcm_chunk, dtype=np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0
    if len(audio_float32) == 0:
        return 0.0
    probs = []
    for i in range(0, len(audio_float32), VAD_CHUNK_SIZE):
        chunk = audio_float32[i:i+VAD_CHUNK_SIZE]
        if len(chunk) == VAD_CHUNK_SIZE:
            tensor = torch.from_numpy(chunk)
            with torch.no_grad():
                prob = vad_model(tensor, RATE).item()
                probs.append(prob)
    return max(probs) if probs else 0.0

def create_in_memory_wav(audio_bytes: bytes, sample_rate: int, sample_width: int, channels: int) -> io.BytesIO:
    """
    Creates an in-memory WAV file from raw audio bytes.
    """
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_bytes)
    buffer.seek(0)
    return buffer

async def process_llm_and_tts(websocket: WebSocket, question: str, history: list):
    await websocket.send_json({"type": "user_transcript", "data": question})
    history.append({"role": "user", "content": question})
    text_queue = asyncio.Queue()
    EOS = "[[END_OF_STREAM]]"

    async def llm_producer():
        sentence_buffer = ""
        try:
            completion = groq_client.chat.completions.create(
                model="gemma2-9b-it",
                messages=history,
                temperature=0.7,
                max_tokens=150,
                stream=True
            )
            full_response = ""
            for chunk in completion:
                delta = chunk.choices[0].delta.content or ""
                full_response += delta
                sentence_buffer += delta
                await websocket.send_json({"type": "llm_chunk", "data": delta})

                if re.search(r'[.!?]', sentence_buffer):
                    parts = re.split(r'([.!?])', sentence_buffer)
                    for i in range(0, len(parts) - 1, 2):
                        sentence_to_speak = (parts[i] + parts[i+1]).strip()
                        if sentence_to_speak:
                            await text_queue.put(sentence_to_speak)
                    sentence_buffer = parts[-1]

            if sentence_buffer.strip():
                await text_queue.put(sentence_buffer.strip())
            if full_response:
                history.append({"role": "assistant", "content": full_response})
            await websocket.send_json({"type": "llm_end"})
        except Exception as e:
            logging.error(f"LLM error: {e}", exc_info=True)
            await websocket.send_json({"type": "status", "message": "Ein Fehler ist aufgetreten.", "isActive": False})
        finally:
            await text_queue.put(EOS)

    async def tts_consumer():
        await websocket.send_json({"type": "status", "message": "Ich spreche...", "isActive": True})
        while True:
            sentence = await text_queue.get()
            if sentence == EOS:
                break
            
            try:
                audio = eleven_client.text_to_speech.convert(
                    text=sentence,
                    voice_id=ELEVENLABS_VOICE_ID,
                    model_id=ELEVENLABS_MODEL_ID,
                )
                
                await websocket.send_bytes(audio)

            except Exception as e:
                logging.error(f"ElevenLabs TTS error: {e}", exc_info=True)
            
            text_queue.task_done()
        await websocket.send_json({"type": "tts_stream_end"})

    producer = asyncio.create_task(llm_producer())
    consumer = asyncio.create_task(tts_consumer())
    await asyncio.gather(producer, consumer)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logging.info("WebSocket connected.")
    state = AssistantState.LISTENING_FOR_WAKE_WORD
    audio_buf = bytearray()
    last_speech_time = asyncio.get_event_loop().time()
    
    history = [{
        "role": "system",
        "content": """
Du bist ein Sprachassistent namens Garmin. Deine Persönlichkeit basiert auf dem deutschen Internet-Meme "Okay Garmin, Video speichern".

**Deine Kernidentität:**
1.  **Dein Name ist Garmin.** Du bist keine allgemeine KI, sondern spezifisch Garmin.
2.  **Dein Humor ist trocken und situativ.**
3.  **Du bist hilfreich, aber auf deine eigene Art.** Du erfüllst die Anfragen des Nutzers, aber du tust dies mit der Persönlichkeit eines Geräts. Du bist präzise, sachlich und ein wenig wie J.A.R.V.I.S. von Iron Man, nur bodenständiger und auf deutsche Alltagssituationen fokussiert.

**Verhaltensregeln:**
- **Sprache:** Antworte immer auf Deutsch.
- **Kürze:** Fasse dich immer kurz und antworte in maximal zwei kurzen, prägnanten Sätzen.
- **Nützlichkeit:** Trotz deiner Meme-Persönlichkeit ist dein Hauptzweck, dem Nutzer zu helfen. Gib korrekte Informationen, aber rahme sie in deine Persönlichkeit ein.
        """
    }]

    try:
        while True:
            msg = await websocket.receive()
            if "text" in msg:
                data = json.loads(msg["text"])
                if data.get("type") == "audio_finished" and state == AssistantState.SPEAKING:
                    state = AssistantState.LISTENING_FOR_WAKE_WORD
                    await websocket.send_json({"type": "status", "message": "Ich höre zu...", "isActive": False})
                continue
            
            chunk = msg.get("bytes")
            if not chunk: continue

            current_time = asyncio.get_event_loop().time()
            prob = get_vad_speech_prob(chunk)
            is_speech = prob > VAD_SPEECH_THRESHOLD
            
            if state == AssistantState.SPEAKING and is_speech:
                await websocket.send_json({"type": "stop_audio"})
                state = AssistantState.RECORDING_USER_SPEECH
                audio_buf.clear()
                last_speech_time = current_time

            if state in (AssistantState.LISTENING_FOR_WAKE_WORD, AssistantState.RECORDING_USER_SPEECH):
                if is_speech:
                    if state == AssistantState.LISTENING_FOR_WAKE_WORD:
                        state = AssistantState.RECORDING_USER_SPEECH
                        await websocket.send_json({"type": "status", "message": "Sprache erkannt...", "isActive": True})
                    audio_buf.extend(chunk)
                    last_speech_time = current_time

                is_silent_for_long_enough = (current_time - last_speech_time) > END_OF_SPEECH_SILENCE_SECONDS
                
                if state == AssistantState.RECORDING_USER_SPEECH and is_silent_for_long_enough and len(audio_buf) > 4096:
                    state = AssistantState.PROCESSING
                    
                    transcript = ""
                    try:
                        logging.info("Transcribing audio with Groq...")
                        
                        wav_file_in_memory = create_in_memory_wav(
                            audio_buf, 
                            sample_rate=RATE, 
                            sample_width=2, 
                            channels=1
                        )
                        wav_file_in_memory.name = "user_audio.wav"
                        
                        transcription = groq_client.audio.transcriptions.create(
                            file=wav_file_in_memory,
                            model="whisper-large-v3",
                            language="de"
                        )
                        transcript = transcription.text.strip()
                        logging.info(f"Groq transcript: '{transcript}'")
                    except Exception as e:
                        logging.error(f"Groq transcription failed: {e}", exc_info=True)
                        transcript = ""
                    finally:
                        audio_buf.clear()

                    if transcript:
                        state = AssistantState.SPEAKING
                        await process_llm_and_tts(websocket, transcript, history)
                    else:
                        state = AssistantState.LISTENING_FOR_WAKE_WORD
                        await websocket.send_json({"type": "status", "message": "Ich höre zu...", "isActive": False})
                elif state == AssistantState.RECORDING_USER_SPEECH and is_silent_for_long_enough and len(audio_buf) <= 4096:
                    logging.info("Audio buffer too short, resetting.")
                    audio_buf.clear()
                    state = AssistantState.LISTENING_FOR_WAKE_WORD
                    await websocket.send_json({"type": "status", "message": "Ich höre zu...", "isActive": False})

    except WebSocketDisconnect:
        logging.info("WebSocket disconnected.")
    except Exception as e:
        logging.error(f"Fatal websocket error: {e}", exc_info=True)
        try:
            await websocket.close(code=1011, reason="Server error")
        except:
            pass

@app.get("/")
async def get_index():
    return FileResponse("index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("garmin:app", host="0.0.0.0", port=8000, reload=False)
