import os
import io
import json
import base64
import logging

from flask import Flask, Response, request, send_from_directory, stream_with_context

import torch
from groq import Groq
from elevenlabs.client import ElevenLabs

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

RATE = 16000
ELEVENLABS_VOICE_ID = "N2lVS1w4EtoT3dr4eOWO"
ELEVENLABS_MODEL_ID = "eleven_flash_v2_5" 

vad_model = None
utils = None
groq_client = None
eleven_client = None

def load_models():
    global vad_model, utils, groq_client, eleven_client
    if all([vad_model, utils, groq_client, eleven_client]):
        logging.info("Models already loaded.")
        return

    try:
        logging.info("Loading models...")
        vad_model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, trust_repo=True
        )
        groq_client = Groq()
        eleven_client = ElevenLabs()
        logging.info("Models loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load models: {e}", exc_info=True)
        vad_model = groq_client = eleven_client = None

with app.app_context():
    load_models()

@app.route('/')
def serve_index():
    """Serves the main HTML page."""
    return send_from_directory(os.getcwd(), 'index.html')

@app.route('/assistant', methods=['POST'])
def assistant_handler():
    """Handles audio input, streams LLM text, then sends complete TTS audio."""
    if not all([vad_model, utils, groq_client, eleven_client]):
        logging.error("Attempted to process request before models were loaded.")
        return "Models not loaded, server is initializing. Please try again in a moment.", 503

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

**Wie antwortest du:**
User: Okay Garmin, wie alt bist du?
Garmin: Ich bin so alt wie das Internet-Meme, das mich berühmt gemacht hat. Aber keine Sorge, ich bin immer noch frisch und bereit zu helfen!
        """
    }]

    @stream_with_context
    def generate_stream():
        try:
            audio_data = request.data
            try:
                import torchaudio
                from torchaudio.transforms import Resample
                
                audio_file_for_vad = io.BytesIO(audio_data)
                waveform, sample_rate = torchaudio.load(audio_file_for_vad, format="webm")

                if sample_rate != RATE:
                    resampler = Resample(orig_freq=sample_rate, new_freq=RATE)
                    waveform = resampler(waveform)

                if waveform.shape[0] > 1: # Convert stereo to mono
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                get_speech_timestamps = utils[0]
                speech_timestamps = get_speech_timestamps(waveform, vad_model, sampling_rate=RATE)
                
                if not speech_timestamps:
                    logging.info("VAD: No speech detected.")
                    yield json.dumps({"type": "status", "data": {"message": "Nichts gehört. Ich höre zu...", "isActive": False}}) + "\n"
                    yield json.dumps({"type": "end_stream"}) + "\n"
                    return
            except Exception as e:
                logging.warning(f"Could not perform VAD pre-check: {e}. Proceeding with transcription anyway.", exc_info=True)
            
            yield json.dumps({"type": "status", "data": {"message": "Transkription...", "isActive": True}}) + "\n"
            
            audio_file = io.BytesIO(audio_data)
            audio_file.name = "user_audio.webm"
            transcription = groq_client.audio.transcriptions.create(
                file=audio_file, model="whisper-large-v3-turbo", language="de"
            )
            transcript = transcription.text.strip()
            logging.info(f"Transcript: '{transcript}'")
            
            if not transcript:
                yield json.dumps({"type": "status", "data": {"message": "Nichts gehört. Ich höre zu...", "isActive": False}}) + "\n"
                yield json.dumps({"type": "end_stream"}) + "\n"
                return

            yield json.dumps({"type": "user_transcript", "data": transcript}) + "\n"
            history.append({"role": "user", "content": transcript})

        except Exception as e:
            logging.error(f"Transcription error: {e}", exc_info=True)
            yield json.dumps({"type": "error", "data": "Transcription failed."}) + "\n"
            yield json.dumps({"type": "end_stream"}) + "\n"
            return

        try:
            yield json.dumps({"type": "status", "data": {"message": "Ich denke nach...", "isActive": True}}) + "\n"
            llm_stream = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile", messages=history, temperature=0.7, max_tokens=150, stream=True
            )
            
            full_response = ""
            for chunk in llm_stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    full_response += delta
                    yield json.dumps({"type": "llm_chunk", "data": delta}) + "\n"
            
            yield json.dumps({"type": "llm_end"}) + "\n"

            clean_response = full_response.strip()
            if clean_response:
                yield json.dumps({"type": "status", "data": {"message": "Stimme wird generiert...", "isActive": True}}) + "\n"
                
                tts_stream = eleven_client.text_to_speech.convert(
                    text=clean_response, 
                    voice_id=ELEVENLABS_VOICE_ID, 
                    model_id=ELEVENLABS_MODEL_ID
                )
                
                full_audio_bytes = b"".join(tts_stream)
                
                encoded_audio = base64.b64encode(full_audio_bytes).decode('utf-8')
                
                yield json.dumps({"type": "tts_chunk", "data": encoded_audio}) + "\n"

            yield json.dumps({"type": "tts_end"}) + "\n"
            yield json.dumps({"type": "status", "data": {"message": "Ich höre zu...", "isActive": False}}) + "\n"

        except Exception as e:
            logging.error(f"LLM/TTS streaming error: {e}", exc_info=True)
            yield json.dumps({"type": "error", "data": "Response generation failed."}) + "\n"
        finally:
            yield json.dumps({"type": "end_stream"}) + "\n"

    return Response(generate_stream(), mimetype='application/jsonl')

if __name__ == '__main__':
    with app.app_context():
        load_models()
    app.run(debug=True, port=8000)
