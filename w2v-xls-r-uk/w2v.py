import torch
import sounddevice as sd
import soundfile as sf
import numpy as np
import asyncio
import re
from transformers import (
    AutoModelForCTC, 
    Wav2Vec2BertProcessor, 
    VitsModel, 
    pipeline
)

# --- Configuration ---
STT_MODEL_NAME = 'Yehor/w2v-bert-uk-v2.1'

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
SAMPLING_RATE_STT = 16_000

# --- Helper Functions ---

def split_text(text, max_words=30):
    parts = re.split(r'(?<=[.,!?])\s+', text)
    result = []
    for part in parts:
        words = part.split()
        for i in range(0, len(words), max_words):
            result.append(" ".join(words[i:i + max_words]))
    return result

def record_audio(duration=7, samplerate=SAMPLING_RATE_STT):
    print("🎤 Recording... Please speak into the microphone.")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    print("✅ Recording finished.")
    sf.write("recorded_audio.wav", audio, samplerate)
    return np.squeeze(audio)


# --- Main Logic ---

async def main():
    print(f"📥 Loading STT model ({STT_MODEL_NAME})...")
    asr_model = AutoModelForCTC.from_pretrained(STT_MODEL_NAME).to(DEVICE)
    if torch.cuda.is_available():
        asr_model = asr_model.to(DTYPE)
    processor = Wav2Vec2BertProcessor.from_pretrained(STT_MODEL_NAME)

    # --- Execution Loop ---
    
    audio_input = await asyncio.to_thread(record_audio, duration=8)
    
    print("⚙️  Processing audio...")
    inputs = processor([audio_input], sampling_rate=SAMPLING_RATE_STT, return_tensors="pt")
    features = inputs.input_features.to(DEVICE)
    if torch.cuda.is_available():
        features = features.to(DTYPE)

    with torch.no_grad():
        logits = asr_model(features).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcribed_text = processor.batch_decode(predicted_ids)[0]
    
    print(f"\n📝 Transcription: {transcribed_text}\n")

    sentences = re.split(r"(?=\b(?:чим|чи|чому|яким|яку|які|як|скільки|чого|що|куди|коли|хто|де|чому)\b)", transcribed_text)
    
    # Clean up, capitalize, add question mark, and remove duplicates
    detected_questions = []
    seen = set()
    
    for sent in sentences:
        clean_sent = sent.strip()
        if len(clean_sent) > 3: 
            question_str = clean_sent.capitalize()
            if not question_str.endswith('?'):
                question_str += "?"
            
            if question_str not in seen:
                detected_questions.append(question_str)
                seen.add(question_str)
    
    if detected_questions:
        print(f"❓ Detected questions from speech: {detected_questions}")
        detected_questions

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped by user.")