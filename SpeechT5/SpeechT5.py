from transformers import VitsModel, AutoTokenizer
import torch
import sounddevice as sd
import numpy as np
import asyncio
import re

# Function to split text into chunks
def split_text(text, max_words=30):
    parts = re.split(r'(?<=[.,!?])\s+', text)
    result = []
    for part in parts:
        words = part.split()
        for i in range(0, len(words), max_words):
            result.append(" ".join(words[i:i + max_words]))
    return result

# Generate audio and add to queue
async def generate_audio(model, tokenizer, text_parts, queue):
    for part in text_parts:
        if not part.strip():
            continue
            
        print(f"Generating audio: '{part[:20]}...'")
        inputs = tokenizer(part, return_tensors="pt")
        inputs['input_ids'] = inputs['input_ids'].long()

        try:
            with torch.no_grad():
                output = model(**inputs).waveform
        except Exception as e:
            print(f"Error generating audio '{part[:20]}...':", e)
            continue

        waveform = output.cpu().numpy().squeeze()
        
        # Normalize to [-1, 1]
        max_val = np.max(np.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val

        # Convert to float32 for sounddevice compatibility
        waveform = waveform.astype(np.float32)

        # Put into queue. If queue is full, this will wait (backpressure logic),
        # but playback starts immediately as soon as the first item is available.
        await queue.put(waveform)

    await queue.put(None)  # Signal completion

# Async audio playback using a continuous stream
async def play_audio(sample_rate, queue):
    print("▶ Starting continuous playback...")
    
    # Open a continuous output stream.
    # We use a context manager to ensure it closes properly.
    with sd.OutputStream(samplerate=sample_rate, channels=1, dtype='float32') as stream:
        while True:
            # Get data from queue immediately when available
            waveform = await queue.get()
            
            if waveform is None:
                break
            
            # Write data to the open stream. 
            # stream.write is blocking, so we run it in a thread to keep the event loop alive.
            # It will automatically wait if the audio buffer is full, ensuring seamless joining.
            await asyncio.to_thread(stream.write, waveform)
            
    print("⏹ Playback finished.")

async def main():
    print("Loading model...")
    model = VitsModel.from_pretrained("facebook/mms-tts-ukr")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-ukr")
    
    # Increased maxsize slightly to create a better buffer against generation lags
    queue = asyncio.Queue(maxsize=5)
    
    text = """
    Наша історія
    Все почалося в дві тисячі сьомому році, за піцою та планом.

    Одного вечора в Роттердамі, Нідерланди, п’ять людей зібралися на піцу. 
    Вони вже успішно допомагали компаніям БІ ТУ СІ процвітати з електронною комерцією. 
    Але вони побачили, що жодна платформа електронної комерції на ринку справді не задовольняє потреби компаній БІ ТУ БІ.
    Ніхто не дозволяв компаніям БІ ТУ БІ забезпечити той самий досвід в Інтернеті, що зробило їх успішними в режимі офлайн.

    Отже, піца в руці, вони прийняли рішення вирішити цей дисбаланс і вирішили створити платформу 
    електронної комерції БІ ТУ БІ на відміну від будь-якої іншої.
    Вони почали на початку: дивлячись на те, на які системні організації БІ ТУ БІ вже покладаються, 
    щоб вести свій бізнес та успішно обслуговувати своїх клієнтів. У серці вони знайшли систему Є еР Пе. 
    І вони подумали, що робити? Що робити, а замість того, щоб починати з веб - магазину,
    Ми почали з існуючої системи Є еР Пе? Що робити, якщо ми змусили Є еР Пе та електронну комерцію працювати як один?

    І народилася Сана Коммерсе
    """
    text_parts = split_text(text)
    sample_rate = model.config.sampling_rate

    # Run generation and playback concurrently
    await asyncio.gather(
        generate_audio(model, tokenizer, text_parts, queue),
        play_audio(sample_rate, queue)
    )

if __name__ == "__main__":
    asyncio.run(main())