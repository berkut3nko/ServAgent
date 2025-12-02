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
    AutoTokenizer, 
    AutoModelForQuestionAnswering, 
    pipeline
)

""" --- Налаштування конфігурації --- """
STT_MODEL_NAME = 'Yehor/w2v-bert-uk-v2.1'
TTS_MODEL_NAME = "facebook/mms-tts-ukr"
QA_MODEL_NAME = "timpal0l/mdeberta-v3-base-squad2"

""" Вибір пристрою: GPU (cuda), якщо доступний, інакше CPU """
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
""" Вибір типу даних: float16 для GPU для економії пам'яті, float32 для CPU """
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
SAMPLING_RATE_STT = 16_000

""" --- Допоміжні функції --- """

def split_text(text, max_words=30):
    """
    Розділяє текст на менші частини для синтезу мови, 
    щоб уникнути перевантаження моделі довгими реченнями.
    """
    parts = re.split(r'(?<=[.,!?])\s+', text)
    result = []
    for part in parts:
        words = part.split()
        for i in range(0, len(words), max_words):
            result.append(" ".join(words[i:i + max_words]))
    return result

def record_audio(duration=7, samplerate=SAMPLING_RATE_STT):
    """
    Записує аудіо з мікрофону заданої тривалості.
    Зберігає запис у файл 'recorded_audio.wav' для відлагодження.
    Повертає аудіо як масив numpy.
    """
    print("🎤 Запис... Будь ласка, говоріть у мікрофон.")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    print("✅ Запис завершено.")
    sf.write("recorded_audio.wav", audio, samplerate)
    return np.squeeze(audio)

""" --- Асинхронні функції синтезу мови (Оптимізовані) --- """

async def generate_audio(model, tokenizer, text_parts, queue):
    """
    Генерує аудіо з тексту частинами та додає їх у чергу.
    Це дозволяє почати відтворення ще до завершення генерації всього тексту.
    """
    for part in text_parts:
        if not part.strip():
            continue
            
        print(f"🔄 Генерація мови: '{part[:30]}...'")
        inputs = tokenizer(part, return_tensors="pt")
        inputs['input_ids'] = inputs['input_ids'].long()

        try:
            with torch.no_grad():
                output = model(**inputs).waveform
        except Exception as e:
            print(f"❌ Помилка генерації мови '{part[:20]}...':", e)
            continue

        waveform = output.cpu().numpy().squeeze()
        
        """ Нормалізація гучності, щоб уникнути спотворень """
        max_val = np.max(np.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val
        waveform = waveform.astype(np.float32)

        """ Додаємо згенерований шматок у чергу для відтворення """
        await queue.put(waveform)

    """ Сигнал про завершення генерації """
    await queue.put(None)

async def play_audio(sample_rate, queue):
    """
    Безперервно відтворює аудіо з черги.
    Використовує OutputStream для плавного переходу між шматками аудіо.
    """
    print("▶ Початок відтворення...")
    with sd.OutputStream(samplerate=sample_rate, channels=1, dtype='float32') as stream:
        while True:
            waveform = await queue.get()
            if waveform is None:
                break
            """ Запис у потік виконується в окремому потоці, щоб не блокувати цикл подій """
            await asyncio.to_thread(stream.write, waveform)
    print("⏹ Відтворення завершено.")

async def speak_text(text, model, tokenizer):
    """
    Керує процесом озвучення: ініціалізує чергу та запускає 
    генерацію і відтворення паралельно.
    """
    queue = asyncio.Queue(maxsize=5)
    text_parts = split_text(text)
    sample_rate = model.config.sampling_rate
    await asyncio.gather(
        generate_audio(model, tokenizer, text_parts, queue),
        play_audio(sample_rate, queue)
    )

""" --- Головна логіка --- """

async def main():
    print(f"📥 Завантаження моделі STT ({STT_MODEL_NAME})...")
    asr_model = AutoModelForCTC.from_pretrained(STT_MODEL_NAME).to(DEVICE)
    if torch.cuda.is_available():
        asr_model = asr_model.to(DTYPE)
    processor = Wav2Vec2BertProcessor.from_pretrained(STT_MODEL_NAME)

    print(f"📥 Завантаження моделі QA ({QA_MODEL_NAME})...")
    qa_tokenizer = AutoTokenizer.from_pretrained(QA_MODEL_NAME)
    qa_model_raw = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL_NAME)
    qa_pipeline = pipeline("question-answering", model=qa_model_raw, tokenizer=qa_tokenizer, device=-1)

    print(f"📥 Завантаження моделі TTS ({TTS_MODEL_NAME})...")
    tts_model = VitsModel.from_pretrained(TTS_MODEL_NAME)
    tts_tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL_NAME)

    """ --- Цикл виконання --- """
    
    """ Запис аудіо тривалістю 8 секунд """
    audio_input = await asyncio.to_thread(record_audio, duration=8)
    
    print("⚙️  Обробка аудіо...")
    inputs = processor([audio_input], sampling_rate=SAMPLING_RATE_STT, return_tensors="pt")
    features = inputs.input_features.to(DEVICE)
    if torch.cuda.is_available():
        features = features.to(DTYPE)

    with torch.no_grad():
        logits = asr_model(features).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcribed_text = processor.batch_decode(predicted_ids)[0]
    
    print(f"\n📝 Транскрипція: {transcribed_text}\n")

    """ Розбиття тексту на речення/запитання за допомогою регулярних виразів """
    sentences = re.split(r"(?=\b(?:чим|чи|чому|яким|яку|які|як|скільки|чого|що|куди|коли|хто|де|чому)\b)", transcribed_text)
    
    """ Очищення тексту, додавання великої літери та знаку питання, видалення дублікатів """
    detected_questions = []
    seen = set()
    
    for sent in sentences:
        clean_sent = sent.strip()
        """ Фільтруємо занадто короткі фрагменти """
        if len(clean_sent) > 3: 
            question_str = clean_sent.capitalize()
            if not question_str.endswith('?'):
                question_str += "?"
            
            if question_str not in seen:
                detected_questions.append(question_str)
                seen.add(question_str)
    
    if detected_questions:
        print(f"❓ Виявлені запитання з мови: {detected_questions}")
        """ Цей рядок просто посилається на змінну, можливо, мав бути присвоєнням або логуванням """
        detected_questions

    """ Контекст (база знань), в якому шукатиметься відповідь """
    context = """
    Мінімальний вклад для відкриття депозиту становить тисячу гривень.
    Умови депозиту включають ставку десять процентів річних, мінімальна сума тисячу гривень та термін від трьох до дванадцяти місяців.
    Процентна ставка за вкладом складає десять процентів річних.
    При достроковому знятті коштів клієнт втрачає всі нараховані відсотки.
    Доступні строки вкладу: три, шість або дванадцять місяців.
    """

    for question in detected_questions:
        print(f"\n🔍 Обробка запитання: {question}")
        
        result = await asyncio.to_thread(qa_pipeline, question=question, context=context)
        answer_text = result['answer']
        
        """ Логіка "Розумного Розширення" (Smart Expansion) """
        if result['end'] < len(context):
            """ Перевірка, чи відповідь закінчується коректним розділовим знаком """
            if not answer_text.strip().endswith(('.', '!', '?')):
                next_char_idx = result['end']
                
                """ Скануємо контекст вперед до кінця речення """
                while next_char_idx < len(context) and context[next_char_idx] not in ['.', '\n', '!', '?']:
                    next_char_idx += 1
                
                """ Якщо знайдено продовження, оновлюємо відповідь """
                if next_char_idx > result['end']:
                    extended_answer = context[result['start']:next_char_idx+1]
                    print(f"   (Розширена відповідь з: '{answer_text}' -> '{extended_answer}')")
                    answer_text = extended_answer

        """ Видалення двокрапки на початку, якщо вона була захоплена """
        if answer_text.strip().startswith(':'):
            answer_text = answer_text.strip()[1:].strip()

        if len(answer_text) < 5: 
             print(f"⚠️ Відповідь занадто коротка ('{answer_text}'), використовується загальна фраза.")
             answer_text = "Вибачте, я не знайшов детальної інформації у контексті."

        print(f"💡 Відповідь: {answer_text}")
        
        await speak_text(answer_text, tts_model, tts_tokenizer)
        
    """ Список тестових запитань (не використовується в основному циклі вище) """
    questions = [
        "Які умови?",
        "Які строки вкладу?",
        "Чи є мінімальна сума вкладу?",
        "Що буде при достроковому знятті коштів?"
    ] # Це мої питання для тестування

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nЗупинено користувачем.")