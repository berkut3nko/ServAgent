# -*- coding: utf-8 -*-
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering

# Завантаження моделі та токенізатора
model_name = "robinhad/ukrainian-qa"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Ініціалізація пайплайну для Відповідей на питання (QA)
qa_model = pipeline("question-answering", model=model.to("cpu"), tokenizer=tokenizer)

context = """
Умови депозиту включають ставку десять процентів річних, мінімальна сума тисячу гривень та термін від трьох до дванадцяти місяців.
Процентна ставка за вкладом складає десять процентів річних.
Мінімальна вклад для відкриття депозиту становить тисячу гривень.
При достроковому знятті коштів клієнт втрачає всі нараховані відсотки.
Доступні строки вкладу: три, шість або дванадцять місяців.
"""

# Приклади запитань до моделі
questions = [
    "Яка мінімальна сума депозиту?",
    "Яка відсоткова ставка?",
    "На який термін можна оформити депозит?",
    "Що станеться, якщо я достроково зніму гроші?"
]

# Отримання відповідей
for question in questions:
    result = qa_model(question=question, context=context)
    print(f"Запитання: {question}")
    print(f"Відповідь: {result['answer']}\n")
