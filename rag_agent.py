import os
import re
import pandas as pd
from typing import List, Tuple
import hashlib
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions
import ollama

#.env 
DOCUMENT_TXT_PATH = "document.txt"
QUESTIONS_CSV = "RAG_questions_table.csv"
ANSWERS_CSV = "RAG_answers.csv"
OUTPUT_CSV = "my_answers.csv"

OLLAMA_MODEL = "gemma4"
CHUNK_SIZE = 250
CHUNK_OVERLAP = 80
TOP_K = 10
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

#Нормализация текста 
def normalize_text(text: str) -> str:
    replacements = {
        'один': '1', 'два': '2', 'три': '3', 'четыре': '4', 'пять': '5',
        'шесть': '6', 'семь': '7', 'восемь': '8', 'девять': '9', 'десять': '10',
        'сто': '100', 'тысяча': '1000', 'миллион': '1000000',
        'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
        'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
        'hundred': '100', 'thousand': '1000', 'million': '1000000'
    }
    for word, num in replacements.items():
        text = re.sub(rf'\b{word}\b', num, text, flags=re.IGNORECASE)
    return text

#Загрузка документа
def load_document() -> str:
    if not os.path.exists(DOCUMENT_TXT_PATH):
        raise FileNotFoundError(f"Файл {DOCUMENT_TXT_PATH} не найден")
    with open(DOCUMENT_TXT_PATH, 'r', encoding='utf-8') as f:
        text = f.read()
    text = re.sub(r'\s+', ' ', text)
    text = normalize_text(text)
    return text

#Разбивка на чанки
def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - overlap
    if len(chunks) > 1 and len(chunks[-1]) < 50:
        chunks[-2] += " " + chunks[-1]
        chunks.pop()
    return chunks

#Создаем векторное хранилище
def build_vectorstore(chunks: List[str], persist_dir: str = "./chroma_db"):
    client = chromadb.PersistentClient(path=persist_dir)
    try:
        client.delete_collection("doc_chunks")
    except:
        pass
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    collection = client.create_collection(
        name="doc_chunks",
        embedding_function=embedding_fn
    )
    ids = [hashlib.md5(chunk.encode()).hexdigest()[:12] for chunk in chunks]
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        collection.add(documents=batch_chunks, ids=batch_ids)
    print(f"Векторное хранилище создано, {len(chunks)} чанков")
    return collection

#Поиск релевантных чанков
def retrieve_context(collection, query: str, top_k: int) -> List[str]:
    results = collection.query(query_texts=[query], n_results=top_k)
    if results['documents']:
        return results['documents'][0]
    return []

#RAG-функция
def ask_llm_with_context(question: str, contexts: List[str]) -> str:
    if not contexts:
        return "0"
    
    context_block = "\n---\n".join([f"[{i+1}] {c}" for i, c in enumerate(contexts)])
    
    system = """Ты — аналитик, извлекающий числовые ответы из контекста.
Правила:
- Найди в контексте число, которое отвечает на вопрос.
- Если точное число найдено, выведи ТОЛЬКО это число.
- Если чисел несколько, выбери то, которое соответствует вопросу.
- Если числа нет, выведи 0.
- Никаких пояснений, только число или 0.

Примеры:
Контекст: [1] В 2022 году рынок ИИ составил 136.6 млрд.
Вопрос: Какой объём рынка ИИ в 2022 году?
Ответ: 136.6

Контекст: [1] Средняя продолжительность жизни лабуба 100 лет.
Вопрос: Сколько лет живут лабуба?
Ответ: 100

Контекст: [1] Дроны летают 30 минут. [2] Скорость дрона 50 км/ч.
Вопрос: Максимальная скорость дрона?
Ответ: 50
"""
    user_prompt = f"""Контекст:
{context_block}

Вопрос: {question}
Числовой ответ:"""
    
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt}
        ],
        options={"temperature": 0.0}
    )
    answer = response['message']['content'].strip()
    
    match = re.search(r'-?\d+(?:\.\d+)?', answer)
    if match:
        return match.group(0)
    
    word_to_num = {
        'ноль':0, 'один':1, 'два':2, 'три':3, 'четыре':4, 'пять':5,
        'шесть':6, 'семь':7, 'восемь':8, 'девять':9, 'десять':10,
        'сто':100, 'тысяча':1000, 'миллион':1000000, 'миллиард':1000000000,
        'zero':0, 'one':1, 'two':2, 'three':3, 'four':4, 'five':5,
        'six':6, 'seven':7, 'eight':8, 'nine':9, 'ten':10
    }
    for word, num in word_to_num.items():
        if word in answer.lower():
            return str(num)
    return "0"

def rag_answer(question: str, collection) -> str:
    contexts = retrieve_context(collection, question, TOP_K)
    if not contexts:
        return "0"
    return ask_llm_with_context(question, contexts)

#Обработка всех вопросов
def process_questions(collection, questions_df: pd.DataFrame) -> pd.DataFrame:
    answers = []
    for q in tqdm(questions_df['Question'], desc="Ответы на вопросы"):
        ans = rag_answer(q, collection)
        answers.append(ans)
    questions_df['Answer'] = answers
    return questions_df

#Оценка качества (а то с ума можно сойти, вручную все проверять:D)
def evaluate(pred_path: str, true_path: str) -> float:
    pred_df = pd.read_csv(pred_path)
    true_df = pd.read_csv(true_path)
    pred_df['row_id'] = pred_df['row_id'].astype(str)
    true_df['row_id'] = true_df['row_id'].astype(str)
    merged = pd.merge(pred_df, true_df, on='row_id', suffixes=('_pred', '_true'))
    merged['pred_num'] = pd.to_numeric(merged['Answer_pred'], errors='coerce')
    merged['true_num'] = pd.to_numeric(merged['Answer_true'], errors='coerce')
    correct = (merged['pred_num'] == merged['true_num']).sum()
    total = len(merged)
    acc = correct / total * 100
    print(f"\nСовпадение: {correct}/{total} = {acc:.2f}%")
    errors = merged[merged['pred_num'] != merged['true_num']]
    if not errors.empty:
        print("\nОшибки (row_id | ответ LLM | правильный):")
        for _, row in errors.iterrows():
            print(f"   {row['row_id']} | {row['Answer_pred']} | {row['Answer_true']}")
    return acc

#main есть main че еще сказать
def main():
    print("1. Загрузка и нормализация документа...")
    text = load_document()
    print(f"   Длина текста: {len(text)} символов")
    
    print("2. Разбивка на чанки...")
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"   Получено {len(chunks)} чанков")
    
    print("3. Построение векторного индекса (ChromaDB)...")
    collection = build_vectorstore(chunks)
    
    print("4. Загрузка вопросов...")
    questions_df = pd.read_csv(QUESTIONS_CSV)
    
    print("5. Генерация ответов через RAG...")
    result_df = process_questions(collection, questions_df)
    result_df.to_csv(OUTPUT_CSV, index=False)
    print(f"   Ответы сохранены в {OUTPUT_CSV}")
    
    print("6. Оценка качества...")
    if os.path.exists(ANSWERS_CSV):
        acc = evaluate(OUTPUT_CSV, ANSWERS_CSV)
        if acc >= 96.4:
            print("Цель достигнута")
        else:
            print(f"Точность {acc:.2f}% ниже 96.4%")
    else:
        print(f"Файл {ANSWERS_CSV} не найден")

if __name__ == "__main__":
    main()
