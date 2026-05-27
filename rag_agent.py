import os
import re
import pandas as pd
from typing import List
import hashlib
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions
import ollama

# .env
DOCUMENT_TXT_PATH = "document.txt"
QUESTIONS_CSV = "RAG_questions_table.csv"
ANSWERS_CSV = "RAG_answers.csv"
OUTPUT_CSV = "my_answers.csv"

OLLAMA_MODEL = "gemma4"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3

#Загрузка документа
def load_document() -> str:
    """Загружает документ из локального файла"""
    if not os.path.exists(DOCUMENT_TXT_PATH):
        raise FileNotFoundError(f"Файл документа {DOCUMENT_TXT_PATH} не найден. Положите его в папку со скриптом")
    with open(DOCUMENT_TXT_PATH, 'r', encoding='utf-8') as f:
        return f.read()

#Разбивка на чанки
def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Разбивает текст на перекрывающиеся чанки по словам"""
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk_words = words[i:i+chunk_size]
        if not chunk_words:
            continue
        if len(chunk_words) < 50 and chunks:
            chunks[-1] += " " + " ".join(chunk_words)
        else:
            chunks.append(" ".join(chunk_words))
    return chunks

#Векторное хранилище
def build_vectorstore(chunks: List[str], persist_dir: str = "./chroma_db"):
    """Создаёт векторное хранилище с эмбеддингами"""
    client = chromadb.PersistentClient(path=persist_dir)
    try:
        client.delete_collection("doc_chunks")
    except:
        pass
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.create_collection(
        name="doc_chunks",
        embedding_function=embedding_fn
    )
    ids = [hashlib.md5(chunk.encode()).hexdigest()[:12] for chunk in chunks]
    collection.add(documents=chunks, ids=ids)
    print(f"Векторное хранилище создано, {len(chunks)} чанков")
    return collection

#Поиск релевантных чанков
def retrieve_context(collection, query: str, top_k: int) -> str:
    """Возвращает топ-k релевантных чанков в виде одного текста"""
    results = collection.query(query_texts=[query], n_results=top_k)
    if results['documents']:
        return "\n---\n".join(results['documents'][0])
    return ""

#Чат с Ollama, и извлекаем только числа
def ask_llm(prompt: str) -> str:
    """Отправляет запрос в Ollama и возвращает ответ"""
    response = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])
    answer = response['message']['content'].strip()
    numbers = re.findall(r'-?\d+\.?\d*', answer)
    if numbers:
        return numbers[0]
    return "0"

#RAG-цикл
def rag_answer(question: str, collection) -> str:
    """Основной RAG-цикл: поиск контекста + генерация ответа"""
    context = retrieve_context(collection, question, TOP_K)
    if not context.strip():
        return "0"
    prompt = f"""Ты — помощник, который отвечает на вопросы, используя ТОЛЬКО контекст.
Твоя задача — дать краткий числовой ответ. Никаких пояснений, только число.
Если в контексте нет точного ответа или он неясен, ответь "0".
Никогда не пиши лишних слов.

Контекст:
{context}

Вопрос: {question}
Числовой ответ:"""
    return ask_llm(prompt)

#Обработка всех вопросов
def process_questions(collection, questions_df: pd.DataFrame) -> pd.DataFrame:
    """Генерирует ответы для всех вопросов из таблицы"""
    answers = []
    for q in tqdm(questions_df['Question'], desc="Ответы на вопросы"):
        ans = rag_answer(q, collection)
        answers.append(ans)
    questions_df['Answer'] = answers
    return questions_df

#Оценка качества
def evaluate(our_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> float:
    """Сравнивает ответы по row_id с эталоном"""
    our_df['row_id'] = our_df['row_id'].astype(str)
    ground_truth_df['row_id'] = ground_truth_df['row_id'].astype(str)
    
    merged = pd.merge(our_df, ground_truth_df, on='row_id', suffixes=('_our', '_true'))
    merged['Answer_our_num'] = pd.to_numeric(merged['Answer_our'], errors='coerce')
    merged['Answer_true_num'] = pd.to_numeric(merged['Answer_true'], errors='coerce')
    correct = (merged['Answer_our_num'] == merged['Answer_true_num']).sum()
    total = len(merged)
    accuracy = correct / total * 100
    print(f"\nСовпадение с правильными ответами: {correct}/{total} = {accuracy:.2f}%")
    return accuracy

# main есть main че еще сказать
def main():
    print("1. Загрузка документа из document.txt...")
    text = load_document()
    print(f"   Длина текста: {len(text)} символов")
    
    print("2. Разбивка на чанки...")
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"   Получено {len(chunks)} чанков")
    
    print("3. Построение векторного индекса...")
    collection = build_vectorstore(chunks)
    
    print("4. Загрузка вопросов из CSV...")
    if not os.path.exists(QUESTIONS_CSV):
        raise FileNotFoundError(f"Не найден файл {QUESTIONS_CSV}")
    questions_df = pd.read_csv(QUESTIONS_CSV)
    if 'row_id' not in questions_df.columns or 'Question' not in questions_df.columns:
        raise ValueError("В CSV вопросов должны быть колонки 'row_id' и 'Question'")
    
    print("5. Генерация ответов через RAG...")
    result_df = process_questions(collection, questions_df)
    result_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Ответы сохранены в {OUTPUT_CSV}")
    
    print("6. Сравнение с эталонными ответами...")
    if not os.path.exists(ANSWERS_CSV):
        print(f"Файл {ANSWERS_CSV} не найден, оценка невозможна.")
        return
    true_df = pd.read_csv(ANSWERS_CSV)
    evaluate(result_df, true_df)
    
    print("\nГотово! Проанализируйте несовпадения в my_answers.csv")

if __name__ == "__main__":
    main()
