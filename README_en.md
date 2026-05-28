## Project Description

RAG Agent uses a local language model (LLM) `gemma4` through Ollama to generate answers to questions. The project allows you to process questions and get responses based on custom documentation using Retrieval-Augmented Generation (RAG) technology.

## Installation and Setup

### 1. Creating a Python Virtual Environment

```bash
python -m venv venv
```

### 2. Activating the Virtual Environment

**On Windows:**
```bash
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
source venv/bin/activate
```

### 3. Installing Dependencies

```bash
pip install -r requirements.txt
```

### 4. Installing and Configuring Ollama

#### Downloading Ollama

Download Ollama from the official website: https://ollama.ai

#### Installing the gemma4 Model

After installing Ollama, execute the command:

```bash
ollama run gemma4
```
This will download and run the gemma4 model

## Preparing Data Files

Place the following files in the root folder of the project (in the same directory as the `rag_agent.py` script):

### 1. `document.txt`
- Download the Google Doc from the link provided in your assignment description

### 2. `RAG_questions_table.csv`
- CSV table with questions for processing
- Should contain a column with questions and id (for example, ``` row_id,Question,Answer ``` )

Example structure:
``` csv
row_id,Question,Answer
11,В каком году Deep Blue победил Каспарова?,100
12,What was AlexNet's error rate in 2012 ImageNet challenge (in percent)?,100
13,Сколько миллионов пользователей ChatGPT достиг за первые пять дней?,100
```

### 3. `RAG_answers.csv`
- CSV table with reference answers to the questions
- Should contain a column with answers for comparison and id (for example, ```row_id,Answer```)

Example structure:
```csv
row_id,Answer
11,1997
12,15.3
13,1
```

## Running the Project

### 1. Run the RAG Agent Script

```bash
python rag_agent.py
```

### 2. Waiting for Results

The script will process questions from `RAG_questions_table.csv` one by one. This may take some time depending on:
- The number of questions
- The size of the documentation
- Your computer's performance

### 3. Getting Results

After the script completes, the following files will be created:

#### `my_answers.csv`
- CSV file with the answers obtained from the LLM
- Contains a mapping between questions and generated answers

Example structure:
```csv
row_id,Question,Answer
11,В каком году Deep Blue победил Каспарова?,0
12,What was AlexNet's error rate in 2012 ImageNet challenge (in percent)?,15.3
13,Сколько миллионов пользователей ChatGPT достиг за первые пять дней?,0
```
*In this version of the script, the prompt specifies that the LLM should write 0 if it doesn't know the answer*
