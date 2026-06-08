# 📊 Interactive Text Analysis Platform

An interactive, NLP-powered web application built with Streamlit that processes text data to deliver deep linguistic insights. The application cleans raw text using advanced tokenization, visualizes word patterns, detects emotions, analyzes sentiment, determines speech tone, evaluates N-gram frequencies, and generates concise summaries using optimized Hugging Face transformer models.

---

## 🚀 Features

* **Flexible Data Inputs**: 
  * **Direct Input**: Copy and paste raw text directly into the platform workspace.
  * **Bulk Upload**: Upload `.csv` data files and apply custom filtering options directly within the UI.
    
* **Text Preprocessing**: Automated cleaning, regex pattern matching, stop-word removal, and text lemmatization using `re` and `spaCy`.
* **Word Cloud Generation**: Dynamic visualization of the most frequent words.
* **N-gram Analysis**: Extracts and displays frequent sequences of words (Unigrams, Bigrams, and Trigrams) to discover context and word combinations.
* **Emotion Detection**: Identifies fine-grained emotions (e.g., joy, anger, sadness, fear).
* **Sentiment Analysis**: Classifies text into positive, negative, or neutral categories.
* **Tone of Speech**: Evaluates structural tone using zero-shot classification techniques.
* **Summary Generation**: Compresses long-form text into concise, coherent summaries.

---

## 🤖 Machine Learning Models Used

This platform leverages specialized, pre-trained transformer pipelines from the Hugging Face Hub:


| NLP Task | Model Name | Description |
| :--- | :--- | :--- |
| **Emotion Detection** | `j-hartmann/emotion-english-distilroberta-base` | Predicts 6 basic emotions plus neutral. |
| **Sentiment Analysis** | `cardiffnlp/twitter-roberta-base-sentiment` | RoBERTa model tuned for positive/neutral/negative sentiment. |
| **Tone of Speech** | `facebook/bart-large-mnli` | Zero-shot classification model used to detect specific speech tones. |
| **Summary Generation** | `google-t5/t5-small` | Lightweight text-to-text transformer for fast text summarization. |

---

### 📁 Project Structure

*   **`app.py`**: The main Streamlit dashboard file managing the user interface, layout, N-gram charts, and file upload filters.
*   **`text_cleaner.py`**: Contains modular utility functions dedicated to regex text cleaning, tokenization, and lemmatization using `re` and `spacy`.
*   **`nlp_functions.py`**: Houses the pipeline initializations, inference logic, and API calls to the Hugging Face models.


---

## 🛠️ Installation & Setup

Follow these steps to get the platform running locally on your machine.

### 1. Clone the Repository
```bash
git clone https://github.com
cd interactive-text-analysis
```

### 2. Set Up a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install Dependencies & Language Models
Install the required packages and download the mandatory spaCy English pipeline:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

> ⚠️ **Important:** Make sure you download the spaCy English model using the command above before launching the application, otherwise it will throw an initialization error.

---

## 💻 Usage

To launch the interactive dashboard, execute the following command in your terminal:

```bash
streamlit run app.py
```

The app will compile and automatically launch in your default web browser at `http://localhost:8501`.

---



