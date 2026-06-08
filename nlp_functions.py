import matplotlib.pyplot as plt
from wordcloud import WordCloud

#WORD CLOUD
def show_wordcloud(text):
    if isinstance(text, list):
        text = " ".join(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    wordcloud = WordCloud(width=900, height=500, background_color="white").generate(text)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    return fig


# N-gram Analysis
import streamlit as st
from nltk.util import ngrams
from collections import Counter
import plotly.graph_objects as go

def plot_top_ngrams_bar_chart(tokens, gram_n=3, top_n=15):
    try:
        ngram = list(ngrams(tokens, gram_n))
        ngram_counts = Counter(ngram).most_common(top_n)

        if not ngram_counts:
            raise ValueError("No n-grams found in the given  token list")

        labels = []
        counts = []
        for biagram, count in ngram_counts:
            labels.append(" ".join(biagram))
            counts.append(count)

        # plotly bar chart
        fig = go.Figure(data=
        [go.Bar(
            x=labels,
            y=counts,
            text=counts,
            textposition="outside")])

        # update layout
        fig.update_layout(height=550,
                          title="Top 15 Trigrams",
                          xaxis_title="Labels",
                          yaxis_title="Frequency")

        st.plotly_chart(fig)
    except Exception as e:
        print(f"An Error Occured: {e}")


# CREATING CHUNKS
import spacy
def split_into_chunks_spacy(text,max_length=500):
  nlp = spacy.load("en_core_web_sm")
  doc=nlp(text)
  chunks=[]
  current_chunk=""
  for sent in doc.sents:
    sentence=sent.text.strip()
    if len(current_chunk)+len(sentence)<=500:
       current_chunk+=" "+sentence
    else:
      chunks.append(current_chunk.strip())
      current_chunk=sentence

  if current_chunk:
    chunks.append(current_chunk.strip())

  return chunks

# EMOTIONAL ANALYSIS
from transformers import pipeline
import pandas as pd
import plotly.express as px
model_name="j-hartmann/emotion-english-distilroberta-base"
emotion_classifier= pipeline("text-classification", model=model_name, tokenizer=model_name, top_k=5)

def detect_emotion(text):
  try: 
    
    results =emotion_classifier(text,truncation=True,max_length=512)
    emotion_data=[{"emotion": res["label"], "confidence": res["score"]} for res in results[0]]
    df=pd.DataFrame(emotion_data)

    max_emotion_row=df.loc[df["confidence"].idxmax()]

    fig=px.bar(df, x="emotion", y="confidence", color="emotion", title=f"Top Five Emotion")
    fig.update_layout(showlegend=False)

    return{
        "text":text,
        "detect_emotion":max_emotion_row["emotion"],
        "confidence":max_emotion_row["confidence"],
        "emotion_table":df ,
        "plot":fig

            }

  except Exception as e:
    return {"error": str(e)}


#SENTIMENT ANALYSIS
model_name="cardiffnlp/twitter-roberta-base-sentiment"
sentiment_classifier=pipeline("sentiment-analysis",model=model_name, tokenizer=model_name, top_k=3,truncation=True, max_length=512)

# creating function
def detect_overall_sentiment_avg(text):
  try:
    sentiment_labels={
        "LABEL_0":"Negative",
        "LABEL_1":"Neutral",
        "LABEL_2":"Positive"
    }
    chunks=split_into_chunks_spacy(text)
    score_total={"Negative":0,"Neutral":0,"Positive":0}
    chunk_count=len(chunks)

    if chunk_count == 0:
      return {"overall_sentiment": "Neutral", "average_scores": {"Negative": 0, "Neutral": 0, "Positive": 0}}

    for chunk in chunks:
      results=sentiment_classifier(chunk)[0]
      for result in results:
        label=sentiment_labels[result["label"]]
        score_total[label]+=result["score"]

    avg_score={}
    for label in score_total:
      avg_score[label]=score_total[label]/chunk_count

    overall_sentiment=max(avg_score, key=avg_score.get)
    return{
        "overall_sentiment":overall_sentiment,
        "average_scores": avg_score
  }

  except  Exception as e:
    return{f"error": str(e)}


#TONE OF SPEECH CLASSIFICATION
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
labels = [
    "factual",
    "opinion",
    "question",
    "command",
    "emotion",
    "personal experience",
    "suggestion",
    "story",
    "prediction",
    "warning",
    "instruction",
    "definition",
    "narrative",
    "news",
    "argument"
]

def classify_custom(text):
  result= classifier(text,candidate_labels=labels)
  return{
      "text":text,
      "predicted_category":result["labels"][0],
      "score":result["scores"][0],
      "all_categories":list(zip(result["labels"], result["scores"]))
  }

# TEXT SUMMARIZATION
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def summarize_large_text(text):
    model_name = "google-t5/t5-small"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Step 1: Split text into manageable chunks
    chunks = split_into_chunks_spacy(text, max_length=500) 

    chunk_summaries = []
    for chunk in chunks:
        input_length = len(chunk.split()) # rough word count 
        
        max_tokens = min(300, max(30, int(input_length * 0.7)))  
        min_tokens = min(100, max(20, int(input_length * 0.3)))  
        
        t5_input = "summarize: " + chunk
        inputs = tokenizer(t5_input, return_tensors="pt", max_length=512, truncation=True)
        
        summary_ids = model.generate(
            inputs["input_ids"], 
            max_new_tokens=max_tokens, 
            min_length=min_tokens, 
            num_beams=4,          
            length_penalty=2.0,   
            early_stopping=True
        )
        
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        chunk_summaries.append(summary)
    
    # Step 3: Combine all chunk summaries into one text
    combined_summary_text = " ".join(chunk_summaries)

    input_length = len(combined_summary_text.split())  
    max_tokens = int(input_length * 0.9) 
    min_tokens = int(input_length * 0.3)
    
    final_t5_input = "summarize: " + combined_summary_text
    final_inputs = tokenizer(final_t5_input, return_tensors="pt", max_length=512, truncation=True)
    
    final_summary_ids = model.generate(
        final_inputs["input_ids"], 
        max_new_tokens=max_tokens, 
        min_length=min_tokens, 
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True
    )
    
    return tokenizer.decode(final_summary_ids[0], skip_special_tokens=True)


