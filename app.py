import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import pandas as pd
from text_cleaner import clean_text, clean_text_spacy
from nlp_functions import show_wordcloud,plot_top_ngrams_bar_chart,detect_emotion,split_into_chunks_spacy,detect_overall_sentiment_avg,classify_custom,summarize_large_text


st.set_page_config(layout="wide")
st.title("INTERACTIVE TEXT ANALYSIS PLATEFORM")
st.divider()


a = st.sidebar.radio("SELECT ONE:", ["Process Textual Data", "Process Csv File"])

if a == "Process Textual Data":
    st.header(" Input Your Textual Data")
    text = st.text_area("Enter Your Text", height=150)

    if st.button("Analyze"):
        if not text.strip():
            st.warning("Please enter your text")
        else:
            #clean and processing
            cleaned= clean_text(text)
            tokens= clean_text_spacy(cleaned)
            st.subheader("Cleaned and Lemmatized Text")
            st.write(" ".join(tokens))
            st.divider()

             # WORD CLOUD
            if tokens:
                st.subheader("WORD CLOUD")
                wc_plot = show_wordcloud(tokens)
                st.pyplot(wc_plot)
            st.divider()


            # N-GRAM ANALYSIS.
            st.subheader("N-GRAM ANALYSIS")
            plot_top_ngrams_bar_chart(tokens, gram_n=3)
            st.divider()


            #EMOTION DETECTION
            st.subheader("EMOTION DETECTION.")
            result_dict = detect_emotion(text)

            if "error" in result_dict:
              st.error(f"Error in emotion detection: {result_dict['error']}")
            else:
              # Extract the pre-calculated values from the dictionary
              Emotion = result_dict["detect_emotion"]
              Score = result_dict["confidence"]
              top_emotions_df = result_dict["emotion_table"]
              fig = result_dict["plot"]

    
            st.write(f"PREDICTED EMOTION :- {Emotion}, with {Score * 100:.2f}% confidence")
    
           # Layout columns
            col1, col2 = st.columns(2)
            with col1:
             st.markdown("Top Emotions:-")
             st.dataframe(top_emotions_df)


            with col2:
             #st.markdown("Visualising through Bar Chart")
             fig.update_layout(
              template='plotly_white',
              height=500
             )
            st.plotly_chart(fig)
            st.divider()


             # SENTIMENTAL ANALYSIS
            st.subheader("SENTIMENT DETECTION")
            result= detect_overall_sentiment_avg(text)
            if "error" in result:
                st.write("Error:", result["error"])
            else:
                st.write("Overall Sentiment:", result["overall_sentiment"])
                st.write("Average Scores:",
                         pd.DataFrame(list(result['average_scores'].items()), columns=['Emotion', 'Score']))
            st.divider()

            #TONE OF SPEECH DETECTION.
            st.subheader("TONE OF SPEECH DETECTION.")
            output= classify_custom(text)
            col1, col2=st.columns(2)
            with col1:
                st.markdown(f"Predicted  : {output['predicted_category']}, score : {output['score']}")
                st.write("Other Top Predicted Category.")
                for label, score in output["all_categories"][1:6]:
                    st.write(f"Label :- {label}, Score:- {score}")

            with col2:
                labels=[]
                scores=[]
                for label, score in output["all_categories"][1:6]:
                    labels.append(label)
                    scores.append(score)

                fig= px.bar(x=labels, y= scores, color=labels, title="Other Top  5 Predicted Category.",
                            height=400
                            )
                st.plotly_chart(fig)
            st.divider()
            

            #SUMMARY GENERATION
        
            st.subheader("SUMMARY GENERATION.")
            output= summarize_large_text(text)
            st.write(output)







if a=="Process Csv File":
    st.header("Upload your CSV file.")
    uploaded_file= st.file_uploader("Choose a Csv file", type="csv")

    if uploaded_file is not None:
        df= pd.read_csv(uploaded_file)
        st.success("File Uploaded Successfully")
        st.divider()

        st.header("Choose filtering option.")

        # user selecting column to filter data
        column_name= st.selectbox("Select an column on which basis you want to filter the table", df.columns)

        #selecting unique values
        unique_vals= df[column_name].dropna().unique()
        selected_value= st.multiselect(f"Please choose value(s)  from {column_name}", unique_vals)

        #select the column that is textual column
        text_processing_column= st.selectbox("Select column for text analsis.", df.columns)

        # filtering
        if selected_value:
           filtered_df=  df[df[column_name].isin(selected_value)]
           filtered_df= filtered_df[text_processing_column]
           st.subheader("filtered Data.")
           st.dataframe(filtered_df)
           st.divider()
           text= " ".join(filtered_df.dropna().astype(str))

           # CLEANING OF TEXT
           cleaned = clean_text(text)
           tokens = clean_text_spacy(cleaned)
           #st.subheader("Cleaned and Lemmitized Text.")
           #st.write(" ".join(tokens) if tokens else "No meaning-full tokens Extracted")

           # WORD CLOUD
           if tokens:
               st.subheader("Word Cloud")
               wc_plot = show_wordcloud(tokens)
               st.pyplot(
                   wc_plot)  # st.pyplot is a function provided by Streamlit to display Matplotlib plots in a Streamlit app.
           st.divider()

           # N-GRAM ANALYSIS
           st.subheader("N-GRAM ANALYSIS")
           plot_top_ngrams_bar_chart(tokens, gram_n=3)
           st.divider()

           
           #EMOTION DETECTION
           st.subheader("EMOTION DETECTION.")
           result_dict = detect_emotion(text)

           if "error" in result_dict:
              st.error(f"Error in emotion detection: {result_dict['error']}")
           else:
              # Extract the pre-calculated values from the dictionary
              Emotion = result_dict["detect_emotion"]
              Score = result_dict["confidence"]
              top_emotions_df = result_dict["emotion_table"]
              fig = result_dict["plot"]

    
           st.write(f"PREDICTED EMOTION :- {Emotion}, with {Score * 100:.2f}% confidence")
    
           # Layout columns
           col1, col2 = st.columns(2)
           with col1:
             st.markdown("Top Emotions:-")
             st.dataframe(top_emotions_df)


           with col2:
             #st.markdown("Visualising through Bar Chart")
             fig.update_layout(
              template='plotly_white',
              height=500
             )
           st.plotly_chart(fig)
           st.divider()

           # SENTIMENT ANALYSIS
           st.subheader("SENTIMENT  DETECTION")
           result = detect_overall_sentiment_avg(text)
           if "error" in result:
               st.write("Error:", result["error"])
           else:
               st.write("Overall Sentiment:", result["overall_sentiment"])
               st.write("Average Scores:",
                        pd.DataFrame(list(result['average_scores'].items()), columns=['Emotion', 'Score']))
           st.divider()

           #TONE OF SPEECH DETECTION.
           st.subheader("TONE OF SPEECH DETECTION.")
           output= classify_custom(text)
           col1, col2=st.columns(2)
           with col1:
                st.markdown(f"Predicted  : {output['predicted_category']}, score : {output['score']}")
                st.write("Other Top Predicted Category.")
                for label, score in output["all_categories"][1:6]:
                    st.write(f"Label :- {label}, Score:- {score}")

           with col2:
                labels=[]
                scores=[]
                for label, score in output["all_categories"][1:6]:
                    labels.append(label)
                    scores.append(score)

                fig= px.bar(x=labels, y= scores, color=labels, title="Other Top  5 Predicted Category.",
                            height=400
                            )
                st.plotly_chart(fig)
           st.divider()