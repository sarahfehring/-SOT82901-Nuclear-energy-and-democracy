# install the necessary libraries:
# pandas
# tensorflow
# keras
# matplotlib
# stop_words

import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from stop_words import get_stop_words

model = load_model("sentiment_analysis_model.keras")

def remove_stopwords_and_clean_text(text):
    stopwords = get_stop_words('french')
    special_stopwords = ['ministre', 'mme', 'sénat']

    text = text.lower().split(" ")
    cleaned_text = []

    for element in text:
        element = ''.join(e for e in element if e not in ".,;:_!?/()[]{}+<>*")
        if element not in stopwords and element not in cleaned_text and element not in special_stopwords:
            cleaned_text.append(element)
    return ' '.join(cleaned_text)

new_data = pd.read_csv("classify4me.csv", encoding='utf-8', usecols=['Text', 'Date'], delimiter='\t')

new_data["Text"] = new_data["Text"].apply(remove_stopwords_and_clean_text)

X_new = new_data["Text"]
dates = new_data["Date"]

max_words = 20000
max_len = 1000

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_new)
X_new_seq = tokenizer.texts_to_sequences(X_new)

X_new_pad = pad_sequences(X_new_seq, maxlen=max_len)

predictions = model.predict(X_new_pad)

predicted_labels = [np.argmax(pred) for pred in predictions]

sentiment_mapping = {0: "positive", 1: "neutral", 2: "negative"}
predicted_sentiments = [sentiment_mapping[label] for label in predicted_labels]

new_data["Predicted_Sentiment"] = predicted_sentiments


new_data.to_csv("new_data_with_predictions.csv", index=False, encoding='utf-8', sep='\t')