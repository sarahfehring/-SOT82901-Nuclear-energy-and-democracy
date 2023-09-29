#install the necessary libraries:
# numpy
# pandas
# tensorflow
# keras

import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

model = load_model("sentiment_analysis_model.keras")

def preprocess(text):
    special_stopwords = ["bundestag", "bundesrat", "bündnis", "grünen", "cdu", "csu", "spd", "linke", "fdp", "afd"]

    text = text.lower().split(" ")
    cleaned_text = []

    for element in text:
        element = ''.join(e for e in element if e not in ".,;:_!?/()[]{}+<>*1234567890")
        if element not in cleaned_text and element not in special_stopwords:
            cleaned_text.append(element)
    return ' '.join(cleaned_text)


new_data = pd.read_csv("classify4me.csv", encoding='utf-8', sep=";")  # Replace with your new data file
print(new_data)
new_data["filtered_text"] = new_data["filtered_text"].apply(preprocess)
X_new = new_data["filtered_text"]

max_words = 15000
max_len = 800

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_new)
X_new_seq = tokenizer.texts_to_sequences(X_new)

X_new_pad = pad_sequences(X_new_seq, maxlen=max_len)

predictions = model.predict(X_new_pad)

predicted_labels = [np.argmax(pred) for pred in predictions]

sentiment_mapping = {0: "positive",1: "neutral",2:"negative"}
predicted_sentiments = [sentiment_mapping[label] for label in predicted_labels]

new_data["predicted_sentiment"] = predicted_sentiments

new_data.to_csv("predictions.csv", index=False, encoding='utf-8')