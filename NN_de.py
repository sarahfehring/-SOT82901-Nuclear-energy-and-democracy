#install the necessary libraries:
# pandas
# tensorflow
# keras
# scikit
# matplotlib

import pandas as pd
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Attention, Input
from keras.models import Sequential, Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from keras_self_attention import SeqSelfAttention


def preprocess_and_mark_important(text):
    special_stopwords = ["Bundestag", "Bundesrat", "Bündnis", "Grünen", "CDU", "CSU", "SPD", "LINKE", "FDP", "AfD"]

    important_words = ["Unfall", "terroristisch", "terroristische", "terroristischen", "Potential", "Zukunftstechnologie", "teuer", "Sicherheit", "Risiko", "Laufzeitverlängerung", "Kosten", "umweltfreundlich",
                       "nuklear", "nuklearer", "nukleare","Abschaltung", "Ukraine", "Ausfall", "Weiterbetrieb", "Weiterbetriebs", "Laufzeit", "Abschaltung", "Ausstieg", "Ausstiegs", "abschalten", "schädlich", "Abfall",
                       "radioaktiv", "radioaktive", "radioaktives", "radioaktiver", "Stilllegung", "Abbau", "nachhaltig", "nachhaltigen", "nachhaltige", "nachhaltiger", "Nachhaltigkeit", "unsicher", "Risiken", "abgeschafft",
                       ]

    text = text.lower().split(" ")
    tokenized_text = []

    for element in text:
        element = ''.join(e for e in element if e not in ".,;:_!?/()[]{}+<>*123456789")
        if element not in special_stopwords:
            if element in important_words:
                tokenized_text.append('IMP_' + element)
            else:
                tokenized_text.append(element)

    return ' '.join(tokenized_text)

political_speeches_df = pd.read_csv("wp_all_sentiment.csv", encoding='utf-8', sep=";")
political_speeches_df.filtered_text = political_speeches_df.filtered_text.astype(str)

political_speeches_df["filtered_text"] = political_speeches_df["filtered_text"].apply(preprocess_and_mark_important)

max_words = 15000
max_len = 800

X = political_speeches_df["filtered_text"]
y = political_speeches_df["sentiment"]
dates = political_speeches_df["date"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

sentiment_mapping = {"positive": 0, "neutral": 1, "negative": 2}
y_train = y_train.map(sentiment_mapping)
y_test = y_test.map(sentiment_mapping)

y_train_onehot = to_categorical(y_train, num_classes=3)
y_test_onehot = to_categorical(y_test, num_classes=3)

# Model architecture
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=192, input_length=max_len))
model.add(SpatialDropout1D(0.5))
model.add(LSTM(32, dropout=0.30000000000000004, recurrent_dropout=0.2))
model.add(Dense(3, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

#plot_model(model,to_file='model.png',show_shapes=True,show_dtype=False,show_layer_names=False,rankdir='TB',expand_nested=True,dpi=96,layer_range=None,show_layer_activations=True,show_trainable=True)

history = model.fit(X_train_pad, y_train_onehot, epochs=10, batch_size=128, validation_split=0.2)
plt.figure(figsize=(10, 5))

# plotting
# loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# eval on test data
loss, accuracy = model.evaluate(X_test_pad, y_test_onehot)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

model.save("sentiment_analysis_model.keras")

