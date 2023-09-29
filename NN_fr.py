# install the necessary libraries:
# pandas
# tensorflow
# keras
# scikit
# matplotlib
# stop_words

import pandas as pd
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
from stop_words import get_stop_words
import matplotlib.pyplot as plt

def preprocess_and_mark_important(text):
    stopwords = get_stop_words('french')
    special_stopwords = ['ministre', 'mme', 'sénat']

    important_words = [
    "centrale", "centrales", "centrale électrique", "centrale nucléaire",
    "fermeture", "fermé", "fermée", "fermés", "fermées", "clôture", "cessation", "arrêt",
    "site", "sites", "emplacement", "endroit", "lieu",
    "nucléaires", "nucléaire", "nucléaires", "atomique", "atomiques",
    "prévention", "préventions", "préventif", "préventive", "prophylaxie",
    "stockage", "stocké", "stockée", "stockés", "stockées", "entreposage", "conservation", "réserve",
    "déchets", "déchet", "ordures", "détritus", "rebuts",
    "radioactifs", "radioactif", "radioactives", "irradié", "irradiée",
    "risque", "risques", "danger", "menace", "péril",
    "matières", "matière", "substance", "élément",
    "réacteur", "réacteurs", "générateur", "réacteur nucléaire",
    "atomique", "atomiques", "nucléaire", "nucléaires",
    "radionucléides", "radionucléide", "radioéléments", "radioisotopes",
    "électronucléaires", "électronucléaire", "énergie nucléaire", "électro-nucléaire",
    "médecine", "médecines", "santé", "soins médicaux",
    "cancers", "cancer", "tumeurs", "néoplasmes",
    "fermé", "fermée", "fermés", "fermées", "clos", "verrouillé", "hermétique",
    "sûreté", "sûretés", "sécurité", "protection", "fiabilité",
    "protection", "protections", "sécurité", "sauvegarde", "défense",
    "environnement", "environnements", "nature", "écosystème", "cadre",
    "l'environnement", "environnements", "le cadre", "le milieu",
    "santé", "santé", "bien-être", "forme", "vitalité",
    "publique", "publique", "collectif", "communautaire", "social",
    "prévention", "préventions", "préventif", "prophylaxie", "préservation",
    "danger", "dangers", "risque", "menace", "péril", "victime", "victimes", "accident", "accidents"
    ]

    text = text.lower().split(" ")
    tokenized_text = []

    for element in text:
        element = ''.join(e for e in element if e not in ".,;:_!?/()[]{}+<>*")
        if element not in stopwords and element not in special_stopwords:
            if element in important_words:
                tokenized_text.append('IMP_' + element)
            else:
                tokenized_text.append(element)

    return ' '.join(tokenized_text)

political_speeches_df = pd.read_csv("data.csv", delimiter='\t', encoding='utf-8')

political_speeches_df["Text"] = political_speeches_df["Text"].apply(preprocess_and_mark_important)
print(political_speeches_df)

max_words = 20000
max_len = 1000

X = political_speeches_df["Text"]
y = political_speeches_df["Classification"]
dates = political_speeches_df["Date"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

class_mapping = {"positive": 0, "neutral": 1, "negative": 2}
y_train = y_train.map(class_mapping)
y_test = y_test.map(class_mapping)

y_train_onehot = to_categorical(y_train, num_classes=3)
y_test_onehot = to_categorical(y_test, num_classes=3)

model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=64, input_length=max_len))
model.add(SpatialDropout1D(0.1))
model.add(LSTM(32, dropout=0.5, recurrent_dropout=0.1))
model.add(Dense(3, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

plot_model(model,to_file='model.png',show_shapes=True,show_dtype=False,show_layer_names=False,rankdir='TB',expand_nested=True,dpi=96,layer_range=None,show_layer_activations=True,show_trainable=True)

history = model.fit(X_train_pad, y_train_onehot, epochs=10, batch_size=128, validation_split=0.2)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

loss, accuracy = model.evaluate(X_test_pad, y_test_onehot)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

model.save("sentiment_analysis_model.keras")