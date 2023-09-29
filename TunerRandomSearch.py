import pandas as pd
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
from keras_tuner.tuners import RandomSearch
import matplotlib.pyplot as plt

political_speeches_df = pd.read_csv("data.csv", delimiter='\t', encoding='utf-8')
print(political_speeches_df)

max_words = 100000
max_len = 2000

X = political_speeches_df["Text"]
y = political_speeches_df["Classification"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

sentiment_mapping = {"positive": 1, "neutral": 0, "negative": -1}
y_train = y_train.map(sentiment_mapping)
y_test = y_test.map(sentiment_mapping)

y_train_onehot = to_categorical(y_train, num_classes=3)
y_test_onehot = to_categorical(y_test, num_classes=3)

def build_model(hp):
    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=hp.Int('embedding_dim', min_value=32, max_value=256, step=32), input_length=max_len))
    model.add(SpatialDropout1D(hp.Float('spatial_dropout', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(LSTM(hp.Int('lstm_units', min_value=32, max_value=128, step=32), dropout=hp.Float('lstm_dropout', min_value=0.1, max_value=0.5, step=0.1), recurrent_dropout=hp.Float('recurrent_dropout', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(Dense(3, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    directory='keras_tuner_dir',
    project_name='sentiment_analysis_tuning',
    overwrite=True
)

tuner.search(X_train_pad, y_train_onehot, epochs=10, batch_size=128, validation_split=0.2)

best_model = tuner.get_best_models(num_models=1)[0]

plot_model(best_model, to_file='model.png', show_shapes=True, show_dtype=False, show_layer_names=False, rankdir='LR', expand_nested=True, dpi=96, layer_range=None, show_layer_activations=True, show_trainable=True)

history = best_model.fit(X_train_pad, y_train_onehot, epochs=10, batch_size=128, validation_split=0.2)

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

loss, accuracy = best_model.evaluate(X_test_pad, y_test_onehot)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

best_model.save("sentiment_analysis_model.keras")