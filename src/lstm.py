# Импорт библиотек
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import random
import os

# Пути
input_path = 'input.txt'
output_model_path = 'lstm_model.keras'
output_gen_path = 'gen.txt'

# Загрузка текста
with open(input_path, encoding='utf-8') as f:
    text = f.read()

# Предобработка
seq_length = 30
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts([text])
total_chars = len(tokenizer.word_index) + 1

sequences = []
for i in range(seq_length, len(text)):
    seq = text[i - seq_length:i + 1]
    sequences.append(tokenizer.texts_to_sequences([seq])[0])
sequences = np.array(sequences)

xs = sequences[:, :-1]
ys = to_categorical(sequences[:, -1], num_classes=total_chars)

# Модель
model = Sequential([
    Embedding(input_dim=total_chars, output_dim=64, input_length=seq_length),
    LSTM(128),
    Dense(total_chars, activation='softmax')
])
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# Обучение с ранней остановкой
early_stopping = EarlyStopping(
    monitor='val_accuracy', patience=5, restore_best_weights=True
)

print("=== START TRAINING ===")
model.fit(
    xs, ys,
    epochs=50,
    batch_size=128,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)
print("=== TRAINING COMPLETE ===\n")

# 1) Генерация текста
def generate_text(seed_text, length=8000, temperature=1.0):
    result = seed_text
    for _ in range(length):
        enc = tokenizer.texts_to_sequences([result[-seq_length:]])[0]
        enc = pad_sequences([enc], maxlen=seq_length)
        probs = model.predict(enc, verbose=0)[0]
        probs = np.log(probs + 1e-8) / temperature
        exp = np.exp(probs)
        probs = exp / np.sum(exp)
        idx = np.random.choice(range(total_chars), p=probs)
        result += tokenizer.index_word[idx]
    return result

print("=== GENERATING TEXT ===")
seed = random.choice(text.split())
generated = generate_text(seed, temperature=1.0)
with open(output_gen_path, 'w', encoding='utf-8') as f:
    f.write(generated)
print(f"Сгенерированный текст сохранён в {output_gen_path}\n")

# 2) Сохранение полной модели
print("=== SAVING MODEL ===")
model.save(output_model_path)
print(f"Модель сохранена в {output_model_path}")
