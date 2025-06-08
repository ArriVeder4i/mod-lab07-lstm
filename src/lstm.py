import os
import random
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

# ==== ПАРАМЕТРЫ ====
INPUT_PATH = 'input.txt'
MODEL_WEIGHTS = 'best_weights.h5'
GENERATED_PATH = 'gen.txt'
MAX_LENGTH = 10
STEP = 1
EMBEDDING_DIM = 100
LSTM_UNITS = 128
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.01
PATIENCE_LR = 3  # patience for ReduceLROnPlateau
TEMPERATURE = 0.2
GENERATED_WORDS = 1000


# ==== ФУНКЦИИ ====

def load_text(path: str) -> str:
    """Читает и возвращает содержимое файла."""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def prepare_data(text: str):
    """Токенизация по словам и подготовка входных (X) и целевых (y) массивов."""
    words = text.split()
    vocab = sorted(set(words))
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    idx_to_word = {i: w for i, w in enumerate(vocab)}

    # Преобразуем в индексы
    seq_indices = [word_to_idx[w] for w in words]

    sentences, next_words = [], []
    for i in range(0, len(seq_indices) - MAX_LENGTH, STEP):
        sentences.append(seq_indices[i: i + MAX_LENGTH])
        next_words.append(seq_indices[i + MAX_LENGTH])

    # Создаём массивы
    X = np.array(sentences, dtype=np.int32)
    y = np.zeros((len(sentences), len(vocab)), dtype=np.bool_)
    for i, idx in enumerate(next_words):
        y[i, idx] = 1

    return X, y, word_to_idx, idx_to_word, len(vocab)


def build_model(vocab_size: int) -> tf.keras.Model:
    """Строит и компилирует модель."""
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=MAX_LENGTH),
        LSTM(LSTM_UNITS),
        Dense(vocab_size),
        Activation('softmax'),
    ])
    optimizer = RMSprop(learning_rate=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model


def sample(preds: np.ndarray, temperature: float) -> int:
    """Сэмплинг с учётом температуры."""
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.argmax(np.random.multinomial(1, preds, 1))


def generate_and_save(model: tf.keras.Model,
                      idx_to_word: dict,
                      seq_indices: list,
                      num_words: int,
                      temperature: float,
                      out_path: str):
    """Генерирует текст и сохраняет в файл."""
    start_idx = random.randint(0, len(seq_indices) - MAX_LENGTH - 1)
    sentence = seq_indices[start_idx: start_idx + MAX_LENGTH]
    generated = sentence.copy()

    for _ in range(num_words):
        x_pred = np.array([sentence], dtype=np.int32)
        preds = model.predict(x_pred, verbose=0)[0]
        next_idx = sample(preds, temperature)
        generated.append(next_idx)
        sentence = sentence[1:] + [next_idx]

    text = ' '.join(idx_to_word[i] for i in generated)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Сгенерированный текст сохранён в {out_path}")


def main():
    # 1. Загрузка и подготовка данных
    text = load_text(INPUT_PATH)
    X, y, w2i, i2w, vocab_size = prepare_data(text)
    print(f"Загружено {len(text.split())} слов, словарь: {vocab_size} токенов.")

    # 2. Построение модели
    model = build_model(vocab_size)

    # 3. Колбэки
    checkpoint = ModelCheckpoint(
        filepath=MODEL_WEIGHTS,
        save_best_only=True,
        monitor='loss',
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=PATIENCE_LR,
        verbose=1
    )

    # 4. Обучение
    model.fit(
        X, y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[checkpoint, reduce_lr],
        verbose=1
    )

    # 5. Генерация и сохранение текста
    generate_and_save(
        model, i2w, [idx for idx in sum([[w2i[w] for w in text.split()]], [])],
        num_words=GENERATED_WORDS,
        temperature=TEMPERATURE,
        out_path=GENERATED_PATH
    )


if __name__ == "__main__":
    main()
