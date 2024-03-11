import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# ข้อมูลตัวอย่าง
conversations = [
    ("สวัสดี", "สวัสดีครับ"),
    ("ทำอะไรอยู่", "กำลังทำการเรียน"),
    ("มีอะไรให้ช่วยไหม", "ไม่มีครับ ขอบคุณ"),
    ("ลาก่อน", "ลาก่อนครับ"),
    ("วันนี้อากาศดีจังเลย", None)  # ตัวอย่างคำถามที่ไม่มีคำตอบ
]

# แยกข้อความและคำตอบ
questions, answers = zip(*conversations)

# กำหนดพารามิเตอร์
vocab_size = 1000
max_length = 20
embedding_dim = 16
trunc_type = 'post'
padding_type = 'post'
oov_token = "<OOV>"

# โมเดลการเทรน
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(questions + [ans if ans else '' for ans in answers])

word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(questions)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# สร้างโมเดล LSTM
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    LSTM(64),
    Dense(vocab_size, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# สร้างตัวแบ่งข้อมูล
num_samples = len(padded_sequences)
split = int(0.8 * num_samples)

train_questions = padded_sequences[:split]
train_answers = to_categorical([0 if ans is None else tokenizer.texts_to_sequences([ans])[0] for ans in answers[:split]], num_classes=vocab_size)

test_questions = padded_sequences[split:]
test_answers = to_categorical([0 if ans is None else tokenizer.texts_to_sequences([ans])[0] for ans in answers[split:]], num_classes=vocab_size)

# การฝึกโมเดล
model.fit(train_questions, train_answers, epochs=10, batch_size=32, validation_data=(test_questions, test_answers))

# การทำนาย
predicted_answers = model.predict_classes(test_questions)
for i, ans in enumerate(predicted_answers):
    if ans != 0:
        print("คำตอบที่ทำนาย:", tokenizer.sequences_to_texts([[ans]])[0])
    else:
        print("ไม่มีคำตอบ")