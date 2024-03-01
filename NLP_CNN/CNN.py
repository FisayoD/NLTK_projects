import re
import fitz
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nrclex import NRCLex
import os
import numpy as np


from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense


def preprocess_text(doc):
    doc = doc.lower()
    doc = re.sub(r'\W', ' ', doc)
    doc = re.sub(r'\s+[a-zA-Z]\s+', ' ', doc)
    doc = re.sub(r'\^[a-zA-Z]\s+', ' ', doc)
    doc = re.sub(r'\s+', ' ', doc, flags=re.I)
   
    lemmatizer = WordNetLemmatizer()
    tokens = doc.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in set(stopwords.words('english'))]
    return ' '.join(tokens)

# def analyze_emotion_sentiment(text):
#     emotion = NRCLex(text)
#     return emotion.affect_frequencies


def preprocess_document(doc_name):
    current_dir = os.path.dirname(__file__)  # Get the directory where the script is located
    file_path = os.path.join(current_dir, doc_name)  # Construct the full file path

    # Check if the file exists
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        return None

    # Open the PDF file
    try:
        with fitz.open(file_path) as doc:
            text = ""
            for page in doc:
                text += page.get_text()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    # Preprocess the extracted text
    processed_text = preprocess_text(text)
    # analyze_emotion_sentiment(text)
    return processed_text

document1 = "we_thought_it_was_oil_it_was_blood.pdf"
document2 = "Criminal crude.pdf"
doc1 = preprocess_document(document1)
doc2 = preprocess_document(document2)
print(doc1)
print(doc2)
# document_paths = [document1, document2]
# texts = []

# for path in document_paths:
#     try:
#         with fitz.open(path) as doc:
#             for page in doc:
#                 text = page.get_text()
#                 sentences = text.split('\n')
#                 texts.extend(sentences)
#     except FileNotFoundError as e:
#         print(f"Error opening {path}: {e}")




# sentences = [...] 
# tokenizer = Tokenizer(num_words=5000)
# tokenizer.fit_on_texts(sentences)
# sequences = tokenizer.texts_to_sequences(sentences)
# data = pad_sequences(sequences, maxlen=100)

# #CNN aspect
# model = Sequential()
# model.add(Embedding(input_dim=5000, output_dim=50, input_length=100))
# model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
# model.add(GlobalMaxPooling1D())
# model.add(Dense(units=1, activation='sigmoid'))

# # Compile model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load dataset
def load_reviews(directory):
    texts = []
    labels = []
    for label in ["positive", "negative"]:
        dir_name = os.path.join(directory, label)
        for fname in os.listdir(dir_name):
            with open(os.path.join(dir_name, fname), 'r', encoding='utf-8') as file:
                texts.append(file.read())
                labels.append(0 if label == "negative" else 1)
    return texts, np.array(labels)

texts, labels = load_reviews('Movie_Data.csv')

# Preprocess dataset
max_words = 10000  # Consider only the top 10,000 words in the dataset
maxlen = 100  # Cut reviews after 100 words
embedding_dim = 50  # Dimensionality of the embedding layer

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print(f'Found {len(word_index)} unique tokens.')

data = pad_sequences(sequences, maxlen=maxlen)

# Split dataset into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Conv1D(128, 5, activation='relu'))
model.add(MaxPooling1D(5))
model.add(Conv1D(128, 5, activation='relu'))
model.add(MaxPooling1D(5))
model.add(Conv1D(128, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc}')

