import nltk
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer  # For VADER
from nrclex import NRCLex  # For NRCLex
import string
import ssl

# For handling file operations and possibly working with lexicons
import os

sia = SentimentIntensityAnalyzer()
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Ensure you have downloaded the necessary NLTK data
nltk.download('punkt')
nltk.download('vader_lexicon')

# List of file paths
file_paths = [
    'Independent_Study_Codebase/reviewBarbie1.txt',
    'Independent_Study_Codebase/reviewBarbie2.txt',
    'Independent_Study_Codebase/reviewBarbie3.txt'
]

for file_path in file_paths:
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()

    # Preprocess data
    data = data.translate(str.maketrans('', '', string.punctuation)).lower()

    # VADER Sentiment Analysis
    vader_scores = sia.polarity_scores(data)
    print(f"VADER Scores for {file_path}:", vader_scores)

    # NRCLex Emotion Analysis for detailed emotion analysis
    text_object = NRCLex(data)
    emotion_scores = text_object.affect_frequencies  # This already uses frequency counting
    print(f"Emotion scores (NRCLex) for {file_path}:", emotion_scores)

    # Print a newline for better readability between files
    print("\n")