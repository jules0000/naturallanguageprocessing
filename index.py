# Import necessary libraries
import pandas as pd  # For data manipulation
import nltk  # Natural Language Toolkit
from nltk.corpus import stopwords  # Stopwords corpus
from nltk.tokenize import word_tokenize  # Tokenization
from nltk.stem import WordNetLemmatizer  # Lemmatization
import re  # Regular expressions
from spellchecker import SpellChecker  # Spell checking
import spacy  # For entity recognition




# Download NLTK resources
nltk.download('punkt')  # Tokenizer
nltk.download('stopwords')  # Stopwords
nltk.download('wordnet')  # WordNet lexical database

# Load dataset
data = pd.read_csv('spam.csv', encoding='latin1')  # Import file with specified encoding

# Initialize NLTK components
stop_words = set(stopwords.words('english'))  # Set of English stopwords
wordnet_lemmatizer = WordNetLemmatizer()  # Initialize WordNetLemmatizer
spell_checker = SpellChecker()  # Initialize SpellChecker
nlp = spacy.load("en_core_web_sm")  # Load English language model from spaCy

# Function to preprocess text
def preprocess_text(text, spell_check=False, pos_tagging=False, entity_recognition=False):
    # Print original text
    print("Original Text:", text)

    # Tokenization
    tokens = word_tokenize(text)  # Tokenize the text
    print("Tokens after tokenization:", tokens)

    # Lemmatization
    tokens = [wordnet_lemmatizer.lemmatize(token) for token in tokens]  # Lemmatize the tokens
    print("Tokens after lemmatization:", tokens)

    # Stopword Removal
    tokens = [token for token in tokens if token not in stop_words]  # Remove stopwords
    print("Tokens after stopword removal:", tokens)

    # Spell Checking if enabled
    if spell_check:
        print("Tokens before spell checking:", tokens)
        tokens = [spell_checker.correction(token) for token in tokens]  # Correct spelling errors
        print("Tokens after spell checking:", tokens)

    # POS Tagging if enabled
    if pos_tagging and len(tokens) > 0:  # Perform POS tagging only if there are tokens after stopword removal
        pos_tags = nltk.pos_tag(tokens)  # Perform POS tagging
        print("POS Tags:", pos_tags)

    # Entity Recognition if enabled
    if entity_recognition:
        doc = nlp(' '.join(tokens))  # Create a spaCy document
        entities = [(entity.text, entity.label_) for entity in doc.ents]  # Extract entities
        print("Entities:", entities)

    # Join tokens back into a single string
    preprocessed_text = ' '.join(tokens)
    print("Preprocessed Text:", preprocessed_text)

    return preprocessed_text

# Determine if spell checking is needed based on the dataset
def needs_spell_checking(data_column):
    for text in data_column:
        # Example condition: Check if the dataset contains text with spelling errors
        if 'spelling_error_indicator' in text:
            return True
    return False

# Determine if POS tagging is needed based on the dataset
def needs_pos_tagging(data_column):
    for text in data_column:
        # Example condition: Check if the dataset contains text where POS tagging is required
        if 'pos_tagging_indicator' in text:
            return True
    return False

# Determine if entity recognition is needed based on the dataset
def needs_entity_recognition(data_column):
    for text in data_column:
        # Example condition: Check if the dataset contains text where entity recognition is required
        if 'entity_recognition_indicator' in text:
            return True
    return False

# Access the text column from the dataset
data_column = data['v2']

# Determine if spell checking, POS tagging, and entity recognition are needed
spell_check_needed = needs_spell_checking(data_column)
pos_tagging_needed = needs_pos_tagging(data_column)
entity_recognition_needed = needs_entity_recognition(data_column)

# Apply preprocessing to each text entry in the dataset
for text in data_column:
    preprocessed_text = preprocess_text(text, spell_check=spell_check_needed,
                                        pos_tagging=pos_tagging_needed,
                                        entity_recognition=entity_recognition_needed)
