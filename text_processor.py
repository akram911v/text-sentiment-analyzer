# text_processor.py
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re

class TextProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
    
    def tokenize(self, text):
        """Tokenize text into words"""
        # Remove punctuation and extra spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.lower().split()
        return tokens
    
    def remove_stopwords(self, tokens):
        """Remove common stop words"""
        return [token for token in tokens if token not in self.stop_words]
    
    def stem(self, tokens):
        """Apply stemming to tokens"""
        return [self.stemmer.stem(token) for token in tokens]
    
    def lemmatize(self, tokens):
        """Apply lemmatization to tokens"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def full_normalization(self, text):
        """Apply all normalization steps and return results at each stage"""
        print("=== TEXT PROCESSING STEPS ===")
        print(f"Original text: {text}")
        
        # Step 1: Tokenization
        tokens = self.tokenize(text)
        print(f"1. After tokenization: {tokens}")
        
        # Step 2: Stopword removal
        tokens_no_stopwords = self.remove_stopwords(tokens)
        print(f"2. After stopword removal: {tokens_no_stopwords}")
        
        # Step 3: Stemming
        stemmed_tokens = self.stem(tokens_no_stopwords)
        print(f"3. After stemming: {stemmed_tokens}")
        
        # Step 4: Lemmatization
        lemmatized_tokens = self.lemmatize(tokens_no_stopwords)
        print(f"4. After lemmatization: {lemmatized_tokens}")
        
        return {
            'tokens': tokens,
            'tokens_no_stopwords': tokens_no_stopwords,
            'stemmed_tokens': stemmed_tokens,
            'lemmatized_tokens': lemmatized_tokens
        }

# Example usage
if __name__ == "__main__":
    processor = TextProcessor()
    sample_text = "I am running quickly through the beautiful parks and enjoying the wonderful weather"
    result = processor.full_normalization(sample_text)
