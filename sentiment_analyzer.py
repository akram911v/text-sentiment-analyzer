# sentiment_analyzer.py
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import string
import pandas as pd

# Download necessary NLTK resources
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt_tab')
except:
    print("NLTK download skipped - using existing data")

class TextProcessor:
    """Class for text processing"""
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

 def tokenize(self, text):
    """Tokenize text"""
    # Simple whitespace-based tokenization that doesn't require punkt_tab
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.lower().split()
     
    def remove_stopwords(self, tokens):
        """Remove stop words"""
        return [token for token in tokens if token not in self.stop_words and token not in string.punctuation]

    def stem(self, tokens):
        """Stem tokens"""
        return [self.stemmer.stem(token) for token in tokens]

    def lemmatize(self, tokens):
        """Lemmatize tokens"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def normalize(self, text):
        """Full text normalization"""
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.stem(tokens)
        tokens = self.lemmatize(tokens)
        return tokens

class SentimentAnalyzer:
    """Class for sentiment analysis"""
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.processor = TextProcessor()

    def analyze_sentiment(self, text):
        """Analyze sentiment of text"""
        sentiment_scores = self.analyzer.polarity_scores(text)
        normalized_tokens = self.processor.normalize(text)
        normalized_text = ' '.join(normalized_tokens)
        normalized_scores = self.analyzer.polarity_scores(normalized_text)

        return {
            'original_text': text,
            'normalized_text': normalized_text,
            'original_scores': sentiment_scores,
            'normalized_scores': normalized_scores,
            'sentiment': self._get_sentiment_label(sentiment_scores['compound'])
        }

    def _get_sentiment_label(self, compound_score):
        """Get sentiment label based on compound score"""
        if compound_score >= 0.05:
            return "Positive"
        elif compound_score <= -0.05:
            return "Negative"
        else:
            return "Neutral"

def demonstrate_analyzer():
    """Demonstrate the sentiment analyzer"""
    analyzer = SentimentAnalyzer()
    texts = [
        "I absolutely love this product! It's amazing and works perfectly.",
        "This is the worst experience I've ever had. Terrible service!",
        "The product is okay. It has some good features but also some drawbacks.",
        "The weather is nice today. I went for a walk in the park.",
        "I'm feeling very disappointed with the results. Not what I expected at all."
    ]
    results = []
    for text in texts:
        result = analyzer.analyze_sentiment(text)
        results.append({
            'Text': text[:50] + "..." if len(text) > 50 else text,
            'Normalized Text': ' '.join(result['normalized_text'].split()[:10]) + "...",
            'Sentiment': result['sentiment'],
            'Compound Score': result['original_scores']['compound']
        })
    df = pd.DataFrame(results)
    return df

def analyze_custom_text():
    """Analyze custom text from user input"""
    analyzer = SentimentAnalyzer()
    print("Enter text for sentiment analysis (or 'quit' to exit):")
    while True:
        user_input = input()
        if user_input.lower() == 'quit':
            break
        result = analyzer.analyze_sentiment(user_input)
        print("\nResults:")
        print(f"Original text: {result['original_text']}")
        print(f"Normalized text: {result['normalized_text']}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Compound Score: {result['original_scores']['compound']:.4f}")
        print(f"Positive Score: {result['original_scores']['pos']:.4f}")
        print(f"Neutral Score: {result['original_scores']['neu']:.4f}")
        print(f"Negative Score: {result['original_scores']['neg']:.4f}")
        print("\n" + "="*50)
        print("Enter next text (or 'quit' to exit):")

def compare_normalization_effect():
    """Compare sentiment scores before and after normalization"""
    analyzer = SentimentAnalyzer()
    sample_text = "I'm extremely happy with the fantastic results! The product is absolutely amazing and wonderful."
    result = analyzer.analyze_sentiment(sample_text)
    print("Comparison of normalization effect:")
    print(f"Original text: {sample_text}")
    print(f"Normalized text: {result['normalized_text']}")
    print(f"Original compound score: {result['original_scores']['compound']:.4f}")
    print(f"Normalized compound score: {result['normalized_scores']['compound']:.4f}")
    print(f"Difference: {abs(result['original_scores']['compound'] - result['normalized_scores']['compound']):.4f}")

if __name__ == "__main__":
    print("Loading and initializing...")
    results_df = demonstrate_analyzer()
    print("Sentiment analysis results:")
    print(results_df)
    print("\n" + "="*50)
    compare_normalization_effect()
    print("\n" + "="*50)
    analyze_custom_text()
