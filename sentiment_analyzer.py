# sentiment_analyzer.py 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

class SimpleSentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
    
    def analyze_sentiment(self, text):
        scores = self.analyzer.polarity_scores(text)
        compound = scores['compound']
        
        if compound >= 0.05:
            sentiment = "Positive"
        elif compound <= -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
            
        return {
            'text': text,
            'sentiment': sentiment,
            'compound_score': compound,
            'scores': scores
        }

def main():
    analyzer = SimpleSentimentAnalyzer()
    
    # Test examples
    test_texts = [
        "I love this product! It's amazing!",
        "This is terrible. Worst experience ever.",
        "The product is okay, nothing special.",
        "The weather is nice today.",
        "I'm disappointed with the results."
    ]
    
    print("Sentiment Analysis Results:")
    print("=" * 50)
    
    for text in test_texts:
        result = analyzer.analyze_sentiment(text)
        print(f"Text: {text}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Score: {result['compound_score']:.4f}")
        print("-" * 30)

if __name__ == "__main__":
    main()
