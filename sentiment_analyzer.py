# sentiment_analyzer.py - Updated with full text processing
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from text_processor import TextProcessor
import pandas as pd

class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.text_processor = TextProcessor()
    
    def analyze_with_processing(self, text):
        """Analyze sentiment while showing all text processing steps"""
        print("\n" + "="*60)
        print("ANALYZING TEXT:", text)
        print("="*60)
        
        # Show all text processing steps
        processing_result = self.text_processor.full_normalization(text)
        
        # Analyze original text sentiment
        original_scores = self.analyzer.polarity_scores(text)
        
        # Analyze normalized text sentiment
        normalized_text = ' '.join(processing_result['lemmatized_tokens'])
        normalized_scores = self.analyzer.polarity_scores(normalized_text)
        
        # Determine sentiment labels
        original_sentiment = self._get_sentiment_label(original_scores['compound'])
        normalized_sentiment = self._get_sentiment_label(normalized_scores['compound'])
        
        print("\nSENTIMENT RESULTS:")
        print(f"Original text sentiment: {original_sentiment} (score: {original_scores['compound']:.4f})")
        print(f"Normalized text sentiment: {normalized_sentiment} (score: {normalized_scores['compound']:.4f})")
        print(f"Normalized text used: '{normalized_text}'")
        
        return {
            'original_text': text,
            'normalized_text': normalized_text,
            'original_sentiment': original_sentiment,
            'normalized_sentiment': normalized_sentiment,
            'original_score': original_scores['compound'],
            'normalized_score': normalized_scores['compound']
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
    """Demonstrate the analyzer with multiple examples"""
    analyzer = SentimentAnalyzer()
    
    test_texts = [
        "I absolutely love this product! It's amazing and works perfectly!",
        "This is the worst experience I've ever had. Terrible service!",
        "The product is okay. It has some good features but also some drawbacks.",
        "I'm feeling very happy with the excellent results and wonderful outcomes!",
        "This is disappointing and unsatisfactory. Not what I expected at all."
    ]
    
    results = []
    for text in test_texts:
        result = analyzer.analyze_with_processing(text)
        results.append(result)
    
    return results

def interactive_analysis():
    """Interactive mode for user input"""
    analyzer = SentimentAnalyzer()
    
    print("\n" + "="*60)
    print("INTERACTIVE SENTIMENT ANALYSIS")
    print("="*60)
    print("Enter text to analyze (or 'quit' to exit):")
    
    while True:
        user_input = input("\nYour text: ").strip()
        
        if user_input.lower() == 'quit':
            break
        elif user_input:
            analyzer.analyze_with_processing(user_input)
        else:
            print("Please enter some text.")
    
    print("Thank you for using the sentiment analyzer!")

if __name__ == "__main__":
    print("Text Sentiment Analyzer with Full Text Processing")
    print("This demonstrates: Tokenization, Stopword Removal, Stemming, Lemmatization")
    
    # Demonstrate with test examples
    print("\nDEMONSTRATION WITH TEST EXAMPLES:")
    results = demonstrate_analyzer()
    
    # Show summary table
    print("\n" + "="*60)
    print("SUMMARY OF RESULTS")
    print("="*60)
    
    summary_data = []
    for result in results:
        summary_data.append({
            'Text': result['original_text'][:40] + "..." if len(result['original_text']) > 40 else result['original_text'],
            'Original Sentiment': result['original_sentiment'],
            'Original Score': f"{result['original_score']:.4f}",
            'Normalized Sentiment': result['normalized_sentiment'],
            'Normalized Score': f"{result['normalized_score']:.4f}"
        })
    
    df = pd.DataFrame(summary_data)
    print(df.to_string(index=False))
    
    # Start interactive mode
    interactive_analysis()
