# Text Sentiment Analyzer

A Python-based sentiment analysis tool that demonstrates full text processing pipeline and VADER sentiment analysis.

## Features Demonstrated

### Text Processing (as required by assignment):
- **Tokenization** - Splitting text into individual words/tokens
- **Stopword Removal** - Removing common words without semantic meaning
- **Stemming** - Reducing words to their root form
- **Lemmatization** - Converting words to their dictionary form
- **Text Normalization** - Full preprocessing pipeline

### Sentiment Analysis:
- **VADER-based analysis** - Rule-based sentiment analysis
- **Before/After comparison** - Shows impact of normalization on sentiment scores
- **Interactive input** - User can input custom text for analysis

## Project Structure

- `sentiment_analyzer.py` - Main analyzer with full processing pipeline demonstration
- `text_processor.py` - Text processing class with all normalization steps
- `requirements.txt` - Project dependencies

## Installation

```bash
# Clone repository
git clone https://github.com/akram911v/text-sentiment-analyzer.git
cd text-sentiment-analyzer

# Install dependencies
pip install -r requirements.txt
