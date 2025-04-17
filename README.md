# News Analysis System

A Streamlit-based application for analyzing news articles and text content using NLP techniques.

## Features

- Sentiment analysis of news articles and text
- Named entity recognition
- Multi-language support (English, Vietnamese, and multilingual models)
- URL content extraction
- Text input analysis
- Visualization of sentiment and entity distribution
- Analysis history tracking
- Export results as CSV or JSON

## Project Structure

```
sent_app/
├── analysis/              # NLP analysis functions
│   ├── __init__.py
│   └── nlp_analyzer.py
├── config/                # Configuration settings
│   ├── __init__.py
│   └── model_config.py
├── database/              # Database operations
│   ├── __init__.py
│   └── db_manager.py
├── utils/                 # Utility functions
│   ├── __init__.py
│   └── text_extractor.py
├── visualization/         # Data visualization
│   ├── __init__.py
│   └── visualizer.py
├── app.py                 # Main application logic
├── main.py                # Entry point
└── requirements.txt       # Dependencies
```

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   streamlit run main.py
   ```

## Usage

1. Navigate to the "Analysis" tab
2. Choose input method (URL, Text Input, or Bulk Analysis)
3. Select appropriate models for sentiment analysis and entity recognition
4. Click "Analyze" to process the content
5. View the results in the interactive dashboard
6. Export results as needed

## Models

The application supports various pre-trained models:

### Sentiment Analysis
- English - RoBERTa
- Vietnamese - ViSoBERT
- Multilingual - XLM-RoBERTa

### Named Entity Recognition
- English - SpaCy
- Vietnamese - ELECTRA
- Multilingual - XLM-RoBERTa

## License

MIT 