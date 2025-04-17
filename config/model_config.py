# Define available model options with language support

# Sentiment analysis models
SENTIMENT_MODEL_OPTIONS = {
    "English - RoBERTa": ("cardiffnlp/twitter-roberta-base-sentiment", None),
    # "Vietnamese - BERT (huynguyen9)": ("huynguyen9/bert-base-vietnamese-sentiment", None),
    # "Vietnamese - PhoBERT (vinai)": ("vinai/phobert-base", "vinai/phobert-base"),
    "Vietnamese - ViSoBERT (5CD-AI)": ("5CD-AI/vietnamese-sentiment-visobert", None),
    "Multilingual - XLM-RoBERTa": ("nlptown/xlm-roberta-base-sentiment", None)
}

# Named entity recognition models
ENTITY_MODEL_OPTIONS = {
    "English - SpaCy": "dslim/bert-base-NER",
    "Vietnamese - ELECTRA (NlpHUST)": "NlpHUST/ner-vietnamese-electra-base",
    "Multilingual - XLM-RoBERTa": "xlm-roberta-large"
}

# Summarization models (for future use)
SUMMARIZATION_MODEL_OPTIONS = {
    "English - T5": "t5-small",
    "Multilingual - mT5": "google/mt5-small"
}

# Default settings
DEFAULT_SETTINGS = {
    "sentiment_model": "English - RoBERTa",
    "entity_model": "English - SpaCy",
    "request_timeout": 15,
    "enable_js_rendering": False,
    "crawl_depth": 1,
    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
} 