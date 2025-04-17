import re
import streamlit as st
from transformers import pipeline
import nltk
import os

# Set NLTK data path explicitly
nltk.data.path.append(os.path.join(os.path.expanduser("~"), "nltk_data"))

# Then in your initialization code
try:
    # Try to use the tokenizer
    from nltk.tokenize import sent_tokenize
    sent_tokenize("This is a test.")
except (LookupError, ImportError):
    # If it fails, download punkt explicitly
    nltk.download('punkt', download_dir=os.path.join(os.path.expanduser("~"), "nltk_data"))
    # Try importing again
    from nltk.tokenize import sent_tokenize

# Cache the NLP pipelines
@st.cache_resource(show_spinner=True)
def load_sentiment_pipeline(model_name, tokenizer_name=None):
    with st.spinner(f"Loading sentiment model {model_name}..."):
        if tokenizer_name is None:
            tokenizer_name = model_name
        return pipeline("sentiment-analysis", model=model_name, tokenizer=tokenizer_name)

@st.cache_resource(show_spinner=False)
def load_ner_pipeline(model_name):
    return pipeline("ner", model=model_name)

@st.cache_resource(show_spinner=False)
def load_summarization_pipeline(model_name):
    return pipeline("summarization", model=model_name)

@st.cache_resource(show_spinner=False)
def load_keywords_pipeline(model_name):
    return pipeline("feature-extraction", model=model_name)

def standardize_sentiment_label(label, score=None):
    """
    Standardize sentiment labels from different models to POSITIVE, NEGATIVE, NEUTRAL
    """
    # For models that return LABEL_0, LABEL_1, etc.
    if label.startswith('LABEL_'):
        # Common mapping for Vietnamese models
        # Often LABEL_0 = negative, LABEL_1 = neutral, LABEL_2 = positive
        # But verify this for your specific models
        label_map = {
            'LABEL_0': 'NEGATIVE',
            'LABEL_1': 'NEUTRAL',
            'LABEL_2': 'POSITIVE'
        }
        return label_map.get(label, 'NEUTRAL')
    
    # For models that return numeric labels
    elif label.isdigit():
        label_num = int(label)
        if label_num == 0:
            return 'NEGATIVE'
        elif label_num == 1:
            return 'NEUTRAL'
        else:
            return 'POSITIVE'
    
    # For models that return sentiment names but might be lowercase
    elif 'positive' in label.lower():
        return 'POSITIVE'
    elif 'negative' in label.lower():
        return 'NEGATIVE'
    elif 'neutral' in label.lower():
        return 'NEUTRAL'
    
    # Return the original if it's already standardized
    elif label in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
        return label
    
    # Use score as fallback if provided
    elif score is not None:
        if score > 0.6:
            return 'POSITIVE'
        elif score < 0.4:
            return 'NEGATIVE'
        else:
            return 'NEUTRAL'
    
    # Default fallback
    else:
        return 'NEUTRAL'

# Sentiment analysis per sentence for detailed view
def analyze_sentiment_by_sentence(text, sentiment_pipeline):
    try:
        # Simple regex-based sentence tokenization instead of NLTK
        import re
        # Split on periods, exclamation marks, or question marks followed by space and uppercase letter
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        results = []
        
        for sentence in sentences:
            if len(sentence.strip()) > 10:  # Skip very short sentences
                # Truncate long sentences to fit model max length
                truncated_sentence = sentence[:512] if len(sentence) > 512 else sentence
                sentiment = sentiment_pipeline(truncated_sentence)[0]
                
                # Standardize the sentiment label
                standardized_label = standardize_sentiment_label(sentiment['label'], sentiment['score'])
                
                results.append({
                    'sentence': sentence,
                    'sentiment': standardized_label,
                    'score': float(sentiment['score'])  # Convert np.float32 to Python float
                })
        
        return results
    except Exception as e:
        print(f"Error in sentence-level analysis: {e}")
        return []

# Entity extraction
def extract_entities(text, ner_pipeline):
    try:
        # Split text into smaller chunks that fit within model's maximum sequence length (512 tokens)
        max_length = 450  # Slightly less than 512 to account for special tokens
        chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
        
        all_entities = []
        for chunk in chunks:
            chunk_entities = ner_pipeline(chunk)
            all_entities.extend(chunk_entities)
        
        # Group similar entities
        grouped_entities = {}
        for entity in all_entities:
            key = f"{entity['word']}_{entity['entity']}"
            if key not in grouped_entities:
                grouped_entities[key] = {
                    'word': entity['word'],
                    'entity_type': entity['entity'],
                    'count': 1,
                    'score': entity['score']
                }
            else:
                grouped_entities[key]['count'] += 1
                grouped_entities[key]['score'] = max(grouped_entities[key]['score'], entity['score'])
        
        return list(grouped_entities.values())
    except Exception as e:
        print(f"Error in entity extraction: {e}")
        return []

# Thêm hàm mới để tính toán cảm xúc tổng thể dựa trên phân tích từng câu
def calculate_overall_sentiment_from_sentences(sentence_sentiments):
    """
    Calculate overall sentiment based on sentence-level analysis
    
    Args:
        sentence_sentiments (list): List of dictionaries containing sentence sentiment data
        
    Returns:
        dict: Dictionary with overall sentiment label and score
    """
    if not sentence_sentiments:
        return {'label': 'NEUTRAL', 'score': 0.5}
    
    # Count sentiments
    sentiment_counts = {
        'POSITIVE': 0,
        'NEUTRAL': 0,
        'NEGATIVE': 0
    }
    
    # Sum up scores for each sentiment type
    sentiment_scores = {
        'POSITIVE': 0.0,
        'NEUTRAL': 0.0,
        'NEGATIVE': 0.0
    }
    
    # Process each sentiment
    for item in sentence_sentiments:
        label = item['sentiment']
        score = item['score']
        
        # Standardize the label if needed
        if label not in sentiment_counts:
            if 'positive' in label.lower():
                label = 'POSITIVE'
            elif 'negative' in label.lower():
                label = 'NEGATIVE'
            else:
                label = 'NEUTRAL'
        
        sentiment_counts[label] += 1
        sentiment_scores[label] += score
    
    # Calculate average score for each sentiment
    for label in sentiment_scores:
        if sentiment_counts[label] > 0:
            sentiment_scores[label] /= sentiment_counts[label]
    
    # Determine dominant sentiment
    dominant_sentiment = max(sentiment_counts.items(), key=lambda x: x[1])[0]
    
    # If there's a tie, use the one with higher average score
    if list(sentiment_counts.values()).count(max(sentiment_counts.values())) > 1:
        dominant_sentiment = max(sentiment_scores.items(), key=lambda x: x[1])[0]
    
    # Return the result
    return {
        'label': dominant_sentiment,
        'score': sentiment_scores[dominant_sentiment]
    } 