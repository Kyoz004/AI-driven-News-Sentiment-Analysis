import requests
import re
from bs4 import BeautifulSoup
from langdetect import detect

def extract_text_from_url(url):
    """
    Extract text content from a URL
    
    Args:
        url (str): The URL to extract text from
        
    Returns:
        dict: Dictionary containing title, content, and source
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Extract title
    title = soup.title.text if soup.title else "No title found"
    
    # Extract content - focusing on article text
    article_text = ""
    article_tags = soup.find_all(['article', 'div', 'section'], class_=re.compile('article|content|body|main'))
    
    if article_tags:
        for tag in article_tags:
            paragraphs = tag.find_all('p')
            for p in paragraphs:
                article_text += p.get_text() + "\n"
    else:
        # Fallback to all paragraphs
        paragraphs = soup.find_all("p")
        article_text = "\n".join(p.get_text() for p in paragraphs)
    
    # Clean the text
    article_text = re.sub(r'\s+', ' ', article_text).strip()
    
    # Try to identify source
    source = None
    meta_source = soup.find('meta', property='og:site_name')
    if meta_source:
        source = meta_source['content']
    else:
        source = url.split('/')[2]
    
    return {
        'title': title,
        'content': article_text,
        'source': source
    }

def detect_language(text):
    """
    Detect the language of a text
    
    Args:
        text (str): The text to detect language from
        
    Returns:
        str: The detected language code
    """
    try:
        return detect(text)
    except:
        return "unknown" 