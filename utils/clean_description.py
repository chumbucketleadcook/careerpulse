import re
import pandas as pd
from bs4 import BeautifulSoup

def clean_description(html: str) -> str:
    if not html:
        return ""
    
    # 1. Strip HTML tags
    text = BeautifulSoup(html, "html.parser").get_text(separator=" ")
    
    # 2. Lowercase
    text = text.lower()
    
    # 3. Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    
    # 4. Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)
    
    # 5. Remove special characters and digits, keep only letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)
    
    # 6. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text