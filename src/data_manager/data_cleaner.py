import re 
from pathlib import Path

def clean_data(path:Path):
    text = path.read_text(encoding='utf-8')

    # Remove Chars accept the ones in ASCII and blank space
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    # Normalize white line
    text = re.sub(r'[ \t\r\f\v]+', ' ', text)
    
    # Max 1 newline at once
    text = re.sub(r'\n{2,}', '\n', text)

    return text.strip().split()
