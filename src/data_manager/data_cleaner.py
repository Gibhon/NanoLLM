import re
from pathlib import Path


def clean_data(path:Path):
    text = path.read_text(encoding='utf-8')

    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'[ \t\r\f\v]+', ' ', text)
    text = re.sub(r'\n{2,}', '\n', text)

    return list(text.strip())