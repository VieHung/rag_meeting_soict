import re
from typing import List


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    paragraphs = _split_by_separators(text)

    chunks = []
    current_text = ""

    for para in paragraphs:
        if len(current_text) + len(para) > chunk_size:
            if current_text:
                chunks.append(current_text.strip())
                current_text = current_text[-overlap:] + para if overlap > 0 else para
            else:
                chunks.append(para[:chunk_size])
                current_text = para[chunk_size:]
        else:
            current_text += " " + para if current_text else para

    if current_text:
        chunks.append(current_text.strip())

    return [c.strip() for c in chunks if c.strip()]


def _split_by_separators(text: str) -> List[str]:
    parts = re.split(r'\n\n+', text)
    result = []
    for part in parts:
        sentences = re.split(r'(?<=[.!?])\s+', part)
        result.extend(sentences)
    return [p.strip() for p in result if p.strip()]