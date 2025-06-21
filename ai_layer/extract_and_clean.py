import pytesseract
import fitz 
import os
import tempfile
import re
from collections import Counter

def typed_ocr_extract(file_path: str) -> str:
    zoom = 4  # Higher zoom improves OCR accuracy (4x ~ 300 DPI)
    mat = fitz.Matrix(zoom, zoom)

    text_data = ''

    with tempfile.TemporaryDirectory() as temp_dir:
        doc = fitz.open(file_path)

        for i, page in enumerate(doc):
            try:
                # Render page to high-res image
                pix = page.get_pixmap(matrix=mat)
                img_path = os.path.join(temp_dir, f"page_{i+1}.png")
                pix.save(img_path)

                # OCR image to text
                text = pytesseract.image_to_string(img_path)
                text_data += text + '\n'

            except Exception as e:
                print(f"[ERROR] Page {i+1} failed: {e}")

        doc.close()

    return text_data.strip()

def normalize_line(line):
    return re.sub(r'\\d+', '', re.sub(r'\\W+', '', line.strip().lower()))

def remove_repeated_lines(lines, min_repeats=3, max_line_length=120):
    normalized = [normalize_line(line) for line in lines if len(line.strip()) < max_line_length]
    counts = Counter(normalized)
    repeated = {line for line, count in counts.items() if count >= min_repeats and line}
    return [line for line in lines if normalize_line(line) not in repeated]

def cleaner(input_string: str) -> str:
    lines = input_string.splitlines()
    lines = remove_repeated_lines(lines)
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped or len(stripped) < 3 or not re.search(r'[a-zA-Z0-9]', stripped): #blannk
            continue 
        if re.match(r'^[^a-zA-Z0-9\s]{4,}$', stripped) or re.match(r'^[\s\-_=~`\*#]{4,}$', stripped): #excess symbol and whhitespaces
            continue
        if re.match(r'^[\d\W]{3,}$', stripped): #page numbers
            continue
        if re.match(r'^(figure|fig\.|table|tab\.)\b', stripped, re.IGNORECASE): #captions
            continue
        stripped = re.sub(r'\s+', ' ', stripped)
        cleaned_lines.append(stripped)
    cleaned = []
    paragraph = ''
    for line in cleaned_lines:
        if re.match(r'^(>|-|\*|\d+\.|\d+\))', line): #bullet points
            if paragraph:
                cleaned.append(paragraph)
                paragraph = ''
            cleaned.append(line)
        else:
            if paragraph:
                if not re.search(r'[.!?;:]$', paragraph): #no punc
                    paragraph += ' ' + line
                else:
                    cleaned.append(paragraph)
                    paragraph = line
            else:
                paragraph = line
    if paragraph:
        cleaned.append(paragraph)
    return '\n'.join(cleaned)


def extract_and_clean(file_path:str):
    text = typed_ocr_extract(file_path)
    clean_text  = cleaner(text)
    return clean_text


    
