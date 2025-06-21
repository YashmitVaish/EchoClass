import pytesseract
import fitz 
import os
import tempfile
import re



def typed_ocr_extract(file_path:str) -> str:
    zoom = 4
    mat = fitz.Matrix(zoom, zoom)

    with tempfile.TemporaryDirectory() as temp_dir:
        doc = fitz.open(file_path)
        text_data = ''
        for i in range(len(doc)):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=mat)
            img_path = os.path.join(temp_dir, f"image_{i+1}.png")
            pix.save(img_path)
            text = pytesseract.image_to_string(img_path)
            text_data += text + '\n'
        doc.close()
        return(text_data)

def text_cleaner(text_data:str) -> str :
    pass
