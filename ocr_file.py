import easyocr

def extract_text(image_path):
    reader = easyocr.Reader(['en'])
    results = reader.readtext(image_path)
    text = " ".join([res[1] for res in results])
    return text
