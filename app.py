import streamlit as st
from ocr_processor import OCRProcessor

# Initialize OCR
ocr = OCRProcessor()

# Title
st.title("📝 Handwritten Text Recognition with TrOCR")

# Image path input
image_path = st.text_input("Enter image path", "images/handwritten_note.jpg")

# Run OCR
if st.button("Extract Text"):
    result = ocr.process_image(image_path)
    st.image(image_path, caption="Input Image", use_column_width=True)
    st.subheader("Extracted Text:")
    st.write(result)