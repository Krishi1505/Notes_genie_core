import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import streamlit as st
import os

@st.cache_resource
def load_trocr_model():
    """Load TrOCR processor and model (cached)"""
    try:
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        model.eval()
        return processor, model
    except Exception as e:
        st.error(f"Failed to load TrOCR model: {e}")
        raise

class OCRProcessor:
    def __init__(self):
        self.processor, self.model = load_trocr_model()

    def process_image(self, image_input):
        """
        Extract text from a handwritten image

        Args:
            image_input (str or PIL.Image): Path to image file or PIL Image object

        Returns:
            str: Extracted text
        """
        try:
            # Load image from path or use directly
            if isinstance(image_input, str):
                if not os.path.exists(image_input):
                    return f"Image path not found: {image_input}"
                image = Image.open(image_input).convert("RGB")
            elif isinstance(image_input, Image.Image):
                image = image_input.convert("RGB")
            else:
                return "Invalid image input type"

            # Preprocess and generate text
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values)
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            return generated_text.strip()

        except Exception as e:
            return f"Error during OCR processing: {e}"

    def process_multiple_images(self, image_inputs):
        """
        Process multiple images and return combined text

        Args:
            image_inputs (list): List of image paths or PIL Image objects

        Returns:
            str: Combined extracted text
        """
        all_text = []
        for img in image_inputs:
            text = self.process_image(img)
            all_text.append(text)

        return "\n\n".join(all_text)