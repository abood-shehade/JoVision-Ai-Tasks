from PIL import Image
import pytesseract
import sys


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_from_image(image_path):
    try:
        # Open an image file
        with Image.open(image_path) as img:
            # Use pytesseract to do OCR on the image
            text = pytesseract.image_to_string(img)
            return text
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_text.py <image_path>")
    else:
        image_path = sys.argv[1]
        extracted_text = extract_text_from_image(image_path)
        print("Extracted Text:")
        print(extracted_text)