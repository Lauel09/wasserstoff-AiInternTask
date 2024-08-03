import easyocr
import cv2
import numpy as np


class TextExtractionModel:
    def __init__(self):
        print("[INFO] Initializing TextExtractor model")
        self.reader = easyocr.Reader(['en'])

    def preprocess_image(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image at {img_path}")
        # Resize the image
        img = cv2.resize(img, (800, 600))
        return img

    def extract_text(self, img_path):
        processed_image = self.preprocess_image(img_path)
        results = self.reader.readtext(processed_image)

        # usually the format of output is 
        # [(text, (x1, y1), (x2, y2), ...]
        # From this we only need the text
        extracted_texts = [text for (text, *_) in results]
        joined_results = " ".join(extracted_texts)
        return joined_results if joined_results else None


if __name__ == "__main__":
    model = TextExtractionModel()
    img_path = "test.jpg"  # Replace with the actual image path
    try:
        (img, joined_results) = model.extract_text(img_path)
        print(f"[INFO] Result of extraction: {joined_results}")
        
        if img is not None:
            cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Image", 800, 600)
            cv2.imshow("Image", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    except Exception as e:
        print(f"An error occurred: {e}")