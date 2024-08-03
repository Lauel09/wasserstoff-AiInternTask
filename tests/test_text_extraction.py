import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from models.text_extraction_model import TextExtractionModel

class TestTextExtractionModel(unittest.TestCase):

	@patch('models.text_extraction_model.easyocr.Reader')
	def setUp(self, MockReader):
		self.model = TextExtractionModel()
		self.mock_reader = MockReader.return_value

	def test_initialization(self):
		self.assertIsInstance(self.model.reader, MagicMock)

	@patch('models.text_extraction_model.cv2.imread')
	@patch('models.text_extraction_model.cv2.resize')
	def test_preprocess_image(self, mock_resize, mock_imread):
		# Mocking imread to return a dummy image
		dummy_img = np.zeros((100, 100), dtype=np.uint8)
		mock_imread.return_value = dummy_img
		mock_resize.return_value = np.zeros((600, 800), dtype=np.uint8)

		img_path = "dummy_path.jpg"
		processed_img = self.model.preprocess_image(img_path)
		self.assertEqual(processed_img.shape, (600, 800))

		# Test for invalid image path
		mock_imread.return_value = None
		with self.assertRaises(ValueError):
			self.model.preprocess_image("invalid_path.jpg")

	@patch('models.text_extraction_model.TextExtractionModel.preprocess_image')
	def test_extract_text(self, mock_preprocess_image):
		# Mocking preprocess_image to return a dummy image
		dummy_img = np.zeros((600, 800), dtype=np.uint8)
		mock_preprocess_image.return_value = dummy_img

		# Mocking the reader's readtext method to see if it returns the expected text
		self.mock_reader.readtext.return_value = [("Hello",), ("World",)]

		img_path = "dummy_path.jpg"
		result = self.model.extract_text(img_path)
		self.assertEqual(result, "Hello World")

		# Test for no text found
		self.mock_reader.readtext.return_value = []
		result = self.model.extract_text(img_path)
		self.assertIsNone(result)

if __name__ == '__main__':
	unittest.main()