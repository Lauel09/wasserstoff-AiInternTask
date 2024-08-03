import unittest
from unittest.mock import patch, MagicMock
from models.summarization_model import SummarizationModel

class TestSummarizationModel(unittest.TestCase):

	def setUp(self):
		self.model = SummarizationModel()

	@patch('models.summarization_model.Client')
	def test_summarize(self, MockClient):
		
		# Mock the response from the Client
		mock_response = MagicMock()
		mock_response.choices[0].message.content = "Mocked summary response"
		MockClient().chat.completions.create.return_value = mock_response

		# Define test inputs
		obj_metadata = [{"object_id": 1, "object_img_path": "path/to/object1.jpg", "obj_seg_bbox": [0, 0, 10, 10], "master_image": "path/to/master.jpg", "master_id": "0001"}]
		descriptions = ["Object 1 is a car."]
		txt_results = ["Text on the car: 'For Sale'"]

		# Call the summarize method
		summary = self.model.summarize(obj_metadata, descriptions, txt_results)

		# Check if the summary is as expected
		self.assertEqual(summary, "Mocked summary response")

	@patch('models.summarization_model.Client')
	def test_summarize_no_objects(self, MockClient):
		# Mock the response from the Client
		mock_response = MagicMock()
		mock_response.choices[0].message.content = "No objects detected"
		MockClient().chat.completions.create.return_value = mock_response

		# Define test inputs with no objects
		obj_metadata = []
		descriptions = []
		txt_results = []

		# Call the summarize method
		summary = self.model.summarize(obj_metadata, descriptions, txt_results)

		# Check if the summary is as expected
		self.assertEqual(summary, "No objects detected")

	@patch('models.summarization_model.Client')
	def test_summarize_no_text(self, MockClient):
		# Mock the response from the Client
		mock_response = MagicMock()
		mock_response.choices[0].message.content = "No text detected"
		MockClient().chat.completions.create.return_value = mock_response

		# Define test inputs with no text
		obj_metadata = [{"object_id": 1, "object_img_path": "path/to/object1.jpg", "obj_seg_bbox": [0, 0, 10, 10], "master_image": "path/to/master.jpg", "master_id": "0001"}]
		descriptions = ["Object 1 is a car."]
		txt_results = []

		# Call the summarize method
		summary = self.model.summarize(obj_metadata, descriptions, txt_results)

		# Check if the summary is as expected
		self.assertEqual(summary, "No text detected")

	@patch('models.summarization_model.Client')
	def test_summarize_no_text(self, MockClient):
		# Mock the response from the Client
		mock_response = MagicMock()
		mock_response.choices[0].message.content = "No text detected"
		MockClient().chat.completions.create.return_value = mock_response

		# inputs with no text
		obj_metadata = [{"object_id": 1, "object_img_path": "path/to/object1.jpg", "obj_seg_bbox": [0, 0, 10, 10], "master_image": "path/to/master.jpg", "master_id": "0001"}]
		descriptions = ["Object 1 is a car."]
		txt_results = []

		# Call the summarize method
		summary = self.model.summarize(obj_metadata, descriptions, txt_results)

		self.assertEqual(summary, "No text detected")

if __name__ == '__main__':
	unittest.main()