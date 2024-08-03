import unittest
import os
import cv2
import numpy as np
from models.segmentation_model import SegmentationModel

class TestSegmentationModel(unittest.TestCase):

	def setUp(self):
		# Initialize the model
		self.model = SegmentationModel()
		
		# Set the path to the existing image for testing
		self.test_img_path = "data/input_images/000000000307.jpg"


	def test_initialization(self):
		# Test if the model initializes correctly
		self.assertIsNotNone(self.model)
		self.assertIsNotNone(self.model.model)

	def test_predict(self):
		# Test the prediction functionality
		segmented_objects, object_metadata = self.model.predict(self.test_img_path)
		
		# Check if the output is as expected
		self.assertIsInstance(segmented_objects, list)
		self.assertIsInstance(object_metadata, list)
		self.assertGreater(len(segmented_objects), 0)
		self.assertGreater(len(object_metadata), 0)

		# Check if metadata contains expected keys
		for metadata in object_metadata:
			self.assertIn("object_id", metadata)
			self.assertIn("object_img_path", metadata)
			self.assertIn("obj_seg_bbox", metadata)
			self.assertIn("master_image", metadata)
			self.assertIn("master_id", metadata)

	def test_invalid_input(self):
		
        # Check if invalid input raises an error
		with self.assertRaises(ValueError):
			self.model.predict(None)

if __name__ == '__main__':
	unittest.main()