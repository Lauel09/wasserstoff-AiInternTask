import unittest
from models.identification_model import IdentificationModel
import os

class TestIdentificationModel(unittest.TestCase):

	def setUp(self):


		self.model = IdentificationModel()

		# sample image
		self.sample_image_path = "test.jpg"
    

	def test_generate_descriptions(self):
		# Check if the sample image exists
		self.assertTrue(os.path.exists(self.sample_image_path), "Sample image does not exist")

		# Generate descriptions using the model
		descriptions, id_model_image_path = self.model.generate_descriptions(self.sample_image_path)

		# Verify that the descriptions are not empty
		self.assertIsNotNone(descriptions, "Descriptions should not be None")
		self.assertIsInstance(descriptions, list, "Descriptions should be a list")
		

        # I will not assert this, because in certain cases there are no objects detected
		# and the descriptions list will be empty
        # self.assertGreater(len(descriptions), 0, "Descriptions list should not be empty")



		# Verify the structure of the descriptions
		for desc in descriptions:
			self.assertIsInstance(desc, dict, "Each description should be a dictionary")
			# obj_id_bbox are the bboxes calculated by Identitifaion model
			self.assertIn("obj_id_bbox", desc, "Description should contain 'obj_id_bbox'")
            
            # conf is the confidence of the object detection
			self.assertIn("conf", desc, "Description should contain 'conf'")

			# assert that id_model_image_path is a path which exists
			self.assertTrue(os.path.exists(id_model_image_path), "Image path should exist")

            # object_class is the class of the object which is saved as a str
			self.assertIn("object_class", desc, "Description should contain 'object_class'")
			
if __name__ == '__main__':
	unittest.main()