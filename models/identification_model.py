import os
import requests
from ultralytics import YOLO
from PIL import Image
import tempfile
class IdentificationModel:
    model_folder = "model_assets/"
    model_name = "yolov8n.pt"

    # What if the model is not available?
    # We can download the model from this URL
    model_url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"  # Replace with the actual URL

    model_path = os.path.abspath(model_folder + model_name)
    model = None
    desc_path = "data/output/desc.json"
    
    temp_dir = tempfile.TemporaryDirectory()


    # Since this is an inference model which means model will predict
    # boxes, confidences and classes of objects in the image
    # Thus I feel there is a need to store the inferenced output image
    # for the visualization of the results, which I did use this in.
    
    output_img_path = temp_dir.name + "/id_model.jpg"


    # Whole process of model not existing
    if not os.path.exists(model_path):
        print(f"[INFO] Model {model_path} not found! Downloading it")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        response = requests.get(model_url)
        with open(model_path, 'wb') as f:
            f.write(response.content)
        print(f"[INFO] Model downloaded to {model_path}")

    print(f"[INFO] Initializing Object Identification model from {model_path}")
    model = YOLO(model_path)

    def __init__(self):
        self.results = None


    # Predict objects with a confidence threshold of 0.30 which means
    # if any object has a probability of belong to a class below than 30%
    # we will consider it as noise and ignore it.
    def identify_objects(self, image_path):
        self.results = self.model.predict(image_path, conf = 0.30)
        print("[INFO] Objects identified")


    # Descriptions generated are simply a list of dictionaries
    # where each dictionary contains the object class, confidence, and
    # the bounding box of the object in the image.
    def generate_descriptions(self, image_path):
        descriptions = []
        if self.results is None:
            print("[INFO] No objects identified yet! Running identification")
            self.identify_objects(image_path)

        for result in self.results:
            for obj in result.boxes:

                # Three variables I am going to use
                class_id = int(obj.cls)
                conf = float(obj.conf)
                bbox = obj.xyxy.tolist()

                # description of the object
                # obj_id_bbox is the bounding box of the object decided by
                # the IdentificationModel which is much more accurate than 
                # the obj_bbox decided by the SegmentationModel
                description = {
                    "object_class": self.model.names[class_id],
                    "conf": f"{conf:.2f}",
                    "obj_id_bbox": bbox
                }
                descriptions.append(description)

            # Save the output image of this model
            result.save(self.output_img_path)
            print(f"[INFO] Output image saved to {self.output_img_path}")
                
        self.desc = descriptions

        # return the descripions and the output image path
        return (descriptions, self.output_img_path)

if __name__ == "__main__":

    # Simple test of the IdentificationModel
    model = IdentificationModel()
    image_path = "data/input_images/000000000025.jpg" 
    results = model.identify_objects(image_path)


    # Generated descriptions by the model
    descriptions, output_image_path = model.generate_descriptions(image_path)



    # Print descriptions
    for desc in descriptions:
        print(desc)

   