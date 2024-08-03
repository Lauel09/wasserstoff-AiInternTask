from ultralytics import YOLO
import os
import cv2
import numpy as np
import tempfile
"""
    This is our SegmentationModel. Simply put, it applies Image Segmentation using
    ultralytics' pretrained model yolov8-s where 's' stands for small.
"""
class SegmentationModel:

    
    # Model using -> yolov8 pretrained for segmentation
    model_name = "yolov8s-seg.pt"
    script_dir = os.path.dirname(__file__) 

    # Directory to save the metadata json file
    output_dir = tempfile.TemporaryDirectory()
    model_path = os.path.abspath("model_assets/" + model_name)

    def __init__(self):
        print(f"[INFO] Initializing Segmentation model from {self.model_path}")
        self.model = YOLO(self.model_path)
    

    def predict(self, img_path):
        img = None

        try:
            img = cv2.imread(os.path.abspath(img_path))
        
        except Exception as e:
            # if can't load image, raise an error
            raise ValueError(f"Unable to load image file due to error: {e}")
           
        results = self.model.predict(img, conf = 0.30)
        
        master_id = os.path.basename(img_path).split('.')[0]
        

        object_metadata = []
        segmented_objects = list() # List to store segmented objects
        # Show results using cv2
        for result in results:
            masks = result.masks
            if masks is not None:

                for idx, mask in enumerate(masks.xy):
                    points = np.int32([mask])

                    # extract these 4 variables from the points
                    # we will be using these to extract cropped segment objects
                    x, y, w, h = cv2.boundingRect(points)

                    cropped_object = img[y:y+h, x:x+w]

                    # Create a binary mask for the cropped object
                    mask_img = np.zeros_like(cropped_object)
                    mask_points = mask - [x, y]  # Adjust mask points to cropped object coordinates
                    cv2.fillPoly(mask_img, [np.int32(mask_points)], (255,255 ,0 ))
                    
                    # Combine the original cropped object with the colored mask
                    shaded_object = cv2.addWeighted(cropped_object, 0.7, mask_img, 0.3, 0)


                    # object_id is an comibination of master_id and idx
                    # where master_id is the name of the original image without the extension
                    object_id = f"{master_id}_obj_{idx}"
                    object_img_path = self.output_dir.name + f"/{object_id}.jpg"

                    # Save the shaded object to the output directory
                    # These shaded objects are those segmented objects
                    # and if wanted can be anytime loaded and used
                    cv2.imwrite(object_img_path, shaded_object)
                    print(f"[INFO] Saved shaded object to {object_img_path}")
                    segmented_objects.append(shaded_object)


                    # this is very important as this variable basically carries all the metadata
                    # we extracted using SegmentationModel
                    object_metadata.append({
                        "object_id": object_id,
                        "object_img_path": object_img_path,
                        "obj_seg_bbox": [x, y, w, h],
                        "master_image": os.path.abspath(img_path),
                        "master_id": master_id
                    })
        
        # return segmented_objects and object_metadata
        # while the task hasn't excplicitly asked for shaded_figures, I still kept them
        # say if I had to show or utilize them later
        return (segmented_objects, object_metadata)
