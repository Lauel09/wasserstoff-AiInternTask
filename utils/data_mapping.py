import os
from models.segmentation_model import SegmentationModel
from models.identification_model import IdentificationModel
from models.text_extraction_model import TextExtractionModel
from models.summarization_model import SummarizationModel
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import cv2
from matplotlib.gridspec import GridSpec
from tabulate import tabulate
from matplotlib.font_manager import FontProperties
from utils.visualization import shorten_path

class DataMapping:
    def __init__(self) -> None:
        pass

    def mapping(self, seg_mode_data: tuple, id_model_data, txt_ext_model_data: str, summ_model_data: str):
        """
        SegmentationModel data -> Tuple of segmented objects and object metadata
        IdentificationModel data -> Descriptions of objects
        TextExtractionModel data -> Extracted text from image
        """
        final_data = []
        segmented_objects, obj_metadata = seg_mode_data
        final_dict = dict()

        desc, id_model_image_path = id_model_data

        master_id = None

        # While all are generated according to segmented objects the TextExtractionModel using EasyOCR
        # generates text all at once
        for (segmented_image, seg_obj_metdata, desc) in zip(segmented_objects, obj_metadata, desc):
            object_id = seg_obj_metdata['object_id']
            object_img_path = seg_obj_metdata['object_img_path']
            object_seg_bbox = seg_obj_metdata['obj_seg_bbox']
            master_image = seg_obj_metdata['master_image']

            object_name = desc['object_class']
            confidence = desc['conf']
            obj_id_bbox = desc['obj_id_bbox']
            master_id = seg_obj_metdata['master_id']

            new_entry = {
                "obj_id": object_id,
                "obj_img_path": object_img_path,
                "obj_seg_bbox": object_seg_bbox,
                "master_image": master_image,
                "obj_name": object_name,
                "confidence": confidence,
                "obj_id_bbox": obj_id_bbox,
            }
            if master_id in final_dict:
                final_dict[master_id]['entries'].append(new_entry)
            else:
                final_dict[master_id] = {
                    "entries": [new_entry],
                    "text": txt_ext_model_data,
                    "summary": summ_model_data,
                    "id_model_image_path": id_model_image_path
                }
        return final_dict

    def visualize(self, final_dict, img_path):
        """
        Visualize the original image with annotations and a table summarizing all data.
        """
        # Load the original image
        id_model_image_path = list(final_dict.values())[0]['id_model_image_path']
        annotated_image = cv2.imread(id_model_image_path)
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        # Create a DataFrame for the table
        data = []
        for master_id, master_data in final_dict.items():
            for entry in master_data['entries']:
                display_entry = entry.copy()
                display_entry['obj_img_path'] = shorten_path(display_entry['obj_img_path'])
                display_entry['master_image'] = shorten_path(display_entry['master_image'])
                data.append(display_entry)
        df = pd.DataFrame(data)

        # Create a figure with a grid layout
        fig = plt.figure(figsize=(20, 15))
        gs = GridSpec(3, 2, width_ratios=[3, 1], height_ratios=[2, 1, 1])

        # Display the pre-annotated image
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.imshow(annotated_image)
        ax0.set_title("Annotated Image")

        # Hide axes for the image
        ax0.axis('off')

        # Display the table
        ax1 = fig.add_subplot(gs[1, :])
        ax1.axis('off')
        table_str = tabulate(df, headers='keys', tablefmt='grid')
        ax1.text(0.5, 0.5, table_str, horizontalalignment='center', verticalalignment='center', fontsize=10, family='monospace')

        # Display the summary text
        ax2 = fig.add_subplot(gs[2, :])
        ax2.axis('off')
        summary_text = final_dict[list(final_dict.keys())[0]]['summary']
        font_prop = FontProperties(fname='streamlit_app/components/fonts/DejaVuSans.ttf')
        ax2.text(0.5, 0.5, summary_text, horizontalalignment='center', verticalalignment='center', fontsize=12, wrap=True, fontproperties=font_prop)

        # Show the final visual output
        plt.tight_layout()
        plt.show()

        # Print the shortened image path
        print("Image Path:", shorten_path(img_path))


if __name__ == "__main__":

    # TEST THE DATA MAPPING

    img_path = "data/input_images/000000000025.jpg"

    seg_model = SegmentationModel()
    segmented_objects, obj_metadata = seg_model.predict(img_path)

    id_model = IdentificationModel()
    desc, id_model_image_path = id_model.generate_descriptions(img_path)

    txt_ext_model = TextExtractionModel()
    text = txt_ext_model.extract_text(img_path)

    summ_model = SummarizationModel()
    summ_model_data = summ_model.summarize(obj_metadata, desc, text)

    data_mapping = DataMapping()
    final_dict = data_mapping.mapping((segmented_objects, obj_metadata), (desc, id_model_image_path), text, summ_model_data)

    # Visualize the final output
    data_mapping.visualize(final_dict, img_path)