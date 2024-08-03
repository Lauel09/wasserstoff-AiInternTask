import streamlit as st
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.font_manager import FontProperties
from tabulate import tabulate
import os

from models.segmentation_model import SegmentationModel
from models.identification_model import IdentificationModel
from models.text_extraction_model import TextExtractionModel
from models.summarization_model import SummarizationModel
from utils.data_mapping import DataMapping
from utils.visualization import shorten_path

def visualize(final_dict, img_path):
    """
    Visualize the original image with annotations and a table summarizing all data.
    """
    # Load the original image
    id_model_image_path = list(final_dict.values())[0]['id_model_image_path']
    annotated_image = cv2.imread(id_model_image_path)
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    # this is the dataframe I am going to use to display the data
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

    # Display the original image with annotations
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(annotated_image)
    ax0.set_title("Annotated Image")

    # Hide axes for the image
    ax0.axis('off')

    # Display the table
    ax1 = fig.add_subplot(gs[1, :])
    ax1.axis('off')
    if not df.empty:
        table_str = tabulate(df, headers='keys', tablefmt='grid')
        ax1.text(0.5, 0.5, table_str, horizontalalignment='center', verticalalignment='center', fontsize=10, family='monospace')
    else:
        ax1.text(0.5, 0.5, "No data available.", horizontalalignment='center', verticalalignment='center', fontsize=10, family='monospace')

    # Display the summary text
    ax2 = fig.add_subplot(gs[2, :])
    ax2.axis('off')
    if final_dict and 'summary' in final_dict.get(list(final_dict.keys())[0], {}):
        summary_text = final_dict[list(final_dict.keys())[0]]['summary']
    else:
        summary_text = "Summary not available."
    font_path = os.path.join(os.path.dirname(__file__),'components', 'fonts', 'DejaVuSans.ttf')
    font_prop = FontProperties(fname=font_path)
    ax2.text(0.5, 0.5, summary_text, horizontalalignment='center', verticalalignment='center', fontsize=12, wrap=True, fontproperties=font_prop)

    # Show the final visual output
    plt.tight_layout()
    return fig

# Streamlit UI
st.title("Pipeline Testing UI")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    img_path = "temp.jpg"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display the uploaded image
    st.image(img_path, caption='Uploaded Image', use_column_width=True)
    st.write("")

    # Now we will load each models one by one and test them on the uploaded image
    seg_model = SegmentationModel()
    segmented_objects, obj_metadata = seg_model.predict(img_path)

    st.write("Object Metadata:", obj_metadata)

    id_model = IdentificationModel()
    desc = id_model.generate_descriptions(img_path)
    st.write("Descriptions:", desc)

    txt_ext_model = TextExtractionModel()
    text = txt_ext_model.extract_text(img_path)
    st.write("Extracted Text:", text)

    summ_model = SummarizationModel()
    summ_model_data = summ_model.summarize(obj_metadata, desc, text)
    st.write("Summarization Model Data:", summ_model_data)

    data_mapping = DataMapping()
    final_dict = data_mapping.mapping((segmented_objects, obj_metadata), desc, text, summ_model_data)
    st.write("Final Dictionary:", final_dict)

    # Visualize the final output
    if final_dict:
        fig = visualize(final_dict, img_path)
        st.pyplot(fig)
    else:
        st.write("No detections were made.")