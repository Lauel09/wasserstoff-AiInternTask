
from models.segmentation_model import SegmentationModel
from models.identification_model import IdentificationModel
from models.text_extraction_model import TextExtractionModel
from g4f.client import Client


"""
    This is our SummarizationModel. It takes in the metadata of the segmented objects,
    the descriptions generated for the objects and the text extracted from the image
    and summarizes the nature and attribute of each object.

    As you can see, I have also given it a large amount of insturctions to follow, this ensures
    that the model is able to understand my task and can generate a consistent output regardless
    of the metadata it is given.

    I have been able to use the GPT-3.5-turbo model using the python package gpt4free which 
    can be installed using: pip install g4f[all]

    This allows me to access the model for free without any API keys or any other requirements.
    Since, I can't hardcode the API key either I have used this package to access the model.

"""
class SummarizationModel:
    
    content_1 = r""" 
        You are a summarization model. Here is the context:
        Semantic segmentation has been applied to an image and segmented objects have
        been extracted from them with their IDs. Those object have been identified by
        an yolov8 model and their descriptions have been generated. EasyOCR has been applied
        and the texts, if any, have been extracted.
        Your task:-
        1.  Your task is to summarize the nature and attribute of each object
        2.  You summary should be divided into information clean bullet points.
        3.  You won't use any markdown.
        4.  Don't ask any questions after your reply.
        5.  If you don't recieve any data on the detected objected and their metadata then
            say no objects detected.
        6. If you don't get any text_result, say No text detected.
        7. Don't show path of master image and object image.
        Here is the info:-
    """
        
    def __init__(self) -> None:
        pass

    def summarize(self, obj_metadata, desc, txt_results):
        content_2 = self.content_1 + f"""\nobj_metadata: {obj_metadata} \n desc: {desc} \n txt_results: {txt_results}
        Summarize the nature and attribute of each object."""

        client = Client()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": content_2}],
            )
        return response.choices[0].message.content



if __name__ == "__main__":

    # My tiny test of SummarizationModel
    img_path = "data/input_images/000000000025.jpg"

    seg_model = SegmentationModel()
    segmented_objects, obj_metadata = seg_model.predict(img_path)

    id_model = IdentificationModel()
    
    descriptions = id_model.generate_descriptions(img_path)

    txt_ext_model = TextExtractionModel()
    txt_processed_image, results = txt_ext_model.extract_text(img_path)

    model = SummarizationModel()

    response = model.summarize(obj_metadata, descriptions, results)

    print(response)