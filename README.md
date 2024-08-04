# Internship Task

This repository contains the code and resources for an internship task focused on image processing and text extraction using machine learning models.

## Folder Structure


The folder structure of the repository is as follows:
    
    ├── data
    │   ├── input_images                # input images
    │   ├── output                      # output results
    │   └── segmented_objects           # segmented objects
    ├── internship_task.egg-info        # Package information
    ├── model_assets                    # Pre-trained model assets
    │   ├── yolov8n.pt                 
    │   └── yolov8s-seg.pt             
    ├── models                          # Model definitions
    │   ├── identification_model.py     
    │   ├── segmentation_model.py       
    │   ├── summarization_model.py      
    │   └── text_extraction_model.py    
    ├── README.md                       # Project documentation
    ├── requirements.txt                # Python dependencies   
    ├── setup.py                        # Setup script for the package
    ├── streamlit_app                   # Streamlit application
    │   ├── app.py                      # Main application file
    │   └── components                  # Components for Streamlit
    ├── tests                           # Unit tests
    │   ├── test_identification.py      
    │   ├── test_segmentation.py        
    │   ├── test_summarization_model.py  
    │   └── test_text_extraction.py     
    └── utils                           # Utility scripts
        ├── data_mapping.py             
        ├── postprocessing.py           
        ├── preprocessing.py            
        └── visualization.py            




## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/Lauel09/wasserstoff-AiInternTask.git
    
    cd wasserstoff-AiInternTask
    ```

2. Create a virtual environment and activate it:
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage


If you are running model files like:-
```bash
    python3 models/identification_model.py  # or any other model
```
Make sure to run from the root folder,i.e., wasserstoff-AiInternTask.

Not even the models, but also the tests and utils files should be run from the root folder, including the streamlit app as shown below.

### Running the Streamlit App

To run the Streamlit application, use the following command:
```sh
    streamlit run streamlit_app/app.py
```

Streamlit app hosted on Huggingface Spaces: [Wasserstoff Internship Task](https://huggingface.co/spaces/Lauel/wasserstoff-AiInternTask)