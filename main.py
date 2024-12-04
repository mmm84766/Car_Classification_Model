import numpy as np
import uvicorn
from PIL import Image
import pickle
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from starlette.responses import RedirectResponse
from io import BytesIO
import logging
import sys
from pydantic import BaseModel
import tf_keras as k3

IMG_SIZE = 224  # Image size as model required

# Logging file format
logging.basicConfig(filename='api_logs.log', 
                    filemode='a', 
                    level=logging.INFO,
                    format='%(asctime)s : %(levelname)s : %(message)s', 
                    datefmt='%m/%d/%Y %I:%M:%S %p')


# Expected output schema
class PredictionResponseDto(BaseModel):
    filename: str
    class_name: str
    confidence: float
    img_resolution: dict
    bbox: dict


def IoU(y_true, y_pred):
    """Function is required for loading the tensorflow model as it is used as a metric for the model"""
    pass


def load_the_pretrained_model():
    # Loading the tensorflow saved model
    model = k3.models.load_model(r'../model/mobile_net_combi_with_aug_1_10-acc_0.8687-IoU_0.8758',
                                 custom_objects={'IoU': IoU})

    # Loading the class label maps with integer class
    class_names = pickle.load(open(r'../model/class_names.pickle', "rb"))

    return model, class_names


model, class_map = load_the_pretrained_model()

app_desc = """
## Try this app by uploading any image with `predict/image`
"""

# Creating FastAPI instance
app = FastAPI(title='Car Detection API', description=app_desc)


# Redirecting the root to docs
@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")


# Query for predicting the image
@app.post("/predict/image/", response_model=PredictionResponseDto)
async def predict(file: UploadFile = File(...)):
    if file.content_type.startswith('image/') is False:
        logging.warning(f"File '{file.filename}' is not an image.")
        raise HTTPException(status_code=400, detail=f"File '{file.filename}' is not an image.")

    try:
        contents = await file.read()

        # Converting image to numpy array
        image = np.asarray(Image.open(BytesIO(contents)).convert('RGB'))

        h_i, w_i, _ = image.shape

        # Resizing the image to 224x224x3
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE]).numpy()
        image = np.expand_dims(image, axis=0)

        # Predicting the image with saved model
        bbox, class_info = model.predict(image)
        output_class_name = class_map[np.argmax(class_info)]
        class_confidence = class_info[0].max()

        # Reshaping the bounding box from 224x224 size to actual height x width of uploaded image
        x, y, w, h = bbox[0]
        height_factor = h_i / IMG_SIZE
        width_factor = w_i / IMG_SIZE

        x, w = x * width_factor, w * width_factor
        y, h = y * height_factor, h * height_factor
        bbox = {
            'x': int(x), 
            'y': int(y), 
            'w': int(w), 
            'h': int(h)
        }

        logging.info(f"'filename':{file.filename},'class_name':{output_class_name},'confidence':{class_confidence}")

        return {
            'filename': file.filename,
            'class_name': output_class_name,
            'confidence': class_confidence,
            'img_resolution': {'h': h_i, 'w': w_i},
            'bbox': bbox
        }

    except Exception as error:
        logging.exception(error)
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Run the FastAPI app on different ports
    # Port 8000 (default)
    uvicorn.run(app, host="localhost", port=8000)