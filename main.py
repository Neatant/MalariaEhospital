from fastapi import FastAPI, File, UploadFile
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import numpy as np

app = FastAPI()


# Load the pre-trained model
model = load_model('/content/model_vgg19.h5')


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save the uploaded image
    with open(file.filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Load and preprocess the image
    img = image.load_img(file.filename, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Make predictions
    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    if preds==0:
        preds="The Person is Infected With Malaria"
    else:
        preds="The Person is not Infected With Malaria"
    
    return {"prediction": preds}
