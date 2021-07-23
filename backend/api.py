from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
import time
from pydantic import BaseModel
from ML import ML
import numpy as np
import cv2
from PIL import Image
import io
import json
app = FastAPI()
@app.post("/api/postImage/")
async def uploadImage(imageFile: UploadFile = File(...)):
    if(imageFile.content_type == 'image/jpg' or imageFile.content_type == 'image/jpeg' or imageFile.content_type == 'image/png'):
        contents = await imageFile.read()
        pil_image = Image.open(io.BytesIO(contents))
    else:
        pil_image = Image.open(imageFile.file)
    try:
        if pil_image.mode == 'RGBA':
            pil_image = pil_image.convert('RGB')
        output = ML(pil_image)
        return output
    except Exception as e:
        print(e)
        return HTTPException(status_code = 400, detail="Upload a valid image")