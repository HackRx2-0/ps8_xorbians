from requests.models import Response
import streamlit as st
import json
import requests
import io
import urllib
import cv2
import numpy as np
from PIL import Image

api_access = r'http://localhost:8000'

@st.cache(ttl=3600, max_entries=10)
def image_out(image):
    return Image.open(image)

def load_image(img):
    im = Image.open(img)
    image = np.array(im)
    return image

def main():
    uploaded_file = st.file_uploader("Upload Files",type=['png','jpeg','jpg'])
    st.sidebar.markdown('''
    # XORbians Face Detector
    This app is made by team XORbians for submission in HackRx
    ''')
    if uploaded_file is not None:
        files = {
            "imageFile": uploaded_file.read()
        }
        img = load_image(uploaded_file)
        st.image(img, channels="RGB")
        upload_uri = api_access+'/api/postImage/'
        response = requests.post(url = upload_uri, files = files)
        res = response.json()
        print(res)
        res1 = res["output"][0]
        res2 = res["output"][1]
        res3 = res["output"][2]
        print(res1)
        if len(res1["issues"]) == 0:
            st.write("valid image with score of :",res2,res3)  
        else:
            st.write("Invalid image with score of",res1,res2,res3)


if __name__ == "__main__":
    st.set_page_config(
    initial_sidebar_state = "expanded",
    page_title="XORbians: Face Detector"
    )
    main()