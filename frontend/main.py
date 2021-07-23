from requests.models import Response
import streamlit as st
import json
import requests
import io
import urllib
from PIL import Image

api_access = r'http://localhost:8000'

@st.cache(ttl=3600, max_entries=10)
def image_out(image):
    return Image.open(image)


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
        upload_uri = api_access+'/api/postImage/'
        response = requests.post(url = upload_uri, files = files)    
        st.write(response.json())


if __name__ == "__main__":
    st.set_page_config(
    initial_sidebar_state = "expanded",
    page_title="XORbians: Face Detector"
    )
    main()