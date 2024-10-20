import streamlit as st
import os
from main import AIaudioGen
import shutil


def save_video(file, file_name):
    """ Save uploaded video to a temporary file """
    try:
        if os.path.exists('Temp'):
            shutil.rmtree('Temp')
    except PermissionError:
        pass
    
    os.makedirs('Temp', exist_ok=True)
    file_path = f'Temp/{file_name}'
    with open(file_path, 'wb') as f:
        f.write(file.getbuffer())
    return file_path

# Streamlit frontend layout
st.title("AutoSync: Fix grammatical mistakes in your video")

# Input: Video file upload
uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
if uploaded_video is not None:
    video_file_name = uploaded_video.name
    video_path = save_video(uploaded_video, video_file_name)
    # Display original video
    st.video(video_path, format="video/mp4")

    # Display gender selection
    gender = st.selectbox("Select gender for voice synthesis:", ["Male", "Female"])
    generator = AIaudioGen(video_path, 'creds.json', gender)

    # Process the video 
    if st.button("Process Video"):
        with st.spinner('Processing video...'):
            original_transcript, output_transcript = generator.run()

            st.subheader("Original Transcript")
            st.text(original_transcript)

            # Display output video
            st.video('Output/output.mp4', format="video/mp4")

            # Display output transcript
            st.subheader("Updated Transcript")
            st.text(output_transcript)
    