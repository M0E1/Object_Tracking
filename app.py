import cv2
import streamlit as st
import numpy as np
import tempfile
import time

# Page config
st.set_page_config(page_title="Object Tracking App", layout="centered")
st.title('📹 Object Tracking App')

# Sidebar: About
with st.sidebar:
    st.header("📘 About")
    st.markdown("""
    This app tracks moving objects in uploaded videos using background subtraction,  
    noise reduction, and contour detection.

    **Developed by:** *Mohamed Mostafa*
    """)
    st.divider()

    st.subheader("⚙️ Settings")

    fps = st.slider("🎚️ Playback FPS", min_value=1, max_value=60, value=30)
    delay = 1 / fps

    min_area = st.slider("🔍 Minimum Object Area", min_value=50, max_value=1000, value=100, step=50)

# ثابت: لون الـ Bounding Box = أخضر
box_color = (0, 255, 0)  # Green

# Upload video
upload_vid = st.file_uploader('📁 Please upload a video file', type=['mp4', 'avi'])

if upload_vid is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(upload_vid.read())

    capture = cv2.VideoCapture(tfile.name)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    back_subtractor = cv2.createBackgroundSubtractorMOG2(varThreshold=50, detectShadows=False)

    stframe1 = st.empty()
    stframe2 = st.empty()
    progress = st.progress(0)
    object_counter = st.empty()

    current_frame = 0

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 360))

        fg_mask = back_subtractor.apply(frame)
        img_blur = cv2.GaussianBlur(fg_mask, (11, 11), 0)
        _, img_thresh = cv2.threshold(img_blur, 200, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        img_morph = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)
        img_morph = cv2.morphologyEx(img_morph, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(img_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        object_count = 0

        for cnt in contours:
            if cv2.contourArea(cnt) > min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
                object_count += 1

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fg_mask_colored = cv2.applyColorMap(fg_mask, cv2.COLORMAP_BONE)

        stframe1.image(frame_rgb, caption='🎥 Original Feed', channels='RGB')
        stframe2.image(fg_mask_colored, caption='🕵️‍♂️ Foreground Mask', channels='BGR')
        object_counter.markdown(f"🔴 **Objects detected:** `{object_count}`")

        current_frame += 1
        progress.progress(min(current_frame / total_frames, 1.0))

        time.sleep(delay)

    capture.release()

else:
    st.info("📌 Please upload a video file to start.")
