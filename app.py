import cv2
import streamlit as st
import numpy as np
import tempfile
import time
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Object Tracking App", layout="centered")
st.title('📹 Object Tracking App')

# Sidebar: About & Controls
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

    # New: Minimum object area slider
    min_area = st.slider("🔍 Minimum Object Area", min_value=50, max_value=1000, value=100, step=50)

    box_color_name = st.selectbox("🎨 Select bounding box color", options=[
        "Red", "Green", "Blue", "Yellow", "Cyan", "Magenta", "White"
    ])

    color_map = {
        "Red": (0, 0, 255),
        "Green": (0, 255, 0),
        "Blue": (255, 0, 0),
        "Yellow": (0, 255, 255),
        "Cyan": (255, 255, 0),
        "Magenta": (255, 0, 255),
        "White": (255, 255, 255)
    }

    box_color = color_map[box_color_name]

    # Control buttons
    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    if "reset_video" not in st.session_state:
        st.session_state.reset_video = False

    col1, col2, col3 = st.columns(3)
    if col1.button("▶️ Start"):
        st.session_state.is_running = True
    if col2.button("⏸️ Pause"):
        st.session_state.is_running = False
    if col3.button("🔁 Reset"):
        st.session_state.reset_video = True

# Upload video
upload_vid = st.file_uploader('📁 Please upload a video file', type=['mp4', 'avi'])

if upload_vid is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(upload_vid.read())

    capture = cv2.VideoCapture(tfile.name)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if st.session_state.reset_video:
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        st.session_state.reset_video = False

    back_subtractor = cv2.createBackgroundSubtractorMOG2(varThreshold=50, detectShadows=False)

    stframe1 = st.empty()
    stframe2 = st.empty()
    progress = st.progress(0)
    object_counter = st.empty()
    chart_placeholder = st.empty()

    current_frame = 0
    object_counts = []

    while capture.isOpened():
        if not st.session_state.is_running:
            time.sleep(0.1)
            continue

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

        object_counts.append(object_count)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fg_mask_colored = cv2.applyColorMap(fg_mask, cv2.COLORMAP_BONE)

        stframe1.image(frame_rgb, caption='🎥 Original Feed', channels='RGB')
        stframe2.image(fg_mask_colored, caption='🕵️‍♂️ Foreground Mask', channels='BGR')
        object_counter.markdown(f"🔴 **Objects detected:** `{object_count}`")

        # current_frame += 1
        # progress.progress(min(current_frame / total_frames, 1.0))
    if current_frame % 10 == 0:
        fig, ax = plt.subplots()
        ax.plot(object_counts, color='blue')
        ax.set_title("📊 Object Count over Time")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Detected Objects")
        chart_placeholder.pyplot(fig)

        time.sleep(delay)

    capture.release()
else:
    st.info("📌 Please upload a video file to start.")
