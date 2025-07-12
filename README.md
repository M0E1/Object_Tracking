# 📹 Real-Time Object Tracking Web App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-app-url.com) <!--- Placeholder: Replace with your deployed app URL -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) <!--- Placeholder -->

A user-friendly web application built with Streamlit and OpenCV for real-time object tracking in video files. This tool leverages background subtraction and contour detection to identify and highlight moving objects, with several customization options available through a clean user interface.

## ✨ Features

-   **Video Upload**: Easily upload your video files (`.mp4`, `.avi`).
-   **Background Subtraction**: Uses `cv2.createBackgroundSubtractorMOG2` to isolate moving foreground objects.
-   **Noise Reduction**: Applies Gaussian blur and morphological transformations to clean up the foreground mask and improve detection accuracy.
-   **Customizable Detection**:
    -   Adjust the minimum area for an object to be detected, filtering out small noise.
    -   Select from a variety of colors for the bounding boxes.
-   **Playback Controls**:
    -   Adjustable playback speed (FPS).
    -   Start, Pause, and Reset the video processing.
-   **Live Feedback**:
    -   Displays the original video feed with detected objects highlighted.
    -   Shows the real-time foreground mask.
    -   Provides a live count of detected objects.
-   **Data Visualization**: Generates a plot of the object count over time, providing insights into the video's activity.

## 📸 App Screenshot 

<img width="2559" height="1303" alt="Screenshot 2025-07-12 103809" src="https://github.com/user-attachments/assets/827e44cb-ad84-4453-9347-4a0fbf9927ee" />

<img width="728" height="818" alt="Screenshot 2025-07-12 103934" src="https://github.com/user-attachments/assets/63e3fadf-ad88-419e-bfae-de5af459bf6a" />



## 🛠️ Technologies Used

-   **Backend**: Python
-   **Computer Vision**: OpenCV
-   **Web Framework**: Streamlit
-   **Data Handling**: NumPy
-   **Plotting**: Matplotlib

## 🚀 Setup and Installation

Follow these steps to run the application on your local machine.

### Prerequisites

-   Python 3.8+
-   `pip` (Python package installer)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

### 2. Create and Activate a Virtual Environment (Recommended)

-   **Windows**:
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```
-   **macOS/Linux**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

### 3. Install Dependencies

Install all the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Run the Application

Launch the Streamlit app from your terminal. The `app.py` file is the most feature-complete version.

```bash
streamlit run app.py
```

Your web browser should automatically open to the application's URL (usually `http://localhost:8501`).

## 📖 How to Use

1.  **Upload a Video**: Click the "Browse files" button to upload a video file (`.mp4` or `.avi`).
2.  **Adjust Settings**: Use the sliders and dropdowns in the sidebar to configure the FPS, minimum object area, and bounding box color.
3.  **Control Playback**: Use the "▶️ Start", "⏸️ Pause", and "🔁 Reset" buttons in the sidebar to control the video analysis.
4.  **View Results**: Observe the original feed with tracking boxes, the foreground mask, the live object count, and the historical object count chart.

## 👤 Author

-   **Mohamed Mostafa** - *(Initial work)*
