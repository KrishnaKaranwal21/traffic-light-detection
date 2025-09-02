import streamlit as st
import os
from pathlib import Path
import cv2
import numpy as np
from detector import process_video  # your real video detection function
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoProcessorBase

# ========= PAGE CONFIG =========
st.set_page_config(
    page_title="🚦 Traffic Light Detector",
    page_icon="🚦",
    layout="wide"
)

# ========= SESSION STATE =========
if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"

# ========= CUSTOM CSS =========
st.markdown("""
    <style>
        /* App background */
        .stApp {
            background: linear-gradient(135deg, #141E30, #243B55);
            color: white;
        }

        /* Headers */
        h1, h2, h3, h4 {
            color: #FFD700 !important;
            font-weight: bold;
        }

        /* Animated Title */
        .big-title {
            font-size: 3.8rem !important;
            font-weight: 900;
            text-align: center;
            margin-top: 0.3em;
            margin-bottom: 0.2em;
            background: linear-gradient(90deg, #FFD700, #FF4B4B, #00FF88, #FFD700);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: shine 5s linear infinite;
            background-size: 300%;
        }
        @keyframes shine {
            0% { background-position: 0% }
            100% { background-position: 300% }
        }

        /* Subtitle */
        .subtitle {
            font-size: 1.3rem !important;
            text-align: center;
            margin-bottom: 2em;
            color: #ddd;
        }

        /* Glassmorphism Card */
        .card {
            background: rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 1.5em;
            text-align: center;
            transition: 0.3s ease-in-out;
            cursor: pointer;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
            box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        }
        .card:hover {
            transform: translateY(-5px) scale(1.03);
            background: rgba(255,255,255,0.18);
            box-shadow: 0 12px 40px rgba(0,0,0,0.3);
        }

        /* Buttons */
        .stButton>button {
            background: linear-gradient(90deg, #ff4b4b, #ff9966);
            color: white;
            border-radius: 12px;
            padding: 0.6em 1.4em;
            font-weight: bold;
            transition: all 0.3s ease-in-out;
            border: none;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #ff3333, #ff7733);
            transform: scale(1.07);
            box-shadow: 0px 6px 20px rgba(0,0,0,0.4);
        }

        /* Sidebar */
        .css-1d391kg, .css-1v3fvcr {
            background: rgba(20, 20, 30, 0.9) !important;
            border-right: 2px solid rgba(255, 255, 255, 0.1);
        }
        .css-qri22k {
            color: #FFD700 !important;
            font-weight: bold !important;
        }

        /* Footer */
        .footer {
            margin-top: 3em;
            text-align: center;
            color: #bbb;
            font-size: 0.9rem;
        }
    </style>
""", unsafe_allow_html=True)

# ========= SIDEBAR NAV =========
st.sidebar.title("🔍 Navigation")

PAGES = [
    "🏠 Home",
    "🎬 Demo Video",
    "📥 Upload Video",
    "🖼️ Upload Image",
    "📷 Webcam Detection",
    "📘 Project Info"
]

# keep current page if it exists, else fallback to Home
current_index = PAGES.index(st.session_state.page) if st.session_state.page in PAGES else 0
choice = st.sidebar.radio("Go to", PAGES, index=current_index)
st.session_state.page = choice

# ========= VIDEO DETECTION =========
def run_detection(video_path, output_path="outputs/processed_output.mp4"):
    os.makedirs("outputs", exist_ok=True)
    ok = process_video(video_path, output_path)
    return output_path if ok else None

# ========= IMAGE DETECTION =========
def process_image(image_file, output_path="outputs/processed_image.jpg"):
    os.makedirs("outputs", exist_ok=True)
    import cv2, numpy as np

    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        return None

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_ranges = {
        "RED": [(0, 120, 70), (10, 255, 255)],
        "YELLOW": [(15, 120, 120), (35, 255, 255)],
        "GREEN": [(45, 100, 100), (75, 255, 255)]
    }

    for color, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)

            if area > 200:
                x, y, w, h = cv2.boundingRect(cnt)
                if color == "RED":
                    box_color = (0, 0, 255)
                elif color == "YELLOW":
                    box_color = (0, 255, 255)
                else:
                    box_color = (0, 255, 0)

                cv2.rectangle(image, (x, y), (x + w, y + h), box_color, 2)
                cv2.putText(image, f"{color} LIGHT", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

    cv2.imwrite(output_path, image)
    return output_path

# ========= Image Frame Detection =========
def process_image_frame(image):
    """Frame-level traffic light detection for webcam mode."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Color ranges
    color_ranges = {
        "RED": [(0, 120, 70), (10, 255, 255)],
        "YELLOW": [(15, 120, 120), (35, 255, 255)],
        "GREEN": [(45, 100, 100), (75, 255, 255)]
    }

    for color, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            if area > 100:  # smaller threshold for ROI
                x, y, w, h = cv2.boundingRect(cnt)
                if color == "RED":
                    box_color = (0, 0, 255)
                elif color == "YELLOW":
                    box_color = (0, 255, 255)
                else:
                    box_color = (0, 255, 0)

                cv2.rectangle(image, (x, y), (x + w, y + h), box_color, 2)
                cv2.putText(image, f"{color}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

    return image

# ========= REAL-TIME WEBCAM DETECTION =========
class TrafficLightProcessor(VideoProcessorBase):
    def __init__(self):
        self.color_ranges = {
            "RED": [(0, 120, 70), (10, 255, 255)],
            "YELLOW": [(15, 120, 120), (35, 255, 255)],
            "GREEN": [(45, 100, 100), (75, 255, 255)]
        }

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        for color, (lower, upper) in self.color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                cnt = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(cnt)
                if area > 200:
                    x, y, w, h = cv2.boundingRect(cnt)
                    if color == "RED":
                        box_color = (0, 0, 255)
                    elif color == "YELLOW":
                        box_color = (0, 255, 255)
                    else:
                        box_color = (0, 255, 0)

                    cv2.rectangle(img, (x, y), (x + w, y + h), box_color, 2)
                    cv2.putText(img, f"{color} LIGHT", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ========= PAGES =========
if st.session_state.page == "🏠 Home":
    # ========= HERO SECTION =========
    st.markdown("<div class='big-title'>🚦 Real-Time Traffic Light Detection</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>AI-powered system to detect and classify traffic lights from videos, images, and live webcam feed.</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ========= FEATURE CARDS =========
    st.markdown("### 🌟 Try It Out")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🎬 Demo Video"):
            st.session_state.page = "🎬 Demo Video"
            st.rerun()
        st.markdown("<div class='card'>See traffic light detection in action using our built-in demo video.</div>", unsafe_allow_html=True)

    with col2:
        if st.button("📥 Upload Video"):
            st.session_state.page = "📥 Upload Video"
            st.rerun()
        st.markdown("<div class='card'>Upload your own driving footage and watch detection in real time.</div>", unsafe_allow_html=True)

    with col3:
        if st.button("🖼️ Upload Image"):
            st.session_state.page = "🖼️ Upload Image"
            st.rerun()
        st.markdown("<div class='card'>Test detection on a single image for quick insights.</div>", unsafe_allow_html=True)

    with col4:
        if st.button("📷 Webcam Detection"):
            st.session_state.page = "📷 Webcam Detection"
            st.rerun()
        st.markdown("<div class='card'>Turn on your webcam and detect live traffic lights instantly.</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ========= ABOUT PROJECT =========
    st.subheader("ℹ️ About This Project")
    st.write("""
    This project demonstrates a **real-time traffic light detection system** built using **Python, OpenCV, and AI concepts**.  
    It can process videos, images, and live webcam streams to identify whether a traffic light is **Red, Yellow, or Green**.  

    Designed for learning and prototyping, it shows how **computer vision** can assist in:
    - 🚗 Autonomous driving systems  
    - 🛣️ Intelligent traffic monitoring  
    - 📊 Smart city infrastructure  
    """)

    st.markdown("---")

    # ========= HOW IT WORKS =========
    st.subheader("⚙️ How It Works")
    st.write("""
    1. Capture frames from video/image/webcam  
    2. Convert to **HSV color space** (better for color detection than RGB)  
    3. Apply **color masks** to isolate Red, Yellow, Green regions  
    4. Validate regions using size & shape filters  
    5. Draw bounding boxes and labels around detected traffic lights  
    """)

    st.markdown("---")

    # ========= USE CASES =========
    st.subheader("🚀 Applications")
    st.write("""
    - Autonomous vehicles (self-driving cars)  
    - Traffic management and monitoring systems  
    - Driver assistance tools  
    - AI & computer vision education projects  
    """)

    st.markdown("---")

    # ========= TECH STACK =========
    st.subheader("🛠️ Tech Stack")
    tech_cols = st.columns(4)
    with tech_cols[0]:
        st.markdown("✅ Python")
    with tech_cols[1]:
        st.markdown("✅ OpenCV")
    with tech_cols[2]:
        st.markdown("✅ NumPy")
    with tech_cols[3]:
        st.markdown("✅ Streamlit")

    st.markdown("---")

    # ========= FOOTER =========
    st.info("👉 Use the sidebar to navigate between **Demo, Uploads, Webcam, and Project Info**.")
    st.markdown("<div class='footer'>Made with ❤️ using Streamlit & OpenCV | Powered by AI & Computer Vision</div>", unsafe_allow_html=True)

elif st.session_state.page == "🎬 Demo Video":
    st.header("🎬 Demo Video Detection")
    demo_path = Path(__file__).parent.parent / "demo_videos" / "traffic_light_demo.mp4"

    if demo_path.exists():
        st.video(str(demo_path))
        if st.button("▶️ Run Detection on Demo"):
            with st.spinner("Processing demo video..."):
                output_path = run_detection(str(demo_path), "outputs/demo_output.mp4")
                if output_path and os.path.exists(output_path):
                    st.success("✅ Detection completed!")
                    st.video(output_path)
                    with open(output_path, "rb") as f:
                        st.download_button("⬇️ Download Processed Demo", f, file_name="processed_demo.mp4")
    else:
        st.error("⚠️ Demo video not found. Please place it in `demo_videos/traffic_light_demo.mp4`")

elif st.session_state.page == "📥 Upload Video":
    st.header("📥 Upload Your Video")
    uploaded_file = st.file_uploader("Choose a video", type=["mp4", "avi", "mov"])

    if uploaded_file:
        save_path = Path("uploads") / uploaded_file.name
        os.makedirs("uploads", exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.read())
        st.video(str(save_path))

        if st.button("▶️ Run Detection on Uploaded Video"):
            with st.spinner("Processing your video..."):
                output_path = run_detection(str(save_path), f"outputs/processed_{uploaded_file.name}")
                if output_path and os.path.exists(output_path):
                    st.success("✅ Detection completed!")
                    st.video(output_path)
                    with open(output_path, "rb") as f:
                        st.download_button("⬇️ Download Processed Video", f, file_name=f"processed_{uploaded_file.name}")

elif st.session_state.page == "🖼️ Upload Image":
    st.header("🖼️ Upload an Image")
    uploaded_img = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_img:
        st.image(uploaded_img, caption="Uploaded Image", use_column_width=True)
        if st.button("🔍 Run Detection on Image"):
            with st.spinner("Processing your image..."):
                output_path = process_image(uploaded_img, f"outputs/processed_{uploaded_img.name}")
                if output_path and os.path.exists(output_path):
                    st.success("✅ Detection completed!")
                    st.image(output_path, caption="Processed Image", use_column_width=True)
                    with open(output_path, "rb") as f:
                        st.download_button("⬇️ Download Processed Image", f, file_name=f"processed_{uploaded_img.name}")

# ========= WEBCAM (REAL-TIME) DETECTION =========
elif st.session_state.page == "📷 Webcam Detection":
    st.markdown("<div class='big-title'>📷 Webcam Detection</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Use your webcam for real-time traffic light detection</div>", unsafe_allow_html=True)

    # Optional: STUN server so it works on hosted environments too
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    # Enhanced HSV-based detector with performance tweaks
    class TrafficLightProcessor(VideoProcessorBase):
        def __init__(self):
            self.color_ranges = {
                "RED":    ((0, 120, 70),   (10, 255, 255)),
                "YELLOW": ((15, 120, 120), (35, 255, 255)),
                "GREEN":  ((45, 100, 100), (75, 255, 255)),
            }

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")

            # Resize down just for faster detection (keeps full frame, no ROI)
            small = cv2.resize(img, (640, 360))
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

            for color, (lower, upper) in self.color_ranges.items():
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    cnt = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(cnt)
                    if area > 150:  # threshold to avoid noise
                        x, y, w_box, h_box = cv2.boundingRect(cnt)

                        # Scale back to original frame size
                        scale_x = img.shape[1] / small.shape[1]
                        scale_y = img.shape[0] / small.shape[0]

                        x1 = int(x * scale_x)
                        y1 = int(y * scale_y)
                        x2 = int((x + w_box) * scale_x)
                        y2 = int((y + h_box) * scale_y)

                        if color == "RED":
                            box = (0, 0, 255)
                        elif color == "YELLOW":
                            box = (0, 255, 255)
                        else:
                            box = (0, 255, 0)

                        cv2.rectangle(img, (x1, y1), (x2, y2), box, 2)
                        cv2.putText(img, f"{color} LIGHT", (x1, max(0, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, box, 2)

            return av.VideoFrame.from_ndarray(img, format="bgr24")



    # Button to open/close the camera
    if "cam_on" not in st.session_state:
        st.session_state.cam_on = False

    colA, colB = st.columns(2)
    with colA:
        if not st.session_state.cam_on:
            if st.button("📷 Open Camera"):
                st.session_state.cam_on = True
                st.rerun()
        else:
            if st.button("🛑 Close Camera"):
                st.session_state.cam_on = False
                st.rerun()

    st.markdown("---")

    if st.session_state.cam_on:
        webrtc_streamer(
            key="traffic-light-webcam",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=TrafficLightProcessor,
        )
    else:
        st.info("Click **📷 Open Camera** to start your webcam.")

elif st.session_state.page == "📘 Project Info":
    st.markdown("<div class='big-title'>📘 Project Information</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Real-time AI-powered traffic light detection system</div>", unsafe_allow_html=True)

    with st.expander("📝 Quest Description", expanded=True):
        st.write("""
        Develop a Python program using OpenCV that processes video from camera/file 
        and identifies traffic light states in real-time.  
        Implement color-based detection with HSV color space segmentation and live visualization.
        """)

    with st.expander("⚡ Power-Up Guidelines"):
        st.write("""
        - Use OpenCV for video processing and color detection.  
        - Implement HSV color space conversion for robust segmentation.  
        - Add object validation using size/shape constraints.  
        - Display live feed with bounding boxes and labels.  
        """)

    with st.expander("🛠️ Quest Steps"):
        st.write("""
        1. Set up video capture from webcam or video file  
        2. Implement frame-by-frame processing pipeline  
        3. Convert frames to HSV color space  
        4. Define color ranges for red, yellow, green detection  
        5. Apply color-based segmentation and masking  
        6. Validate detected regions using size/shape constraints  
        7. Classify traffic light state based on detected colors  
        8. Draw bounding boxes and labels on detected lights  
        9. Display live video feed with annotations  
        10. Handle multiple traffic lights in single frame  
        """)

    with st.expander("🧰 Tech Stack"):
        st.write("""
        - Python  
        - OpenCV  
        - NumPy  
        - HSV Color Space  
        - Video Processing  
        """)

    with st.expander("📦 Deliverables"):
        st.write("""
        - Python script for traffic light detection  
        - Real-time video processing with annotations  
        - Detection accuracy report  
        - Sample output videos/screenshots  
        """)

    with st.expander("📊 Evaluation Criteria"):
        st.write("""
        - System should accurately detect and classify traffic light states in real-time.  
        - Visualization should clearly show detected regions with proper labels.  
        """)

    with st.expander("📂 Datasets"):
        st.write("""
        - Live webcam feed  
        - Traffic light video files  
        - Test images with traffic lights  
        """)

    st.markdown("<div class='footer'>🚀 Project Info Section Complete</div>", unsafe_allow_html=True)

