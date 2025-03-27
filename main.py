import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
import tempfile
import os
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import chromadb

# Initialize ChromaDB client
db = chromadb.PersistentClient(path="chroma_db")
collection = db.get_or_create_collection(name="user_profiles")

# Fixed path for storing images
IMAGE_DIR = "images/target/"
os.makedirs(IMAGE_DIR, exist_ok=True)

class FaceDetectionTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame = None

    # def transform(self, frame):
    #     img = frame.to_ndarray(format="bgr")
        
    #     # Detect faces
    #     try:
    #         faces = cv2.CascadeClassifier(
    #             cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    #         )
    #         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #         detected_faces = faces.detectMultiScale(
    #             gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    #         )
            
    #         # Draw rectangles around detected faces
    #         for x, y, w, h in detected_faces:
    #             cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #     except Exception as e:
    #         st.error(f"Face detection error: {e}")
        
    #     self.captured_frame = img
    #     return img

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame = img
        return img


def save_image(image, filename):
    path = os.path.join(IMAGE_DIR, filename)
    cv2.imwrite(path, image)
    return path


def extract_embedding(image_path, model_name="Facenet"):
    try:
        embedding = DeepFace.represent(img_path=image_path, model_name=model_name)[0]["embedding"]
        return embedding
    except Exception as e:
        st.error(f"Error processing {image_path}: {e}")
        return None


def register_user(name, email, captured_image_path):
    embedding = extract_embedding(captured_image_path)
    if embedding is not None:
        collection.add(
            ids=[email],
            embeddings=[embedding],
            metadatas=[{"name": name, "email": email, "image_path": captured_image_path}]
        )
        st.success("User registered successfully!")
    else:
        st.error("Failed to register user.")


def verify_user(email, captured_image_path):
    user_data = collection.get(ids=[email])
    if user_data["ids"]:
        profile_image_path = user_data["metadatas"][0]["image_path"]
        is_verified, distance = DeepFace.verify(
            img1_path=captured_image_path,
            img2_path=profile_image_path,
            enforce_detection=False,
        )
        if is_verified:
            st.success(f"Face Verified! (Distance: {distance:.4f})")
            st.balloons()
        else:
            st.error(f"Face Verification Failed. (Distance: {distance:.4f})")
    else:
        st.error("User not found!")


def main():
    st.title("Face Recognition System")

    action = st.sidebar.radio("Select Action", ["Register", "Login"])

    face_detector = FaceDetectionTransformer()

    webrtc_ctx = webrtc_streamer(
        key="face-detection",
        video_transformer_factory=lambda: face_detector,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640, "max": 1280},
                "height": {"ideal": 480, "max": 720},
                "facingMode": "user",
            },
            "audio": False,
        },
        async_transform=True,
    )

    if webrtc_ctx.video_transformer:
        if st.button("Capture Frame"):
            # frame = webrtc_ctx.video_transformer.frame
            if webrtc_ctx and webrtc_ctx.video_transformer:
                frame = webrtc_ctx.video_transformer.frame
            else:
                st.error("WebRTC stream is not active. Please ensure your camera is connected.")
                return

            if frame is not None:
                st.image(frame, channels="BGR")
                # verify_image(TARGET_IMAGE_PATH, frame)

                if action == "Register":
                    st.header("User Registration")
                    name = st.text_input("Enter your name")
                    email = st.text_input("Enter your email")

                    if st.button("Register"):
                        if not webrtc_ctx.state.playing:
                            st.error("WebRTC stream is not active. Please ensure camera is connected.")
                            return
                        captured_frame = face_detector.captured_frame
                        if captured_frame is not None and captured_frame.size > 0:
                            image_path = save_image(captured_frame, f"{email}.jpg")
                            register_user(name, email, image_path)
                        else:
                            st.error("No valid frame captured. Please check camera connection.")

                elif action == "Login":
                    st.header("User Login")
                    email = st.text_input("Enter your email")

                    if st.button("Login"):
                        if not webrtc_ctx.state.playing:
                            st.error("WebRTC stream is not active. Please ensure camera is connected.")
                            return
                        captured_frame = face_detector.captured_frame
                        if captured_frame is not None and captured_frame.size > 0:
                            image_path = save_image(captured_frame, "login_attempt.jpg")
                            verify_user(email, image_path)
                        else:
                            st.error("No valid frame captured. Please check camera connection.")
            else:
                st.warning("No frame captured yet.")



if __name__ == "__main__":
    main()
    