import streamlit as st
import google.generativeai as genai
from PIL import Image, ImageDraw
import pytesseract
from gtts import gTTS
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import os
import tempfile
import pygame

# Set the Gemini API key directly (not recommended for security reasons)
genai.configure(api_key="AIzaSyBpBmyOZ1bF2zKWrTU-cyWw2XDLJFVHmpc")

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# config your Gemini API key
my_api_key = "AIzaSyBpBmyOZ1bF2zKWrTU-cyWw2XDLJFVHmpc"
genai.configure(api_key=my_api_key)

# Streamlit App
st.set_page_config(page_title="Vision AI", layout="centered", page_icon="🤖")

# response function
def get_response(input_prompt, image_data):
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content([input_prompt, image_data[0]])
    return response.text

# function to convert image to bytes
def image_to_bytes(uploaded_file):
    try:
        bytes_data = uploaded_file.getvalue()

        image_parts = [
            {
                "mime_type": uploaded_file.type,
                "data": bytes_data
            }
        ]

        return image_parts
    
    except Exception as e:
        raise FileNotFoundError(f"Failed to process image. Please try again. Error: {e}")

# function to extract text from image
def extract_text_from_image(uploaded_file):
    try:
        img = Image.open(uploaded_file)

        # pytesseract to extract text
        extracted_text = pytesseract.image_to_string(img)

        if not extracted_text.strip():
            return "No text found in the image."
        
        return extracted_text

    except Exception as e:
        raise ValueError(f"Failed to extract text. Error: {e}")

# function for text to speech using gTTS
def text_to_speech_gtts(text):
    try:
        tts = gTTS(text, lang='en')
        
        # Save the audio file to a temporary file
        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            tts.save(temp_file.name)
            # Play the audio
            pygame.mixer.init()
            pygame.mixer.music.load(temp_file.name)
            pygame.mixer.music.play()
        
    except Exception as e:
        raise RuntimeError(f"Failed to convert text to speech. Error: {e}")

# Load object detection model (Faster R-CNN)
@st.cache_resource
def load_object_detection_model():
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

object_detection_model = load_object_detection_model()

def detect_objects(image, threshold=0.5, iou_threshold=0.5):
    try:
        # Transform image to tensor
        transform = transforms.Compose([transforms.ToTensor()])
        img_tensor = transform(image)
        
        # Get predictions
        predictions = object_detection_model([img_tensor])[0]
        
        # Perform Non-Maximum Suppression
        keep = torch.ops.torchvision.nms(predictions['boxes'], predictions['scores'], iou_threshold)
        
        # Filter results based on NMS and score threshold
        filtered_predictions = {
            'boxes': predictions['boxes'][keep],
            'labels': predictions['labels'][keep],
            'scores': predictions['scores'][keep]
        }
        
        return filtered_predictions
    except Exception as e:
        raise RuntimeError(f"Failed to detect objects. Error: {e}")

# COCO class labels (91 categories)
COCO_CLASSES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A",
    "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "N/A", "dining table", "N/A", "N/A", "toilet",
    "N/A", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "N/A", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Highlight detected objects in the image
def draw_boxes(image, predictions, threshold=0.5):
    draw = ImageDraw.Draw(image)
    labels = predictions['labels']
    boxes = predictions['boxes']
    scores = predictions['scores']

    for label, box, score in zip(labels, boxes, scores):
        if score > threshold:
            x1, y1, x2, y2 = box
            class_name = COCO_CLASSES[label.item()]  # Map label ID to class name
            draw.rectangle([x1, y1, x2, y2], outline="yellow", width=3)
            draw.text((x1, y1), f"{class_name} ({score:.2f})", fill="black")
    return image


# Response function for Personalized Assistance
def get_assistance_response(input_prompt, image_data):
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content([input_prompt, image_data[0]])
    return response.text

# Prompt Engineering
input_prompt = """
You are an AI assistant designed to assist visually impaired individuals
by analyzing images and providing descriptive outputs.
Your task is to:
- Analyze the uploaded image and describe its content in clear and simple language.
- Provide detailed information about objects, people, settings, or activities in the scene.
"""

st.title("TrueVision AI 🧠")

# Sidebar for features
st.sidebar.header("Features")
st.sidebar.markdown("""
- **Scene Interpretation**: Analyze and describe the content of uploaded images.
- **Object Recognition**: Identify objects or obstacles within the image for enhanced understanding.
- **Task-Specific Assistance**: Offer tailored guidance based on the image content.
- **Text-to-Audio Conversion**: Convert extracted text from images into spoken words for better accessibility.
""")

# Main section for file uploader and buttons
uploaded_file = st.file_uploader("Upload an image:", type=['jpg', 'jpeg', 'png', 'webp'])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

# Buttons with custom colors
col1, col2, col3, col4 = st.columns(4)
scene_analysis_button = col1.button("Describe Scene 🏞️", key="scene_button", help="Analyze the uploaded image and provide a description.")
assist_button = col2.button("Assist Tasks 🤖", key="assist_button", help="Provide personalized assistance based on the image.")
text_tts_button = col3.button("Extract Text 📝", key="text_tts_button", help="Extract and process text from the uploaded image.")
stop_audio_button = col4.button("Stop Audio ⏹️", key="stop_audio_button", help="Stop currently playing audio.")

# Apply CSS for button colors
st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #4CAF50; color: white; font-size: 14px; margin: 2px;
}
div.stButton > button:hover {
    background-color: #45a049; color: white;
}
</style>
""", unsafe_allow_html=True)

# Personalized Assistance
if assist_button and uploaded_file:
    with st.spinner("Providing task-specific assistance..."):
        st.subheader("🤖 Assistance:")
        image_data = image_to_bytes(uploaded_file)
        assist_prompt = """
        You are a helpful AI assistant. Analyze the uploaded image and identify tasks
        you can assist with, such as recognizing objects or reading labels.
        """
        response = get_assistance_response(assist_prompt, image_data)
        st.write(response)
        
        # Convert response to audio
        text_to_speech_gtts(response)

# Scene Analysis
if scene_analysis_button and uploaded_file:
    with st.spinner("Analyzing scene..."):
        st.subheader("🏞️ Scene Description:")
        image_data = image_to_bytes(uploaded_file)
        scene_response = get_response(input_prompt, image_data)
        st.write(scene_response)
        
        # Convert scene description to audio
        text_to_speech_gtts(scene_response)

# Extract Text from Image
if text_tts_button and uploaded_file:
    with st.spinner("Extracting text..."):
        st.subheader("📝 Extracted Text:")
        text = extract_text_from_image(uploaded_file)
        st.write(text)
        
        # Convert extracted text to audio
        text_to_speech_gtts(text)

# Stop audio
if stop_audio_button:
    pygame.mixer.music.stop()
