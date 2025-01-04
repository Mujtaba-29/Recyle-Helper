
import os
import streamlit as st
from groq import Groq
from ultralytics import YOLO
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import numpy as np

# Access Groq API key from Streamlit Secrets
api_key = st.secrets["GROQ_API_KEY"]

# Set it as an environment variable
os.environ["GROQ_API_KEY"] = api_key

# Initialize Groq client
client = Groq(api_key=api_key)

# Carbon footprint reduction data (kg CO2 per kg recycled)
carbon_reduction_data = {
    "plastic bottle": 3.8,
    "metal container": 9.0,
    "burnable waste": 2.0,
    "glass bottle": 0.5,
    "paper": 1.3,
    "plastic bag": 2.5,
    "wood": 1.7,
    "rubber": 6.0,
}

# ADE20K class label mapping for SegFormer
ade20k_labels = {
    17: "plastic bottle",
    36: "glass bottle",
    49: "paper",
    72: "wood",
    85: "metal container",
    108: "burnable waste",
    120: "plastic bag",
    150: "rubber",
}

# Predefined list of clutter objects with emojis
predefined_clutter_items = {
    "plastic bottle": "🧴",
    "metal container": "🛢️",
    "burnable waste": "🔥",
    "glass bottle": "🍾",
    "paper": "📄",
    "plastic bag": "🛍️",
    "wood": "🪵",
    "rubber": "🚗",
}

# Load YOLOv8 model
@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")

model = load_yolo_model()

# Load SegFormer model and feature extractor
@st.cache_resource
def load_segformer_model():
    feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    return feature_extractor, model

segformer_extractor, segformer_model = load_segformer_model()

# Function to call Groq LLM for recycling suggestions
def get_recycling_suggestions_from_groq(item, quantity):
    prompt = (
        f"You are an expert in recycling and sustainability. "
        f"Suggest profitable and eco-friendly uses for {quantity} kg of {item}, "
        f"including household uses, ways to monetize them, and calculate carbon footprint reduction. "
        f"Keep your response concise and practical. Add emojis to enhance clarity."
    )
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            stream=False,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error fetching suggestions: {e}"

# Sidebar
st.sidebar.markdown(
    '''
    <div style="text-align: center;">
        <h2 style="color: #004d40;">♻️ Navigation</h2>
        <p style="color: #00796b;">Use the app to identify waste items and generate recycling suggestions.</p>
    </div>
    ''',
    unsafe_allow_html=True,
)
action = st.sidebar.radio("Choose an action:", ["Upload Image", "Get Suggestions for Items"])

# Main app
st.markdown(
    '''
    <div style="text-align: center; background-color: #004d40; padding: 20px; border-radius: 10px;">
        <h1 style="color: #ffffff;">♻️ Recycle-Smart-PK</h1>
        <p style="font-size: 18px; color: #ffffff;">Powered by LLM 🌍</p>
    </div>
    ''',
    unsafe_allow_html=True,
)

if action == "Upload Image":
    st.markdown(
        '''
        <div style="text-align: center; background-color: #e3f2fd; padding: 10px; border-radius: 5px;">
            <h3 style="color: #01579b;">Upload an image of waste, and we'll identify items, suggest recycling ideas, and calculate carbon footprint reduction!</h3>
        </div>
        ''',
        unsafe_allow_html=True,
    )
    uploaded_image = st.file_uploader("Upload an image of the waste:", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        st.write("### YOLOv8: Detecting Waste Items...")
        yolo_results = model.predict(image, conf=0.1)
        yolo_detected_items = [model.model.names[int(pred[5])] for pred in yolo_results[0].boxes.data.tolist()]

        st.write("### SegFormer: Analyzing Segmentation...")
        segformer_inputs = segformer_extractor(images=image, return_tensors="pt")
        segformer_outputs = segformer_model(**segformer_inputs)
        segmentation_map = segformer_outputs.logits.argmax(dim=1).squeeze().numpy()
        segformer_detected_items = [
            ade20k_labels[class_id]
            for class_id in np.unique(segmentation_map)
            if class_id in ade20k_labels
        ]

        combined_items = set(yolo_detected_items + segformer_detected_items)

        if combined_items:
            st.write("### Combined Results:")
            st.write(", ".join(combined_items))

            total_carbon_reduction = 0
            for item in combined_items:
                st.markdown(f"**Recycling Idea for {item}:**")
                response = get_recycling_suggestions_from_groq(item, 1)
                carbon_reduction = max(0.5, min(2.5, carbon_reduction_data.get(item.lower(), 0) * 1))
                total_carbon_reduction += carbon_reduction

                st.write(response)
                st.markdown(
                    f'''<p style="color: #2e7d32;">🌍 Carbon Footprint Reduction: {carbon_reduction:.2f} kg CO₂</p>''',
                    unsafe_allow_html=True,
                )
                st.write("---")

            st.markdown(
                f'''<div style="padding: 15px; text-align: center; background-color: #004d40; color: #ffffff; border-radius: 5px;">
                    🌟 Total Carbon Footprint Reduction: <b>{total_carbon_reduction:.2f} kg CO₂ saved</b>
                </div>''',
                unsafe_allow_html=True,
            )
        else:
            st.error("No recognizable waste items detected.")

# Motivational Message
st.markdown(
    '''
    <div style="text-align: center; padding: 20px; background-color: #dcedc8; border-radius: 10px;">
        <h3 style="color: #33691e;">🌍 Let's Keep Our Planet Green!</h3>
        <p style="color: #2e7d32;">Recycling is not just an action but a responsibility. Together, we can make a difference. ♻️💚</p>
    </div>
    ''',
    unsafe_allow_html=True,
)
