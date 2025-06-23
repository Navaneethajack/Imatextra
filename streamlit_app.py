import streamlit as st
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torchvision.models import vit_b_16
import easyocr

# Load EasyOCR (optional, for OCR output)
ocr_reader = easyocr.Reader(['en'], gpu=False)

# Load Vision Transformer model
@st.cache_resource
def load_vit_model():
    model = vit_b_16(pretrained=True)
    model.eval()
    return model

vit_model = load_vit_model()

# Image transformation for ViT
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# Title
st.title("📷 OCR with Vision Transformer (ViT)")
st.write("Upload an image to extract text using EasyOCR and analyze using ViT.")

# Upload
uploaded_img = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_img is not None:
    image = Image.open(uploaded_img).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ---------- OCR with EasyOCR ----------
    with st.spinner("Extracting text with EasyOCR..."):
        ocr_result = ocr_reader.readtext(np.array(image))

    st.subheader("📝 OCR Extracted Text:")
    for box in ocr_result:
        st.write(f"- {box[1]}")

    # ---------- ViT Feature Extraction ----------
    with st.spinner("Analyzing image with Vision Transformer (ViT)..."):
        input_tensor = transform(image).unsqueeze(0)  # Add batch dim
        with torch.no_grad():
            vit_output = vit_model(input_tensor)
            top_probs = torch.nn.functional.softmax(vit_output[0], dim=0)
            top5_prob, top5_catid = torch.topk(top_probs, 5)

        from torchvision.models import get_model_weights
        weights = get_model_weights("vit_b_16").DEFAULT
        labels_map = weights.meta["categories"]

        st.subheader("🔍 ViT Top-5 Predictions:")
        for i in range(top5_prob.size(0)):
            st.write(f"{labels_map[top5_catid[i]]}: {top5_prob[i].item():.4f}")
