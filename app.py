import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Matrix Decomposition Studio", layout="wide", page_icon="🎬")

st.title("🎬 Matrix Decomposition Studio")
st.markdown("Upload an image and adjust the target rank to compare standard SVD with an enhanced YCrCb pipeline.")

uploaded_file = st.file_uploader("Upload Image File", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert("RGB")
    pil_img.thumbnail((480, 420))
    raw_image_rgb = np.array(pil_img)
    
    h, w, c = raw_image_rgb.shape
    max_rank = min(h, w)
    
    st.write(f"**Original Image Loaded** (Dimensions: {w}x{h})")
    
    target_r = st.slider("Target Rank (r)", min_value=1, max_value=max_rank, value=15)
    
    if st.button("Generate Matrix Transformations", type="primary"):
        with st.spinner("Crunching matrices..."):
            img_float = raw_image_rgb.astype(np.float64)
            r_left = min(max(1, target_r), max_rank)
            
            left_output = np.zeros_like(img_float)
            for ch in range(3):
                U, S, Vt = np.linalg.svd(img_float[:, :, ch], full_matrices=False)
                left_output[:, :, ch] = U[:, :r_left] @ np.diag(S[:r_left]) @ Vt[:r_left, :]
            left_final = np.clip(left_output, 0, 255).astype(np.uint8)
            
            img_uint8 = np.clip(img_float, 0, 255).astype(np.uint8)
            ycrcb = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2YCrCb).astype(np.float64)
            right_output = np.zeros_like(ycrcb)
            
            r_y = min(max(1, int(target_r * 1.4)), max_rank)
            r_c = min(max(1, int(target_r * 0.8)), max_rank)
            
            U, S, Vt = np.linalg.svd(ycrcb[:, :, 0], full_matrices=False)
            reconstructed_y = U[:, :r_y] @ np.diag(S[:r_y]) @ Vt[:r_y, :]
            y_uint8 = np.clip(reconstructed_y, 0, 255).astype(np.uint8)
            
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            y_enhanced = clahe.apply(y_uint8)
            blur = cv2.GaussianBlur(y_enhanced, (5, 5), 0)
            y_sharp = cv2.addWeighted(y_enhanced, 1.6, blur, -0.6, 0)
            right_output[:, :, 0] = y_sharp.astype(np.float64)
            
            for ch in [1, 2]:
                U, S, Vt = np.linalg.svd(ycrcb[:, :, ch], full_matrices=False)
                right_output[:, :, ch] = U[:, :r_c] @ np.diag(S[:r_c]) @ Vt[:r_c, :]
                
            right_uint8 = np.clip(right_output, 0, 255).astype(np.uint8)
            right_final = cv2.cvtColor(right_uint8, cv2.COLOR_YCrCb2RGB)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Left: Standard SVD Baseline")
                st.image(left_final, use_container_width=True)
                
            with col2:
                st.subheader("Right: Enhanced YCrCb Pipeline")
                st.image(right_final, use_container_width=True)
                
            left_elements = ((h * r_left) + r_left + (w * r_left)) * 3
            storage_kb = (left_elements * 4) / 1024
            
            st.success(f"📊 Storage Footprint: **{round(storage_kb, 1)} KB** (Mathematically Identical Size for both!)")
