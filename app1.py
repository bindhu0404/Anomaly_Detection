import streamlit as st
import os
import tempfile
from ethical import predict_video

st.set_page_config(page_title="Ethical Anomaly Detection", layout="centered")

# Title
st.title("🔍 Ethical Anomaly Detection")
st.write("Upload a video and the model will classify it as **Normal, Abuse, Assault, Arrest, or Arson**.")

# File uploader
uploaded_file = st.file_uploader("📂 Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file:
    # Save uploaded file temporarily
    temp_dir = tempfile.mkdtemp()
    temp_video_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.read())

    st.video(temp_video_path)  # Display uploaded video

    if st.button("🔍 Analyze Video"):
        with st.spinner("Processing..."):
            result, confidence = predict_video(temp_video_path)

        if result:
            st.success(f"**Prediction:** {result} (Confidence: {confidence*100:.2f}%)")
            if result != "Normal":
                st.warning(f"⚠️ This video contains **{result}** activity. Please review.")
            else:
                st.info("✅ This video appears **normal**.")
        else:
            st.error(confidence)  # Display error message

# Footer
st.markdown("---")
st.write("🚀 Developed with Streamlit & TensorFlow for seamless anomaly detection")
