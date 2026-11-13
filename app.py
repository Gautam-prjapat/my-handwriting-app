import os
import gdown
import numpy as np
import streamlit as st
import tensorflow as tf

# Import helper modules from the original repository
import nn
import preprocessing
import utils

st.set_page_config(
    page_title="Diffusion Handwriting Generation",
    page_icon="✍️",
    layout="centered",
)

st.title("✍️ Diffusion Handwriting Generation")
st.markdown("Upload a reference handwriting image and generate custom handwritten text using diffusion models.")


@st.cache_resource
def load_diffusion_model():
    """Downloads weights from Google Drive once and caches the initialized model."""
    weights_dir = "./weights"
    weights_path = os.path.join(weights_dir, "model_weights.h5")

    # 1. Download weights from Google Drive if not already present
    if not os.path.exists(weights_path):
        os.makedirs(weights_dir, exist_ok=True)
        file_id = "1U1h_sfTx2F51vmBGnELfZvKenospyITo"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, weights_path, quiet=False)

    # 2. Initialize model architecture (default hyperparameters from inference script)
    num_attlayers = 2
    C1 = 128
    C2 = C1 * 3 // 2
    C3 = C1 * 2

    style_extractor = nn.StyleExtractor()
    model = nn.DiffusionWriter(num_layers=num_attlayers, c1=C1, c2=C2, c3=C3)

    # 3. Warm-up forward pass required before loading weights
    _stroke = tf.random.normal([1, 400, 2])
    _text = tf.random.uniform([1, 40], dtype=tf.int32, maxval=50)
    _noise = tf.random.uniform([1, 1])
    _style_vector = tf.random.normal([1, 14, 1280])
    _ = model(_stroke, _text, _noise, _style_vector)

    # 4. Load trained weights
    model.load_weights(weights_path)

    tokenizer = utils.Tokenizer()
    beta_set = utils.get_beta_set()

    return model, style_extractor, tokenizer, beta_set


# Load model on startup
with st.spinner("Downloading weights & loading model into memory (first boot takes ~30s)..."):
    try:
        model, style_extractor, tokenizer, beta_set = load_diffusion_model()
    except Exception as e:
        st.error(f"Failed to load model or download weights: {e}")
        st.stop()

# --- User Interface Inputs ---
text_string = st.text_input("Enter text to convert to handwriting:", "Hello Streamlit!")
diff_mode = st.selectbox(
    "Diffusion Sampling Mode",
    options=["new", "standard"],
    index=0,
    help="'new' uses Eq 12 from the paper; 'standard' uses Eq 9.",
)

# --- File Uploader for Style Reference ---
uploaded_file = st.file_uploader(
    "Upload Reference Handwriting Image",
    type=["png", "jpg", "jpeg", "tif"],
    help="Upload a sample image of handwriting to extract its style."
)

if uploaded_file is not None:
    # Save uploaded file temporarily so preprocessing can read it via path
    sourcename = "./temp_style_image.tif"
    with open(sourcename, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with st.expander("Preview uploaded style"):
        st.image(sourcename, width=300)
else:
    st.info("👆 Please upload a reference handwriting image above to proceed.")
    st.stop()

# --- Generation Trigger ---
if st.button("Generate Handwriting", type="primary"):
    if not text_string.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Generating handwriting strokes... This may take a few seconds on CPU."):
            try:
                # Calculate timesteps divisible by 8
                timesteps = len(text_string) * 16
                timesteps = timesteps - (timesteps % 8) + 8

                output_path = "./generated_sample.png"
                if os.path.exists(output_path):
                    os.remove(output_path)

                # Extract writer style vector
                writer_img = tf.expand_dims(preprocessing.read_img(sourcename, 96), 0)
                style_vector = style_extractor(writer_img)

                # Run inference and save output image
                utils.run_batch_inference(
                    model,
                    beta_set,
                    text_string,
                    style_vector,
                    tokenizer=tokenizer,
                    time_steps=timesteps,
                    diffusion_mode=diff_mode,
                    show_samples=False,
                    path=output_path,
                )

                # Display result
                if os.path.exists(output_path):
                    st.success("Done!")
                    st.image(output_path, caption=f"Generated: '{text_string}'", use_container_width=True)
                    with open(output_path, "rb") as img_file:
                        st.download_button(
                            label="Download Image",
                            data=img_file,
                            file_name="handwriting.png",
                            mime="image/png",
                        )
                else:
                    st.error("Inference completed, but the output image file was not created.")

            except Exception as e:
                st.error(f"Error during generation: {e}")
