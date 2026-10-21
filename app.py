
import tempfile
from pathlib import Path

import streamlit as st
from predict import CricketShotPredictor

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT = BASE_DIR / "cricket_shot_prototype_model.pth"

st.set_page_config(
    page_title="Cricket Shot Classification",
    page_icon="🏏",
    layout="centered"
)

@st.cache_resource
def load_predictor():
    return CricketShotPredictor(
        checkpoint_path=CHECKPOINT
    )

st.title("🏏 Cricket Shot Classification")
st.write(
    "Upload a cricket batting video to identify "
    "the shot using EfficientNet-B0, BiGRU "
    "and Temporal Attention."
)

uploaded_file = st.file_uploader(
    "Upload a cricket video",
    type=["mp4", "avi", "mov", "mkv"]
)

if uploaded_file is not None:
    st.subheader("Uploaded Video")
    st.video(uploaded_file)

    if st.button(
        "🔍 Predict Cricket Shot",
        type="primary",
        use_container_width=True
    ):
        temp_path = None

        try:
            suffix = Path(uploaded_file.name).suffix

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_path = Path(temp_file.name)

            with st.spinner("Analyzing cricket shot..."):
                predictor = load_predictor()
                result = predictor.predict(temp_path)

            st.success("Prediction completed!")

            st.subheader("Prediction Result")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Predicted Shot",
                    result["predicted_class"]
                    .replace("_", " ").title()
                )

            with col2:
                st.metric(
                    "Model Confidence",
                    f'{result["confidence"]:.2f}%'
                )

            st.subheader("Class Probabilities")

            probabilities = result["probabilities"]

            for shot, probability in sorted(
                probabilities.items(),
                key=lambda item: item[1],
                reverse=True
            ):
                st.write(
                    f"{shot.replace('_', ' ').title()}: "
                    f"{probability:.2f}%"
                )
                st.progress(
                    min(max(probability / 100, 0.0), 1.0)
                )

            st.caption(
                "Confidence represents the model's "
                "estimated probability, not guaranteed "
                "prediction accuracy."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")

        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

st.divider()
st.caption(
    "Deep Learning Cricket Shot Classification "
    "| EfficientNet-B0 + BiGRU + Attention"
)
