"""Streamlit dashboard for Chest Disease Classifier inference."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st

# Make src/config importable from dashboard/
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import (  # noqa: E402
    DISEASE_CLASSES,
    PROCESSED_DATA_DIR,
    STREAMLIT_LAYOUT,
    STREAMLIT_PAGE_TITLE,
)


def configure_page() -> None:
    st.set_page_config(
        page_title=STREAMLIT_PAGE_TITLE,
        layout=STREAMLIT_LAYOUT,
        initial_sidebar_state="expanded",
    )


def fetch_health(api_base_url: str, timeout_sec: int) -> Optional[Dict[str, Any]]:
    try:
        response = requests.get(f"{api_base_url}/health", timeout=timeout_sec)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def call_predict_api(
    api_base_url: str,
    image_name: str,
    image_bytes: bytes,
    content_type: Optional[str],
    threshold: float,
    top_k: int,
    timeout_sec: int,
) -> Dict[str, Any]:
    resolved_content_type = content_type or mimetypes.guess_type(image_name)[0] or "image/png"
    files = {"file": (image_name, image_bytes, resolved_content_type)}
    params = {"threshold": threshold, "top_k": top_k}

    response = requests.post(
        f"{api_base_url}/predict",
        params=params,
        files=files,
        timeout=timeout_sec,
    )
    response.raise_for_status()
    return response.json()


def render_upload_preview(uploaded_name: str, image_bytes: bytes, content_type: Optional[str]) -> None:
    """Show metadata-only preview and open-in-new-tab link (no inline image render)."""
    resolved_content_type = content_type or mimetypes.guess_type(uploaded_name)[0] or "image/png"
    size_kb = len(image_bytes) / 1024.0

    st.subheader("Upload Preview")
    left, middle, right = st.columns(3)
    left.markdown(f"**File:** `{uploaded_name}`")
    middle.markdown(f"**Type:** `{resolved_content_type}`")
    right.markdown(f"**Size:** `{size_kb:.1f} KB`")

    encoded = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:{resolved_content_type};base64,{encoded}"
    st.markdown(
        f'<a href="{data_url}" target="_blank" rel="noopener noreferrer">'
        "Open image in new tab"
        "</a>",
        unsafe_allow_html=True,
    )
    st.caption("Inline image is hidden by design.")


def render_prediction_summary(result: Dict[str, Any]) -> None:
    st.subheader("Prediction Summary")

    predicted_labels: List[str] = result.get("predicted_labels", [])
    inference_ms = float(result.get("inference_ms", 0.0))
    top_predictions = result.get("top_predictions", [])

    left, middle, right = st.columns(3)
    left.metric("Inference Time (ms)", f"{inference_ms:.1f}")
    middle.metric("Predicted Labels", len(predicted_labels))
    if top_predictions:
        right.metric("Top Disease", top_predictions[0]["disease"])
    else:
        right.metric("Top Disease", "N/A")

    if predicted_labels:
        st.success("Predicted diseases: " + ", ".join(predicted_labels))
    else:
        st.info("No disease crossed the current threshold. Try lowering threshold for demo exploration.")


def render_prediction_tables(result: Dict[str, Any]) -> None:
    top_df = pd.DataFrame(result.get("top_predictions", []))
    all_df = pd.DataFrame(result.get("all_predictions", []))

    if not top_df.empty:
        st.subheader("Top Predictions")
        st.dataframe(top_df, use_container_width=True)

    if not all_df.empty:
        st.subheader("All Class Probabilities")
        st.bar_chart(all_df.set_index("disease")["probability"])
        st.dataframe(all_df, use_container_width=True)


def render_gradcam_hint(uploaded_name: str) -> None:
    st.subheader("Grad-CAM")
    gradcam_dir = Path(PROCESSED_DATA_DIR) / "gradcam"
    if not gradcam_dir.exists():
        st.caption("No Grad-CAM folder found yet. Run `python src/gradcam.py` to generate explainability images.")
        return

    stem = Path(uploaded_name).stem
    matches = sorted(gradcam_dir.glob(f"*{stem}*.png"))

    if not matches:
        st.caption(
            "No pre-generated Grad-CAM image matched this upload name. "
            "Run `python src/gradcam.py` again to refresh visualizations."
        )
        return

    st.image(str(matches[0]), caption=f"Existing Grad-CAM match: {matches[0].name}", use_container_width=True)


def main() -> None:
    configure_page()

    st.title("Chest Disease Classifier Dashboard")
    st.caption("Phase 5: Streamlit frontend connected to FastAPI inference service")

    with st.sidebar:
        st.header("Inference Settings")
        default_api_url = os.getenv("API_BASE_URL", "https://chest-disease-classifier.onrender.com")
        api_base_url = st.text_input("API Base URL", value=default_api_url)
        threshold = st.slider("Threshold", min_value=0.01, max_value=0.90, value=0.50, step=0.01)
        top_k = st.slider("Top-K", min_value=1, max_value=len(DISEASE_CLASSES), value=5, step=1)
        timeout_sec = st.slider("Request Timeout (sec)", min_value=10, max_value=180, value=60, step=5)

        st.divider()
        st.subheader("API Status")
        health = fetch_health(api_base_url, timeout_sec=5)
        if health is None:
            st.error("API unreachable")
            st.caption("Start API: `uvicorn api.main:app --reload`")
        else:
            st.success("API connected")
            st.json(health)

    uploaded = st.file_uploader("Upload Chest X-ray", type=["png", "jpg", "jpeg"])

    if uploaded is None:
        st.info("Upload an image to run prediction.")
        return

    image_bytes = uploaded.getvalue()
    render_upload_preview(uploaded.name, image_bytes, getattr(uploaded, "type", None))

    if st.button("Run Prediction", type="primary"):
        with st.spinner("Running model inference..."):
            try:
                result = call_predict_api(
                    api_base_url=api_base_url,
                    image_name=uploaded.name,
                    image_bytes=image_bytes,
                    content_type=getattr(uploaded, "type", None),
                    threshold=threshold,
                    top_k=top_k,
                    timeout_sec=timeout_sec,
                )
            except requests.HTTPError as exc:
                detail = ""
                try:
                    detail = json.dumps(exc.response.json(), indent=2)
                except Exception:
                    detail = str(exc)
                st.error(f"API request failed.\n{detail}")
                return
            except requests.RequestException as exc:
                st.error(f"Could not reach API: {exc}")
                return

        render_prediction_summary(result)
        render_prediction_tables(result)
        render_gradcam_hint(uploaded.name)


if __name__ == "__main__":
    main()
