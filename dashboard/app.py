from __future__ import annotations

import json
import mimetypes
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import DISEASE_CLASSES, PROCESSED_DATA_DIR, STREAMLIT_LAYOUT, STREAMLIT_PAGE_TITLE


def fetch_health(api_url: str, timeout: int = 5) -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(f"{api_url}/health", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def predict(api_url: str, name: str, data: bytes, mime: Optional[str], threshold: float, top_k: int, timeout: int) -> Dict[str, Any]:
    ctype = mime or mimetypes.guess_type(name)[0] or "image/png"
    r = requests.post(
        f"{api_url}/predict",
        params={"threshold": threshold, "top_k": top_k},
        files={"file": (name, data, ctype)},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def main() -> None:
    st.set_page_config(page_title=STREAMLIT_PAGE_TITLE, layout=STREAMLIT_LAYOUT)

    st.title("Chest Disease Classifier")

    with st.sidebar:
        api_url = st.text_input("API URL", os.getenv("API_BASE_URL", "http://127.0.0.1:8000"))
        threshold = st.slider("Threshold", 0.01, 0.90, 0.50, 0.01)
        top_k = st.slider("Top-K", 1, len(DISEASE_CLASSES), 5)
        timeout = st.slider("Timeout (s)", 10, 180, 60, 5)

        health = fetch_health(api_url)
        if health is None:
            st.error("API offline")
            st.caption("uvicorn api.main:app --reload")
        else:
            st.success("API online")
            st.json(health)

    upload = st.file_uploader("X-ray image", type=["png", "jpg", "jpeg"])
    if upload is None:
        st.info("Upload an image to predict.")
        return

    data = upload.getvalue()
    st.image(data, caption=upload.name, use_container_width=True)

    if not st.button("Predict"):
        return

    with st.spinner("Running..."):
        try:
            result = predict(
                api_url,
                upload.name,
                data,
                getattr(upload, "type", None),
                threshold,
                top_k,
                timeout,
            )
        except requests.HTTPError as exc:
            try:
                st.error(json.dumps(exc.response.json(), indent=2))
            except Exception:
                st.error(str(exc))
            return
        except requests.RequestException as exc:
            st.error(str(exc))
            return

    labels: List[str] = result.get("predicted_labels", [])
    ms = float(result.get("inference_ms", 0))
    top = result.get("top_predictions", [])

    c1, c2, c3 = st.columns(3)
    c1.metric("ms", f"{ms:.1f}")
    c2.metric("hits", len(labels))
    c3.metric("top", top[0]["disease"] if top else "-")

    if labels:
        st.success(", ".join(labels))
    else:
        st.info("Nothing above threshold.")

    top_df = pd.DataFrame(result.get("top_predictions", []))
    all_df = pd.DataFrame(result.get("all_predictions", []))
    if not top_df.empty:
        st.dataframe(top_df, use_container_width=True)
    if not all_df.empty:
        st.bar_chart(all_df.set_index("disease")["probability"])

    gradcam_dir = Path(PROCESSED_DATA_DIR) / "gradcam"
    if gradcam_dir.exists():
        stem = Path(upload.name).stem
        matches = sorted(gradcam_dir.glob(f"*{stem}*.png"))
        if matches:
            st.image(str(matches[0]), caption=matches[0].name, use_container_width=True)


if __name__ == "__main__":
    main()
