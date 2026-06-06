from __future__ import annotations

import json
import mimetypes
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import DISEASE_CLASSES, PROCESSED_DATA_DIR, STREAMLIT_LAYOUT, STREAMLIT_PAGE_TITLE


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def fetch_health(api_url: str, timeout: int = 60) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Returns (health_dict, error_message).
    Separates timeout (slow/cold start) from actual offline errors so the UI
    gives the user an actionable message instead of just "API offline".
    """
    url = api_url.rstrip("/") + "/health"
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.Timeout:
        return None, f"Health check timed out after {timeout}s — the API may still be starting (Render cold start). Try again in 30s."
    except requests.exceptions.ConnectionError:
        return None, f"Cannot connect to {url}. Check the API URL or start the server."
    except requests.exceptions.HTTPError as exc:
        return None, f"API returned HTTP {exc.response.status_code}."
    except requests.RequestException as exc:
        return None, str(exc)


def predict(
    api_url: str,
    name: str,
    data: bytes,
    mime: Optional[str],
    threshold: float,
    top_k: int,
    timeout: int,
) -> Dict[str, Any]:
    url = api_url.rstrip("/") + "/predict"
    ctype = mime or mimetypes.guess_type(name)[0] or "image/png"
    r = requests.post(
        url,
        params={"threshold": threshold, "top_k": top_k},
        files={"file": (name, data, ctype)},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title=STREAMLIT_PAGE_TITLE, layout=STREAMLIT_LAYOUT)
    st.title("Chest Disease Classifier")

    # -----------------------------------------------------------------------
    # Sidebar
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.header("Settings")
        api_url = st.text_input(
            "API URL",
            os.getenv("API_URL", "https://chest-disease-classifier.onrender.com"),
        )
        threshold = st.slider("Threshold", 0.01, 0.90, 0.50, 0.01)
        top_k = st.slider("Top-K", 1, len(DISEASE_CLASSES), 5)
        # Increase timeout so Render cold-start (up to 60s) doesn't look like offline
        timeout = st.slider("Timeout (s)", 10, 180, 90, 5)

        st.divider()
        st.subheader("API Status")

        # Cache the health result per (url, timeout) so slider moves don't
        # re-hit the network on every Streamlit rerun.
        @st.cache_data(ttl=30, show_spinner=False)
        def cached_health(url: str, t: int):
            return fetch_health(url, timeout=t)

        with st.spinner("Checking API…"):
            health, err = cached_health(api_url, timeout)

        if health is not None:
            st.success("✅ API online")
            st.json(health)
        else:
            st.error(f"❌ {err}")
            st.caption("If on Render free tier, wait 30–60 s for cold start then refresh.")

        if st.button("🔄 Re-check API"):
            # Clear cache and force a fresh health check
            cached_health.clear()
            st.rerun()

    # -----------------------------------------------------------------------
    # Main area
    # -----------------------------------------------------------------------
    upload = st.file_uploader("X-ray image", type=["png", "jpg", "jpeg"])
    if upload is None:
        st.info("Upload a chest X-ray image to predict.")
        return

    data = upload.getvalue()
    st.image(data, caption=upload.name, use_container_width=True)

    if not st.button("Predict", type="primary"):
        return

    if health is None:
        st.warning("API appears offline. Attempting prediction anyway…")

    with st.spinner("Running inference…"):
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
        except requests.exceptions.Timeout:
            st.error(
                f"Prediction timed out after {timeout}s. "
                "The Render free tier may be cold-starting — increase Timeout and retry."
            )
            return
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
    top: List[dict] = result.get("top_predictions", [])

    # -----------------------------------------------------------------------
    # Results
    # -----------------------------------------------------------------------
    c1, c2, c3 = st.columns(3)
    c1.metric("Inference (ms)", f"{ms:.1f}")
    c2.metric("Diseases detected", len(labels))
    c3.metric("Top prediction", top[0]["disease"] if top else "—")

    if labels:
        st.success("**Detected:** " + ", ".join(labels))
    else:
        st.info("No disease detected above the threshold.")

    st.subheader("Top Predictions")
    top_df = pd.DataFrame(top)
    if not top_df.empty:
        st.dataframe(top_df, use_container_width=True)

    st.subheader("All Class Probabilities")
    all_df = pd.DataFrame(result.get("all_predictions", []))
    if not all_df.empty:
        st.bar_chart(all_df.set_index("disease")["probability"])

    # Grad-CAM overlay if available
    gradcam_dir = Path(PROCESSED_DATA_DIR) / "gradcam"
    if gradcam_dir.exists():
        stem = Path(upload.name).stem
        matches = sorted(gradcam_dir.glob(f"*{stem}*.png"))
        if matches:
            st.subheader("Grad-CAM")
            st.image(str(matches[0]), caption=matches[0].name, use_container_width=True)


if __name__ == "__main__":
    main()