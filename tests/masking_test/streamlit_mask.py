#!/usr/bin/env python3
"""
Streamlit interface that overlays `streamlit_drawable_canvas` on an image and
lets users download the resulting mask.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

try:
    import streamlit.elements.image as st_image
except ImportError:  # pragma: no cover - best effort for patched Streamlit version
    st_image = None


def _ensure_image_to_url() -> None:
    if st_image is None or hasattr(st_image, "image_to_url"):
        return

    def _image_to_url(
        image: Image.Image,
        width: int,
        include_base64: bool,
        image_format: str,
        image_encoding: str,
        key: str,
    ) -> str:
        buffer = io.BytesIO()
        target_format = image_encoding or image_format or "PNG"
        image.convert(image_format or "RGB").save(buffer, format=target_format)
        return f"data:image/{target_format.lower()};base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"

    st_image.image_to_url = _image_to_url


_ensure_image_to_url()


def get_mask_from_canvas(
    image: Image.Image,
    image_data: np.ndarray | None,
    threshold: int = 16,
) -> Image.Image | None:
    if image_data is None:
        return None

    bg_array = np.array(image.convert("RGBA"), dtype=np.int16)
    drawn_array = image_data.astype(np.int16)
    diff = np.abs(drawn_array[..., :3] - bg_array[..., :3])
    mask = np.any(diff > threshold, axis=-1)
    mask_image = Image.fromarray((mask * 255).astype("uint8"), mode="L")
    return mask_image


def main() -> None:
    st.set_page_config(page_title="Mask Creator", layout="wide")
    try:
        st._config.set_option("server.baseUrlPath", "")
    except AttributeError:
        st.warning("Unable to override Streamlit baseUrlPath; background images may not load.")  # type: ignore[attr-defined]
    st.title("Streamlit: Mask Painter")
    st.write(
        "Upload an image, trace the regions you want to mask, "
        "and download the resulting grayscale mask."
    )

    col1, col2 = st.columns([2, 1])

    with col2:
        brush_width = st.slider("Brush width", 4, 120, 32)
        threshold = st.slider(
            "Mask detection sensitivity",
            4,
            64,
            16,
            help="Controls how different a pixel must be from the "
            "background to be considered part of the mask.",
        )
        stroke_color = st.color_picker("Stroke color", "#FF0000")
        uploader = st.file_uploader(
            "Image (PNG/JPEG/BMP/TIFF)",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
            accept_multiple_files=False,
        )

    if not uploader:
        st.info("Upload an image to get started.")
        return

    image = Image.open(uploader).convert("RGBA")

    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.0)",
        stroke_width=brush_width,
        stroke_color=stroke_color,
        background_image=image,
        height=image.height,
        width=image.width,
        drawing_mode="freedraw",
        update_streamlit=True,
        key="mask_canvas",
    )

    mask_image = get_mask_from_canvas(image, canvas_result.image_data, threshold)

    if mask_image:
        with col1:
            st.subheader("Mask preview")
            st.image(mask_image, clamp=True, caption="255 = Mask")
            buf = io.BytesIO()
            mask_image.save(buf, format="PNG")
            buf.seek(0)
            st.download_button(
                "Download mask",
                data=buf.getvalue(),
                file_name=f"{Path(uploader.name).stem}_mask.png",
                mime="image/png",
            )


if __name__ == "__main__":
    main()
