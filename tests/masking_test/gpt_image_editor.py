#!/usr/bin/env python3
"""
CLI for editing an image using `gpt-image-1` with a mask and a prompt.

The mask controls which regions are edited (white/transparent = editable, black = preserved)
and the tool saves each generated image under the output directory.
"""

from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path

from openai import OpenAI
from PIL import Image


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Edit an image with gpt-image-1 using an uploaded mask."
    )
    parser.add_argument("image", type=Path, help="Path to the base image (PNG/JPEG).")
    parser.add_argument(
        "mask",
        type=Path,
        help="Path to the mask image (white=editable, black=preserve).",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        required=True,
        help="Natural-language description of the desired edits.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("edited_images"),
        help="Where to save the edited PNG outputs.",
    )
    parser.add_argument(
        "--count",
        "-n",
        type=int,
        default=1,
        help="How many variations to request from gpt-image-1.",
    )
    parser.add_argument(
        "--size",
        "-s",
        default="1024x1024",
        choices=["256x256", "512x512", "1024x1024"],
        help="Resolution of the generated image.",
    )
    parser.add_argument(
        "--output-base",
        help="Base filename for outputs; indexes will be appended.",
        default=None,
    )
    parser.add_argument(
        "--no-auto-resize",
        action="store_true",
        help="If set, fail instead of resizing the mask to match the base image size.",
    )
    return parser.parse_args()


def _save_image_bytes(data: bytes, path: Path) -> None:
    path.write_bytes(data)
    print(f"Saved edited image to {path}")


def _load_png_bytes(
    path: Path, target_size: tuple[int, int] | None = None, resample=Image.LANCZOS
) -> tuple[io.BytesIO, tuple[int, int]]:
    img = Image.open(path)
    original_size = img.size
    if target_size and img.size != target_size:
        img = img.resize(target_size, resample=resample)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    # Give BytesIO a name so the API detects the content-type as PNG instead of octet-stream.
    buf.name = f"{path.stem}.png"
    return buf, original_size


def _parse_size(size_str: str) -> tuple[int, int]:
    try:
        w, h = size_str.lower().split("x")
        return int(w), int(h)
    except Exception as exc:
        raise SystemExit(f"Invalid size format: {size_str}. Use like 1024x1024.") from exc


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    target_size = _parse_size(args.size)

    client = OpenAI()

    image_bytes, image_size = _load_png_bytes(args.image, target_size=target_size)
    mask_bytes, mask_orig_size = _load_png_bytes(
        args.mask,
        target_size=target_size,
        resample=Image.NEAREST,
    )

    if mask_orig_size != target_size:
        if args.no_auto_resize:
            raise SystemExit(
                f"Mask size {mask_orig_size} does not match image size {image_size}; "
                "rerun with matching dimensions or drop --no-auto-resize."
            )
        print(
            f"Mask size {mask_orig_size} resized to {target_size} to satisfy API requirements."
        )
    if image_size != target_size:
        print(
            f"Image size {image_size} resized to {target_size} to match requested --size."
        )

    response = client.images.edit(
        model="gpt-image-1",
        image=image_bytes,
        mask=mask_bytes,
        prompt=args.prompt,
        n=args.count,
        size=args.size,
    )

    for idx, item in enumerate(response.data):
        decoded = base64.b64decode(item.b64_json)
        base_name = args.output_base or args.image.stem
        file_name = f"{base_name}_edit_{idx + 1}.png"
        _save_image_bytes(decoded, args.output_dir / file_name)


if __name__ == "__main__":
    main()
