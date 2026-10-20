
from pathlib import Path

import numpy as np
import streamlit as st
import torch

from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.linalg import lstsq

import sys

# Allow importing the model from the training directory.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "training"))

from neural_net.elunet import ELUnet


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MEAN = 25.36114133
STD = 44.31162568
PIXEL_SIZE = 6 / 1000.0
RADIUS = 5

CHECKPOINT_PATH = ROOT / "models" / "latest_model.pt"

DEVICE = torch.device("cpu")


# --------------------------------------------------
# Load the user's trained model
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    model = ELUnet(in_channels=1, out_channels=2).to(DEVICE)

    state_dict = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model


# --------------------------------------------------
# Trilateration centroid estimation
# Adapted from the evaluation code supplied by the user.
# --------------------------------------------------

def trilateration_centroid(
    dist_map,
    seg_map,
    radius=RADIUS,
    pixel_size=PIXEL_SIZE,
):
    dist_map = dist_map.astype(np.float64)
    seg_map = seg_map.astype(np.float64).copy()

    height, width = dist_map.shape
    threshold = 2

    dist_min = ndimage.minimum_filter(
        dist_map,
        size=(10, 10),
        mode="constant",
    )

    local_minima = dist_map == dist_min
    centroid_est = []

    for current_row in range(height):
        for current_col in range(width):

            if not (
                dist_map[current_row, current_col] <= threshold
                and seg_map[current_row, current_col] > 0
                and local_minima[current_row, current_col]
            ):
                continue

            x_i, y_i, r_i = [], [], []

            for row_shift in range(-radius, radius + 1):
                for col_shift in range(-radius, radius + 1):

                    row = current_row + row_shift
                    col = current_col + col_shift

                    # Ignore out-of-image positions.
                    if not (0 <= row < height and 0 <= col < width):
                        continue

                    if seg_map[row, col] > 0:
                        x_i.append(col + 0.5)
                        y_i.append(row + 0.5)
                        r_i.append(dist_map[row, col])

                        # Prevent reusing these pixels for another
                        # nearby centroid, as in the evaluation code.
                        seg_map[row, col] = 0

            if len(x_i) < 2:
                continue

            x_n, y_n, r_n = x_i[-1], y_i[-1], r_i[-1]
            n = len(x_i)

            A = np.zeros((n - 1, 2))
            B = np.zeros(n - 1)

            for i in range(n - 1):
                A[i, 0] = 2 * (x_n - x_i[i])
                A[i, 1] = 2 * (y_n - y_i[i])

                B[i] = (
                    r_i[i] ** 2
                    - r_n ** 2
                    - x_i[i] ** 2
                    - y_i[i] ** 2
                    + x_n ** 2
                    + y_n ** 2
                )

            solution = lstsq(A, B, lapack_driver="gelsy")[0]

            # Preserve the original evaluation code's mm convention.
            u = solution[0] * pixel_size
            v = solution[1] * pixel_size

            centroid_est.append([float(u), float(v)])

    return centroid_est


# --------------------------------------------------
# Inference
# --------------------------------------------------

def detect_stars(model, image_array):
    # Match the normalization used in evaluation.
    image_tensor = torch.from_numpy(
        image_array.astype(np.float32)
    ).unsqueeze(0).unsqueeze(0)

    image_tensor = (image_tensor - MEAN) / STD
    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():
        prediction = model(image_tensor)

    # Channel 0: segmentation logits
    # Channel 1: distance map
    seg_prediction = (
        torch.sigmoid(prediction[0, 0]) > 0.5
    ).cpu().numpy().astype(np.float32)

    dist_prediction = prediction[0, 1].cpu().numpy()

    centroids_mm = trilateration_centroid(
        dist_prediction.copy(),
        seg_prediction.copy(),
        radius=RADIUS,
        pixel_size=PIXEL_SIZE,
    )

    # Convert mm coordinates back to image pixel coordinates.
    # The original algorithm uses (column + 0.5, row + 0.5).
    centroids_px = [
        (u / PIXEL_SIZE - 0.5, v / PIXEL_SIZE - 0.5)
        for u, v in centroids_mm
    ]

    return seg_prediction, dist_prediction, centroids_px


# --------------------------------------------------
# Streamlit interface
# --------------------------------------------------

st.set_page_config(
    page_title="CNN-Based Star Detection",
    page_icon="✨",
    layout="centered",
)

st.title("CNN-Based Star Detection")
st.write(
    "Upload a 256 × 256 grayscale star-field image to detect "
    "stars using the trained ELUnet model."
)

st.caption(
    "Model: your trained checkpoint | "
    "Inference: CPU | "
    "Input: 256 × 256 grayscale"
)

uploaded_file = st.file_uploader(
    "Upload a star-field image",
    type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
)

if uploaded_file is not None:
    try:
        pil_image = Image.open(uploaded_file).convert("L")
        image_array = np.asarray(pil_image, dtype=np.float32)

        if image_array.shape != (256, 256):
            st.error(
                f"This version requires a 256 × 256 image. "
                f"Your image is {image_array.shape[1]} × "
                f"{image_array.shape[0]}. "
                "Please use a correctly sized image; it will "
                "not be resized automatically."
            )
            st.stop()

        st.subheader("Input image")
        st.image(
            pil_image,
            caption="Uploaded grayscale image",
            width=400,
        )

        if st.button("Detect stars", type="primary"):
            try:
                model = load_model()

                with st.spinner("Running star detection..."):
                    seg_mask, dist_map, centroids = detect_stars(
                        model, image_array
                    )

                # Draw centroid markers on a copy of the input.
                overlay = pil_image.convert("RGB")
                draw = ImageDraw.Draw(overlay)

                for x, y in centroids:
                    cx, cy = int(round(x)), int(round(y))
                    draw.ellipse(
                        (cx - 4, cy - 4, cx + 4, cy + 4),
                        outline=(255, 0, 0),
                        width=1,
                    )

                st.subheader("Detection results")
                st.image(
                    overlay,
                    caption="Predicted centroids marked in red",
                    width=400,
                )

                st.metric("Detected centroids", len(centroids))

                st.write(
                    "Coordinates are shown in image pixel units, "
                    "with x increasing to the right and y increasing "
                    "downward."
                )

                if centroids:
                    import csv
                    import io

                    csv_buffer = io.StringIO()
                    writer = csv.writer(csv_buffer)
                    writer.writerow(["x_pixel", "y_pixel"])
                    writer.writerows(
                        [[round(x, 4), round(y, 4)]
                         for x, y in centroids]
                    )

                    st.dataframe(
                        [
                            {
                                "x_pixel": round(x, 4),
                                "y_pixel": round(y, 4),
                            }
                            for x, y in centroids
                        ],
                        use_container_width=True,
                    )

                    st.download_button(
                        "Download centroid coordinates (CSV)",
                        data=csv_buffer.getvalue(),
                        file_name="detected_centroids.csv",
                        mime="text/csv",
                    )
                else:
                    st.info(
                        "No centroids were detected. Try a suitable "
                        "test image from the project's dataset."
                    )

                with st.expander("View model output maps"):
                    st.image(
                        seg_mask,
                        caption="Segmentation mask",
                        clamp=True,
                    )
                    st.image(
                        dist_map,
                        caption="Predicted distance map",
                        clamp=True,
                    )

            except Exception as exc:
                st.error(f"Inference failed: {exc}")
                st.exception(exc)

    except Exception as exc:
        st.error(f"Could not read the uploaded image: {exc}")
