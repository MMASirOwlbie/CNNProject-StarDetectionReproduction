
from pathlib import Path

import numpy as np
import streamlit as st
import torch

from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.linalg import lstsq

import csv
import io
import os
import tempfile
import cv2
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


def detect_video_frame(model, frame):
    """Run the existing detection pipeline on one video frame."""
    _, _, centroids_px = detect_stars(model, frame)
    return centroids_px

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
    "Detect stars in grayscale images or process a sequence "
    "of frames using the trained ELUNet model."
)

st.caption("Model: trained ELUNet | Inference: CPU")

mode = st.radio(
    "Choose input type",
    ["Image Detection", "Video Detection"],
    horizontal=True,
)

# ==================================================
# IMAGE DETECTION
# ==================================================

if mode == "Image Detection":

    uploaded_file = st.file_uploader(
        "Upload a star-field image",
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
        key="image_upload",
    )

    if uploaded_file is not None:
        try:
            pil_image = Image.open(uploaded_file).convert("L")
            image_array = np.asarray(pil_image, dtype=np.float32)

            if image_array.shape != (256, 256):
                st.error(
                    f"Expected a 256 × 256 image, but received "
                    f"{image_array.shape[1]} × {image_array.shape[0]}."
                )
            else:
                st.image(
                    pil_image,
                    caption="Input image",
                    width=400,
                )

                if st.button("Detect stars", type="primary"):
                    model = load_model()

                    with st.spinner("Detecting stars..."):
                        seg_mask, dist_map, centroids = detect_stars(
                            model, image_array
                        )
                        st.write("Distance map minimum:", float(dist_map.min()))
                        st.write("Distance map maximum:", float(dist_map.max()))
                        st.write("Distance map mean:", float(dist_map.mean()))

                    overlay = pil_image.convert("RGB")
                    draw = ImageDraw.Draw(overlay)

                    for x, y in centroids:
                        if 0 <= x < 256 and 0 <= y < 256:
                            cx, cy = int(round(x)), int(round(y))
                            draw.ellipse(
                                (cx - 4, cy - 4, cx + 4, cy + 4),
                                outline=(255, 0, 0),
                                width=1,
                            )

                    st.subheader("Detection results")
                    st.image(
                        overlay,
                        caption="Detected centroids in red",
                        width=400,
                    )

                    st.metric("Detected centroids", len(centroids))

                    csv_buffer = io.StringIO()
                    csv_writer = csv.writer(csv_buffer)
                    csv_writer.writerow(["x_pixel", "y_pixel"])
                    csv_writer.writerows(
                        [[round(x, 4), round(y, 4)]
                         for x, y in centroids]
                    )

                    st.download_button(
                        "Download centroid CSV",
                        data=csv_buffer.getvalue(),
                        file_name="detected_centroids.csv",
                        mime="text/csv",
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
            st.error(f"Image processing failed: {exc}")
            st.exception(exc)

# ==================================================
# VIDEO DETECTION
# ==================================================

else:

    st.write(
        "Upload a NumPy grayscale video array (.npy), such as "
        "video_Test3.npy. Each array should have the shape "
        "(number of frames, height, width)."
    )

    video_file = st.file_uploader(
        "Upload grayscale video",
        type=["npy"],
        key="video_upload",
    )

    fps = st.number_input(
        "Output video FPS",
        min_value=1,
        max_value=60,
        value=20,
        step=1,
    )

    if video_file is not None:

        if st.button("Process video", type="primary"):

            temp_path = None

            try:
                frames = np.load(
                    io.BytesIO(video_file.getvalue()),
                    allow_pickle=False,
                )

                if frames.ndim != 3:
                    st.error(
                        "Invalid video array. Expected shape "
                        "(frames, height, width)."
                    )
                    st.stop()

                if frames.shape[0] == 0:
                    st.error("The uploaded video contains no frames.")
                    st.stop()

                if not np.issubdtype(frames.dtype, np.number):
                    st.error("The video array must contain numeric pixels.")
                    st.stop()

                if (
                    not np.isfinite(frames).all()
                    or frames.min() < 0
                    or frames.max() > 255
                ):
                    st.error(
                        "Video pixel values must be finite and "
                        "within the range 0–255."
                    )
                    st.stop()

                height, width = frames.shape[1:]

                if height == 0 or width == 0:
                    st.error("The video has invalid frame dimensions.")
                    st.stop()

                st.info(
                    f"Frames: {len(frames)} | "
                    f"Resolution: {width} × {height}"
                )

                model = load_model()

                progress = st.progress(0)
                status = st.empty()

                csv_rows = []

                with tempfile.NamedTemporaryFile(
                    suffix=".mp4",
                    delete=False,
                ) as temp_file:
                    temp_path = temp_file.name

                writer = cv2.VideoWriter(
                    temp_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    float(fps),
                    (width, height),
                    True,
                )

                if not writer.isOpened():
                    writer.release()
                    raise RuntimeError(
                        "Could not create the output MP4. "
                        "Check OpenCV video codec support."
                    )

                try:
                    for frame_index, frame in enumerate(frames):

                        centroids = detect_video_frame(model, frame)

                        output_frame = cv2.cvtColor(
                            frame.astype(np.uint8),
                            cv2.COLOR_GRAY2BGR,
                        )

                        for centroid_index, point in enumerate(centroids):
                            x, y = float(point[0]), float(point[1])

                            # Keep the raw coordinates in the CSV.
                            csv_rows.append([
                                frame_index,
                                centroid_index,
                                x,
                                y,
                            ])

                            # Draw only finite coordinates inside the frame.
                            if (
                                np.isfinite(x)
                                and np.isfinite(y)
                                and 0 <= x < width
                                and 0 <= y < height
                            ):
                                cv2.circle(
                                    output_frame,
                                    (int(round(x)), int(round(y))),
                                    4,
                                    (0, 255, 0),
                                    1,
                                )

                        writer.write(output_frame)

                        progress.progress(
                            (frame_index + 1) / len(frames)
                        )
                        status.write(
                            f"Processed frame {frame_index + 1} "
                            f"of {len(frames)}"
                        )

                finally:
                    writer.release()

                # Read the finished video after releasing the writer.
                with open(temp_path, "rb") as output_file:
                    video_bytes = output_file.read()

                st.success("Video processing completed!")

                st.video(video_bytes)

                st.download_button(
                    "Download processed video (MP4)",
                    data=video_bytes,
                    file_name="star_detection_output.mp4",
                    mime="video/mp4",
                )

                csv_buffer = io.StringIO()
                csv_writer = csv.writer(csv_buffer)
                csv_writer.writerow([
                    "frame_index",
                    "centroid_index",
                    "x_pixel",
                    "y_pixel",
                ])
                csv_writer.writerows(csv_rows)

                st.download_button(
                    "Download all centroid coordinates (CSV)",
                    data=csv_buffer.getvalue(),
                    file_name="video_centroids.csv",
                    mime="text/csv",
                )

                st.metric("Frames processed", len(frames))
                st.metric("Total centroid detections", len(csv_rows))

                st.caption(
                    "Coordinates are retained in the CSV even if a "
                    "centroid falls outside the frame. Such centroids "
                    "are not drawn on the output video."
                )

                with st.expander("Preview centroid data"):
                    st.dataframe(
                        csv_rows[:100],
                        column_config={
                            0: "frame_index",
                            1: "centroid_index",
                            2: "x_pixel",
                            3: "y_pixel",
                        },
                        use_container_width=True,
                    )
                    st.caption(
                        "Showing up to the first 100 detections."
                    )

            except Exception as exc:
                st.error(f"Video processing failed: {exc}")
                st.exception(exc)

            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

