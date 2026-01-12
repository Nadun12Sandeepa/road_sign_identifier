import gradio as gr
import cv2
import numpy as np
import tempfile
import os
from inference_sdk import InferenceHTTPClient

# ----------------------------
# Roboflow Client
# ----------------------------
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="DIAhXQf6AUsyM1PRfdFa"   # 🔴 Replace if needed
)

MODEL_ID = "road-sign-peqgi/1"


# ----------------------------
# Image Detection
# ----------------------------
def detect_image(image):
    if image is None:
        return None

    # Save image temporarily
    temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(temp_file.name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # Run inference
    result = CLIENT.infer(temp_file.name, model_id=MODEL_ID)

    # Draw predictions
    for pred in result["predictions"]:
        x, y, w, h = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
        label = pred["class"]
        confidence = pred["confidence"]

        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"{label} {confidence:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    os.unlink(temp_file.name)
    return image


# ----------------------------
# Video Detection
# ----------------------------
def detect_video(video_path):
    cap = cv2.VideoCapture(video_path)

    temp_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(temp_output.name, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame temporarily
        temp_frame = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        cv2.imwrite(temp_frame.name, frame)

        result = CLIENT.infer(temp_frame.name, model_id=MODEL_ID)

        for pred in result["predictions"]:
            x, y, w, h = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
            label = pred["class"]
            confidence = pred["confidence"]

            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{label} {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        out.write(frame)
        os.unlink(temp_frame.name)

    cap.release()
    out.release()

    return temp_output.name


# ----------------------------
# Gradio UI
# ----------------------------
with gr.Blocks(title="🚦 Road Sign Detection") as demo:
    gr.Markdown("# 🚦 Road Sign Detection (Images & Videos)")
    gr.Markdown("Powered by **Roboflow Serverless + Gradio**")

    with gr.Tab("🖼 Image Detection"):
        img_input = gr.Image(type="numpy")
        img_output = gr.Image()
        img_btn = gr.Button("Detect Road Signs")
        img_btn.click(detect_image, img_input, img_output)

    with gr.Tab("🎥 Video Detection"):
        video_input = gr.Video()
        video_output = gr.Video()
        video_btn = gr.Button("Detect Road Signs in Video")
        video_btn.click(detect_video, video_input, video_output)

demo.launch()
