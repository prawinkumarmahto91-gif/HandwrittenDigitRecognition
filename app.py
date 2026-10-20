import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image


# ---------------------------------
# 1. Page configuration
# ---------------------------------
st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢",
    layout="centered"
)

st.title("🔢 Handwritten Digit Recognition")
st.write(
    "Upload a handwritten digit image. "
    "Our CNN model will predict a digit from 0 to 9."
)
st.caption("Built with PyTorch, CNN, OpenCV and Streamlit")


# ---------------------------------
# 2. Device configuration
# ---------------------------------
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ---------------------------------
# 3. Define the same CNN architecture
# ---------------------------------
class DigitCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            1, 16, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv2d(
            16, 32, kernel_size=3, padding=1
        )

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# ---------------------------------
# 4. Load model only once
# ---------------------------------
@st.cache_resource
def load_model():

    model = DigitCNN().to(DEVICE)

    state_dict = torch.load(
        "digit_cnn_trained.pth",
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model


# ---------------------------------
# 5. OpenCV preprocessing
# ---------------------------------
def preprocess_image(pil_image):

    # Convert uploaded image to grayscale
    gray = np.array(pil_image.convert("L"))

    # Invert bright-background images to match MNIST
    if gray.mean() > 127:
        gray = 255 - gray

    # Threshold to isolate the digit
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Locate foreground pixels
    coords = cv2.findNonZero(binary)

    if coords is None:
        raise ValueError(
            "No visible digit found. Upload a clearer image."
        )

    x, y, w, h = cv2.boundingRect(coords)
    cropped = binary[y:y + h, x:x + w]

    # Resize while maintaining aspect ratio
    scale = 20 / max(w, h)

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(
        cropped,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    # Center on a black 28x28 canvas
    canvas = np.zeros((28, 28), dtype=np.uint8)

    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = resized

    # Normalize and create [batch, channel, height, width]
    tensor = torch.from_numpy(
        canvas.astype(np.float32) / 255.0
    ).unsqueeze(0).unsqueeze(0)

    return canvas, tensor


# ---------------------------------
# 6. Upload image
# ---------------------------------
uploaded_file = st.file_uploader(
    "Choose a handwritten digit image",
    type=["png", "jpg", "jpeg"]
)


# ---------------------------------
# 7. Predict digit
# ---------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Uploaded image")
    st.image(image, caption="Original image", width=250)

    try:
        processed_image, input_tensor = preprocess_image(image)

        st.subheader("OpenCV preprocessing")
        st.image(
            processed_image,
            caption="Processed 28 × 28 image",
            width=200
        )

        model = load_model()

        with torch.no_grad():
            logits = model(input_tensor.to(DEVICE))
            probabilities = torch.softmax(logits, dim=1)[0]

        predicted_digit = int(
            torch.argmax(probabilities).item()
        )

        confidence = float(
            probabilities[predicted_digit].item() * 100
        )

        st.success(f"Predicted digit: {predicted_digit}")
        st.metric(
            "Model confidence",
            f"{confidence:.2f}%"
        )

        st.subheader("Probability for each digit")

        probability_values = probabilities.cpu().numpy()

        chart_data = {
            "Digit": [str(i) for i in range(10)],
            "Probability": probability_values
        }

        st.bar_chart(
            chart_data,
            x="Digit",
            y="Probability"
        )

        st.info(
            "Confidence is the model's estimated probability, "
            "not a guarantee that the prediction is correct."
        )

    except Exception as error:
        st.error(f"Could not process this image: {error}")

else:
    st.info("Upload an image to start predicting.")