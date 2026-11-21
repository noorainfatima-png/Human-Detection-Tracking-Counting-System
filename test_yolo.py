
from ultralytics import YOLO

print("Loading YOLO model...")

# Load a small pretrained model
model = YOLO("yolo26n.pt")

print("Model loaded!")
print("Starting video detection...")

# Detect only people in your video
model.predict(
    source="people.mp4",
    classes=[0],
    conf=0.25,
    show=True
)
