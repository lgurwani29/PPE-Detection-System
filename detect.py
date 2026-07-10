from ultralytics import YOLO

# Load your trained model
model = YOLO("best.pt")

# Run prediction on the test image
results = model.predict(
    source="test.jpg",
    conf=0.4,
    save=True
)

print("Prediction completed successfully!")