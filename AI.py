from ultralytics import YOLO

# Load our pretrained crack-detection model
model = YOLO("model/crack.pt")

# Run the model on test image
results = model.predict(
    source="test_images",
    conf=0.25,
    save=True
)

# Print what the model found
for result in results:
    print("\nImage:", result.path)

    if result.boxes is None or len(result.boxes) == 0:
        print("No cracks detected.")
        continue

    for box in result.boxes:
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])

        print(
            "Detected:",
            model.names[class_id],
            "| Confidence:",
            round(confidence, 3)

         )
