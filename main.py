
import cv2
from ultralytics import YOLO

# 1. LOAD YOLO MODEL
print("Loading YOLO model...")
model = YOLO("yolo26n.pt")

# 2. LOAD VIDEO
cap = cv2.VideoCapture("people.mp4")

if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

# 3. STORE UNIQUE PERSON IDs
person_ids = {}

print("Human Tracking Started!")
print("Press 'q' to quit.")

# 4. PROCESS VIDEO
while True:
    ret, frame = cap.read()

    if not ret:
        print("Video completed.")
        break

    # 5. DETECT AND TRACK PEOPLE
    results = model.track(
        source=frame,
        persist=True,
        classes=[0],
        conf=0.25,
        tracker="bytetrack.yaml",
        verbose=False
    )

    # 6. GET TRACKING INFORMATION
    boxes = results[0].boxes

    if boxes.id is not None:
        tracking_ids = boxes.id.int().cpu().tolist()
        coordinates = boxes.xyxy.int().cpu().tolist()

        for track_id, (x1, y1, x2, y2) in zip(tracking_ids, coordinates):

            # Assign a new P1, P2, P3 ID
            if track_id not in person_ids:
                person_ids[track_id] = f"P{len(person_ids) + 1}"

            person_name = person_ids[track_id]

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Display person ID
            cv2.putText(
                frame,
                person_name,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    # 7. DISPLAY TOTAL UNIQUE PEOPLE
    total_people = len(person_ids)

    cv2.rectangle(frame, (10, 10), (340, 65), (0, 0, 0), -1)

    cv2.putText(
        frame,
        f"Total People: {total_people}",
        (20, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    # 8. DISPLAY VIDEO
    cv2.imshow("Human Detection and Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 9. CLEANUP
cap.release()
cv2.destroyAllWindows()

print("--------------------------------")
print("VIDEO PROCESSING COMPLETED")
print("Total Unique People:", len(person_ids))
print("--------------------------------")
