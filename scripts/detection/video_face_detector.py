import cv2
import os
from mtcnn import MTCNN

# ---------------- PATHS ----------------
VIDEO_PATH = "videos/test_video.mp4"
FACE_OUTPUT_DIR = "output/video_faces"
BOX_OUTPUT_DIR = "output/video_frames_with_boxes"

os.makedirs(FACE_OUTPUT_DIR, exist_ok=True)
os.makedirs(BOX_OUTPUT_DIR, exist_ok=True)

# ---------------- SETTINGS ----------------
FRAME_SKIP = 3        # process every Nth frame
MIN_FACE_SIZE = 40

# ---------------- INIT ----------------
detector = MTCNN()
cap = cv2.VideoCapture(VIDEO_PATH)

frame_count = 0
saved_faces = 0

# ---------------- PROCESS VIDEO ----------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    try:
        faces = detector.detect_faces(rgb)
    except Exception:
        print(f"⚠️ Skipping frame {frame_count} due to detector error")
        continue

    h_img, w_img = frame.shape[:2]

    for i, face in enumerate(faces):
        x, y, w, h = face["box"]
        x, y = max(0, x), max(0, y)

        if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
            continue

        x2 = min(x + w, w_img)
        y2 = min(y + h, h_img)

        cropped_face = frame[y:y2, x:x2]
        if cropped_face.size == 0:
            continue

        saved_faces += 1
        face_path = os.path.join(
            FACE_OUTPUT_DIR,
            f"frame_{frame_count:04d}_face_{i+1}.jpg"
        )
        cv2.imwrite(face_path, cropped_face)

        # draw bounding box
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

    # save frame with boxes (optional)
    cv2.imwrite(
        os.path.join(BOX_OUTPUT_DIR, f"frame_{frame_count:04d}.jpg"),
        frame
    )

    cv2.imshow("Video Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("✅ Video processing completed")
print(f"📸 Total faces saved: {saved_faces}")
