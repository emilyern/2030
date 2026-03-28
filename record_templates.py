"""
Sign Language Recording — Record your own templates
Usage: python record_templates.py

- Reads labels from labels.txt (one word per line)
- Automatically SKIPS labels that already have enough recordings
- Only records NEW labels or incomplete ones
"""

import cv2
import mediapipe as mp
import numpy as np
import os

# SET YOUR PATHS HERE 
SAVE_DIR    = r"C:\Users\emily\OneDrive\Documents\GitHub\2030\my_templates"
LABELS_FILE = r"C:\Users\emily\OneDrive\Documents\GitHub\2030\labels.txt"

RECORDINGS_PER_LABEL = 5
SEQUENCE_LENGTH      = 30
NO_HAND_THRESHOLD    = 12

# Read LABELS from file
def load_labels(path):
    with open(path, "r") as f:
        labels = [line.strip().lower() for line in f if line.strip()]
    return labels

# check recording
def count_existing(label):
    folder = os.path.join(SAVE_DIR, label)
    if not os.path.exists(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.endswith(".npy")])

mp_hands          = mp.solutions.hands
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def extract_landmarks(results):
    left_hand  = np.zeros(63, dtype=np.float32)
    right_hand = np.zeros(63, dtype=np.float32)
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_lm, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            coords = np.array([[p.x, p.y, p.z] for p in hand_lm.landmark]).flatten()
            if handedness.classification[0].label == "Left":
                left_hand = coords
            else:
                right_hand = coords
    return np.concatenate([left_hand, right_hand])


def resize_sequence(seq, target_len=SEQUENCE_LENGTH):
    if len(seq) == target_len:
        return seq
    indices = np.linspace(0, len(seq) - 1, target_len).astype(int)
    return seq[indices]


#record
def record_one(cap, label, rec_num, total):
    sequence      = []
    is_recording  = False
    no_hand_count = 0
    saved         = False

    print(f"\n  [{label}] Recording {rec_num}/{total} — Show hand to START, hide to STOP.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame   = cv2.flip(frame, 1)
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(rgb)

        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

        hand_detected = results.multi_hand_landmarks is not None
        frame_vec     = extract_landmarks(results)
        h, w          = frame.shape[:2]

        if hand_detected:
            no_hand_count = 0
            if not is_recording:
                is_recording = True
                sequence     = []
            sequence.append(frame_vec)

            progress = min(len(sequence) / SEQUENCE_LENGTH, 1.0)
            bar_w    = int(w * 0.6 * progress)
            cv2.rectangle(frame, (w//5, h-20), (w//5 + int(w*0.6), h-8), (50,50,50),   -1)
            cv2.rectangle(frame, (w//5, h-20), (w//5 + bar_w,       h-8), (0,200,100), -1)
            cv2.circle(frame, (w-30, 25), 8, (0,0,255), -1)
            cv2.putText(frame, f"{len(sequence)} frames", (w-120, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
        else:
            no_hand_count += 1
            if is_recording and no_hand_count >= NO_HAND_THRESHOLD:
                if len(sequence) >= 8:
                    saved = True
                    break
                else:
                    is_recording  = False
                    sequence      = []
                    no_hand_count = 0
                    cv2.putText(frame, "Too short! Try again", (10, h//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,100,255), 2)

        # Header
        cv2.rectangle(frame, (0,0), (w, 50), (0,0,0), -1)
        cv2.putText(frame, f"Sign: {label.upper()}  ({rec_num}/{total})", (10, 32),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (0,255,150), 2)

        hint  = "Hide hand to STOP recording" if is_recording else "Show hand to START recording"
        color = (0,220,100) if is_recording else (150,150,150)
        cv2.putText(frame, hint,               (10, h-35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        cv2.putText(frame, "S: skip  Q: quit", (10, h-55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120,120,120), 1)

        cv2.imshow("Record Templates", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return None, "quit"
        if key == ord('s'):
            return None, "skip"

    if saved:
        seq_array = np.array(resize_sequence(np.array(sequence)), dtype=np.float32)
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            cv2.rectangle(frame, (0,0), (frame.shape[1], frame.shape[0]), (0,0,0), -1)
            cv2.putText(frame, f"SAVED! ({len(sequence)} frames)", (80, frame.shape[0]//2),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (0,255,100), 2)
            cv2.imshow("Record Templates", frame)
            cv2.waitKey(800)
        return seq_array, "ok"

    return None, "ok"

#main
def run():
    os.makedirs(SAVE_DIR, exist_ok=True)

    labels = load_labels(LABELS_FILE)

    print("=" * 50)
    print("Sign Language Template Recording Tool")
    print(f"Labels file : {LABELS_FILE}")
    print(f"Labels found: {labels}")
    print("=" * 50)

    # Figure out which labels actually need recording
    todo = []
    for label in labels:
        existing = count_existing(label)
        if existing >= RECORDINGS_PER_LABEL:
            print(f"  [SKIP] '{label}' — already has {existing} recordings")
        else:
            needed = RECORDINGS_PER_LABEL - existing
            print(f"  [TODO] '{label}' — has {existing}, needs {needed} more")
            todo.append((label, existing))

    if not todo:
        print("\nAll labels already recorded! Nothing to do.")
        print("To re-record a word, delete its folder inside my_templates and run again.")
        return

    print(f"\nWill record: {[t[0] for t in todo]}\n")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cv2.namedWindow("Record Templates", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Record Templates", 640, 480)
    cv2.setWindowProperty("Record Templates", cv2.WND_PROP_TOPMOST, 1)

    for label, existing_count in todo:
        label_dir = os.path.join(SAVE_DIR, label)
        os.makedirs(label_dir, exist_ok=True)

        needed = RECORDINGS_PER_LABEL - existing_count

        # Prep screen
        for _ in range(60):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            cv2.rectangle(frame, (0,0), (frame.shape[1], frame.shape[0]), (20,20,20), -1)
            cv2.putText(frame, f"Next: {label.upper()}", (60, frame.shape[0]//2 - 20),
                        cv2.FONT_HERSHEY_DUPLEX, 2.0, (255,220,0), 3)
            cv2.putText(frame, f"Need {needed} more recording(s)", (150, frame.shape[0]//2 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180,180,180), 1)
            cv2.imshow("Record Templates", frame)
            cv2.waitKey(30)

        saved_count = 0
        for i in range(needed):
            next_index = existing_count + i + 1   # avoid overwriting existing files
            seq, status = record_one(cap, label, i + 1, needed)

            if status == "quit":
                print("\nExiting...")
                cap.release()
                cv2.destroyAllWindows()
                return
            if status == "skip" or seq is None:
                print(f"  Skipping recording {i+1}")
                continue

            save_path = os.path.join(label_dir, f"my_{next_index:02d}.npy")
            np.save(save_path, seq)
            saved_count += 1
            print(f"  Saved: {save_path}")

        print(f"  [{label}] Done! Saved {saved_count} new recording(s).")

    cap.release()
    cv2.destroyAllWindows()
    hands_detector.close()
    print("\nAll done!")
    print(f"Templates saved in: {SAVE_DIR}")


if __name__ == "__main__":
    run()
