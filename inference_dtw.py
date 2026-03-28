"""
Sign Language Recognition — DTW Template Matching (No training required)
More accurate than LSTM when data is scarce.
Usage: python inference_dtw.py
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time

# SET YOUR PATHS HERE 
DATA_ROOT   = r"C:\Users\emily\OneDrive\Documents\GitHub\2030\my_templates"
LABELS_FILE = r"C:\Users\emily\OneDrive\Documents\GitHub\2030\labels.txt"


def load_labels(path):
    with open(path, "r") as f:
        labels = [line.strip().lower() for line in f if line.strip()]
    return labels


LABELS = load_labels(LABELS_FILE)

SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.15  # DTW distance threshold (lower is stricter)
HOLD_FRAMES = 45
NO_HAND_THRESHOLD = 10


# DTW DISTANCE CALCULATION
def dtw_distance(seq1, seq2):
    """
    Calculates the DTW distance between two sequences.
    seq1, seq2: shape (T, 126)
    """
    n, m = len(seq1), len(seq2)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(seq1[i - 1] - seq2[j - 1])
            dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])

    return dtw[n, m] / (n + m)  # Normalized distance


# ──────────────────────────────────────────────
# LOAD ALL TEMPLATES
# ──────────────────────────────────────────────
def load_templates():
    templates = {}  # {label: [seq1, seq2, ...]}
    print("Loading template data...")

    for label in LABELS:
        folder = os.path.join(DATA_ROOT, label)
        if not os.path.exists(folder):
            print(f"  [!] Folder missing: {folder}")
            continue
            
        seqs = []
        for fname in os.listdir(folder):
            if fname.endswith(".npy"):
                seq = np.load(os.path.join(folder, fname))
                seq = normalize_sequence(seq)
                seqs.append(seq)
        templates[label] = seqs
        print(f"  {label}: {len(seqs)} templates loaded")

    print(f"Templates loaded! Total: {sum(len(v) for v in templates.values())}\n")
    return templates


def normalize_sequence(seq):
    """Normalization: Subtract mean and divide by std to remove position/scale bias"""
    mean = seq.mean(axis=0)
    std = seq.std(axis=0) + 1e-6
    return (seq - mean) / std


def resize_sequence(seq, target_len=SEQUENCE_LENGTH):
    """Resizes sequence to standard length"""
    if len(seq) == target_len:
        return seq
    indices = np.linspace(0, len(seq) - 1, target_len).astype(int)
    return seq[indices]

# PREDICTION LOGIC
def predict(query_seq, templates):
    """
    Returns (label, confidence_score 0~1, all_distances)
    """
    query = normalize_sequence(resize_sequence(query_seq))

    label_distances = {}
    for label, seqs in templates.items():
        if not seqs: continue
        # Average distance of the top 3 closest templates
        dists = sorted([dtw_distance(query, s) for s in seqs])
        label_distances[label] = np.mean(dists[:3])

    # Find the label with the minimum distance
    best_label = min(label_distances, key=label_distances.get)
    best_dist = label_distances[best_label]

    # Convert distance to a 0~1 confidence score (smaller distance = higher confidence)
    all_dists = list(label_distances.values())
    worst_dist = max(all_dists)
    confidence = 1.0 - (best_dist / (worst_dist + 1e-6))

    return best_label, confidence, label_distances

# MEDIAPIPE INITIALIZATION
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def extract_landmarks(results):
    left_hand = np.zeros(63, dtype=np.float32)
    right_hand = np.zeros(63, dtype=np.float32)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_lm, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            coords = np.array([[p.x, p.y, p.z] for p in hand_lm.landmark]).flatten()
            if handedness.classification[0].label == "Left":
                left_hand = coords
            else:
                right_hand = coords

    return np.concatenate([left_hand, right_hand])


# GESTURE DETECTOR STATE MACHINE
class GestureDetector:
    def __init__(self):
        self.sequence = []
        self.is_recording = False
        self.no_hand_count = 0
        self.result_text = ""
        self.result_conf = 0.0
        self.result_hold = 0
        self.all_dists = {}

    def update(self, frame_vec, hand_detected, templates):
        if hand_detected:
            self.no_hand_count = 0
            if not self.is_recording:
                self.is_recording = True
                self.sequence = []
            self.sequence.append(frame_vec)
        else:
            self.no_hand_count += 1
            if self.is_recording and self.no_hand_count >= NO_HAND_THRESHOLD:
                if len(self.sequence) >= 8:
                    label, conf, dists = predict(np.array(self.sequence), templates)
                    self.all_dists = dists
                    if conf >= CONFIDENCE_THRESHOLD:
                        self.result_text = label.upper()
                        self.result_conf = conf
                    else:
                        self.result_text = "?"
                        self.result_conf = conf
                    self.result_hold = HOLD_FRAMES
                    print(f"Prediction: {self.result_text} (confidence: {conf*100:.1f}%) | frames: {len(self.sequence)}")
                self.is_recording = False
                self.sequence = []
                self.no_hand_count = 0

        if self.result_hold > 0:
            self.result_hold -= 1
        else:
            self.result_text = ""


# UI DRAWING
def draw_ui(frame, detector, fps):
    h, w = frame.shape[:2]

    # FPS Display
    cv2.putText(frame, f"FPS: {fps:.0f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Recording Status
    if detector.is_recording:
        cv2.circle(frame, (w - 30, 25), 8, (0, 0, 255), -1)
        n = len(detector.sequence)
        cv2.putText(frame, f"{n} frames", (w - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        
        # Recording progress bar
        progress = min(n / SEQUENCE_LENGTH, 1.0)
        bar_w = int(w * 0.6 * progress)
        cv2.rectangle(frame, (w//5, h - 20), (w//5 + int(w*0.6), h - 8), (50, 50, 50), -1)
        cv2.rectangle(frame, (w//5, h - 20), (w//5 + bar_w, h - 8), (0, 200, 100), -1)
    else:
        cv2.putText(frame, "Show your hand to start", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)

    # Prediction Result UI
    if detector.result_text:
        color = (0, 255, 100) if detector.result_text != "?" else (0, 100, 255)
        text = detector.result_text
        font_scale = 2.5 if len(text) <= 5 else 1.8
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, 3)
        tx = (w - tw) // 2
        ty = h // 2 + th // 2
        
        # Result background box
        cv2.rectangle(frame, (tx - 16, ty - th - 16), (tx + tw + 16, ty + 16), (0, 0, 0), -1)
        cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, font_scale, color, 3)
        cv2.putText(frame, f"{detector.result_conf*100:.0f}%", (tx, ty + 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)

    cv2.putText(frame, "Q: quit", (10, h - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)
    return frame


# MAIN EXECUTION LOOP
def run():
    templates = load_templates()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cv2.namedWindow("Sign Language Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Sign Language Detection", 640, 480)
    cv2.setWindowProperty("Sign Language Detection", cv2.WND_PROP_TOPMOST, 1)

    detector = GestureDetector()
    prev_time = time.time()

    print("Camera started! Place your hand in the frame. Move it away after finishing the gesture.\nPress Q to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(rgb)

        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

        hand_detected = results.multi_hand_landmarks is not None
        frame_vec = extract_landmarks(results)
        detector.update(frame_vec, hand_detected, templates)

        # Calculate FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time

        frame = draw_ui(frame, detector, fps)
        cv2.imshow("Sign Language Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands_detector.close()
    print("Program exited.")


if __name__ == "__main__":
    run()
