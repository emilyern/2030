
import numpy as np
import json
import os

TEMPLATES_DIR = r"C:\Users\emily\OneDrive\Documents\GitHub\2030\my_templates"
OUTPUT_FILE = "templates.json"

LABELS = [f for f in os.listdir(TEMPLATES_DIR) if os.path.isdir(os.path.join(TEMPLATES_DIR, f))]
all_templates = {}

for label in LABELS:
    folder = os.path.join(TEMPLATES_DIR, label)
    if not os.path.exists(folder):
        print(f"[!] Folder not found: {folder}")
        continue

    seqs = []
    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".npy"):
            seq = np.load(os.path.join(folder, fname))  # shape (30, 126)
            seqs.append(seq.tolist())                    # convert to plain list

    all_templates[label] = seqs
    print(f"  {label}: {len(seqs)} templates")

with open(OUTPUT_FILE, "w") as f:
    json.dump(all_templates, f)

print(f"\nDone! Saved to: {OUTPUT_FILE}")
print("Now put templates.json in the same folder as your HTML file.")