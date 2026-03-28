# Silent Voice (SignBridge)
### Real-Time Sign Language Recognition via DTW and Spatial Landmark Extraction

**Silent Voice** is a real-time Sign Language Recognition (SLR) system designed to reduce communication barriers between Deaf, non-verbal, and hearing communities. By prioritizing accessibility and efficiency, this project directly supports **United Nations Sustainable Development Goal 10: Reduced Inequalities**.

## 🚀 Key Features
* **Real-Time Landmark Detection:** Utilizes Google MediaPipe to extract 21 3D hand landmarks per hand, forming a 126-dimensional feature vector.
* **Zero-Training Architecture:** Employs Dynamic Time Warping (DTW) for temporal gesture matching, allowing recognition with minimal reference samples and zero model training time.
* **Natural Language Synthesis:** Features an "AI Sentence" module that uses the **Qwen** model to convert segmented signs from a Word Queue into grammatically correct sentences.
* **Low-Resource Efficiency:** Optimized for standard consumer hardware, requiring no GPU acceleration for high-fidelity recognition.

## 🛠️ Technology Stack
* **Programming Language:** Python
* **Computer Vision:** OpenCV and Google MediaPipe Hands
* **Numerical Processing:** NumPy for temporal sequence normalization
* **Algorithm:** Dynamic Time Warping (DTW) for non-linear temporal alignment

## 📂 Project Structure
The repository is organized to support a modular pipeline from data collection to real-time inference:

| File | Description |
| :--- | :--- |
| `inference_dtw.py` | The main engine for real-time gesture matching and DTW prediction. |
| `server.py` | Flask-based server to host the SignBridge web dashboard. |
| `record_templates.py` | Utility to record new gesture templates as `.npy` files. |
| `convert_templates.py` | Script to pre-process and normalize landmark data for the dictionary. |
| `index.html` | The front-end interface for visualizing detections and AI sentences. |
| `templates.json` | The pre-recorded landmark dictionary used for matching. |
| `requirements.txt` | List of Python dependencies required to run the project. |

## 🧠 How it Works
The system follows a linear data pipeline: **Webcam Input $\rightarrow$ MediaPipe Extraction $\rightarrow$ Normalization $\rightarrow$ DTW Matching $\rightarrow$ Output**.

### Mathematical Foundation
The similarity between live input ($q$) and stored templates ($c$) is calculated using the **Euclidean distance** for local costs:
$$cost(q_{i},c_{j})=||q_{i}-c_{j}||_{2}$$

The Accumulated Alignment Cost is then determined to find the optimal path through the sequences:

$$
D(i, j) = \text{cost}(q_i, c_j) + \min \{ D(i-1, j), D(i, j-1), D(i-1, j-1) \}
$$

## ⚙️ Installation & Usage
1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/silent-voice.git](https://github.com/your-username/silent-voice.git)
    cd silent-voice
    ```
2.  **Install Requirements:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run SignBridge:**
    * Start the web interface: `python server.py`
    * Run standalone inference: `python inference_dtw.py`

## 🔮 Future Roadmap
* **Transition to Deep Learning:** Moving to **LSTM or Transformer** architectures for higher accuracy.
* **Advanced Feature Engineering:** Integrating **landmark velocity** and **joint angles** for motion-invariant detection.
* **Multimodal Recognition:** Adding facial expression and posture tracking to capture non-manual markers.

## 👥 Contributors
**Team Y2NE**
