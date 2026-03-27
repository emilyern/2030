"""
SignBridge — Local Backend Server
==================================
Handles Featherless.ai API requests so the API key
never appears in the browser or HTML.

Usage:
  1. Set your API key in the line below (API_KEY = "...")
  2. Run:  python server.py
  3. Open: http://localhost:5000  in your browser

Requirements:
  pip install flask flask-cors requests
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
import os

# ============================================================
#  SET YOUR API KEY HERE — it stays on your machine only
# ============================================================
API_KEY = "rc_aa646c4c775576a99d82a6d0fc377d9165c6a7d356e64dd851260b4063b0b5f6"
# ============================================================

FEATHERLESS_URL = "https://api.featherless.ai/v1/chat/completions"
MODEL           = "Qwen/Qwen2.5-7B-Instruct"

app = Flask(__name__, static_folder=".")
CORS(app)  # Allow requests from the browser


# ── Serve the HTML frontend ──────────────────────────────
@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(".", filename)


# ── AI sentence conversion endpoint ─────────────────────
@app.route("/convert", methods=["POST"])
def convert():
    data = request.get_json()
    words = data.get("words", [])

    if not words:
        return jsonify({"error": "No words provided"}), 400

    if API_KEY == "YOUR_FEATHERLESS_API_KEY_HERE":
        return jsonify({"error": "API key not set. Open server.py and set your API_KEY."}), 500

    prompt = (
        "You are a sign language interpreter. "
        "Convert these detected sign language words into a natural, fluent sentence "
        "a hearing person can easily understand. Keep it simple and concise. "
        "Only reply with the final sentence, nothing else.\n\n"
        f"Detected words: {' '.join(words)}"
    )

    try:
        response = requests.post(
            FEATHERLESS_URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "max_tokens": 150,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()
        sentence = result["choices"][0]["message"]["content"].strip()
        return jsonify({"sentence": sentence})

    except requests.exceptions.Timeout:
        return jsonify({"error": "Request timed out. Try again."}), 504
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code
        if status == 401:
            return jsonify({"error": "Invalid API key. Check API_KEY in server.py"}), 401
        return jsonify({"error": f"Featherless API error: HTTP {status}"}), status
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("=" * 55)
    print("  SignBridge — Local Server")
    print("=" * 55)
    if API_KEY == "YOUR_FEATHERLESS_API_KEY_HERE":
        print("  ⚠  API key not set!")
        print("  Open server.py and fill in your API_KEY.")
    else:
        print("  ✓  API key loaded")
    print(f"  Model: {MODEL}")
    print("=" * 55)
    print("  Open http://localhost:5000 in your browser")
    print("=" * 55)
    app.run(host="localhost", port=5000, debug=False)