from flask import Flask, render_template, request, jsonify
import hashlib
import pandas as pd
import joblib
import os
import re
from ahocorasick import Automaton
from urllib.parse import urlparse
import numpy as np

app = Flask(__name__)

# ===== GLOBAL VARIABLES =====
HASH_AUTOMATON = None
PHISHING_URLS = set()
PHISHING_MODEL = None
PHISHING_FEATURES = None

# ===== DATASET VALIDATION =====
def validate_datasets():
    errors = []

    if not os.path.exists("hash_dataset.csv"):
        errors.append("❌ Missing hash_dataset.csv")
    else:
        try:
            with open("hash_dataset.csv", 'r') as f:
                if not f.readline().strip():
                    errors.append("❌ hash_dataset.csv is empty")
        except Exception as e:
            errors.append(f"❌ Error reading hash_dataset.csv: {str(e)}")

    required_datasets = {
        "phishinglist.csv": 1,  # expects at least one column (URL)
        "phishing.csv": 2       # expects URL and status
    }

    for dataset, min_cols in required_datasets.items():
        if not os.path.exists(dataset):
            errors.append(f"❌ Missing {dataset}")
        else:
            try:
                df = pd.read_csv(dataset, on_bad_lines='skip')
                if len(df.columns) < min_cols:
                    errors.append(f"❌ {dataset} has insufficient columns (expected {min_cols})")
                elif len(df) == 0:
                    errors.append(f"❌ {dataset} is empty")
            except Exception as e:
                errors.append(f"❌ Error reading {dataset}: {str(e)}")

    if errors:
        print("\n".join(errors))
        return False
    return True

# ===== (1) FILE HASH SCANNING =====
def build_hash_automaton(dataset_file="hash_dataset.csv"):
    automaton = Automaton()
    try:
        # Read the CSV file using pandas to handle different formats
        df = pd.read_csv(dataset_file, header=None, on_bad_lines='skip')
        
        # Iterate through all rows and columns to find hash values
        for _, row in df.iterrows():
            for value in row:
                if pd.notna(value):
                    hash_value = str(value).strip()
                    if len(hash_value) == 64 and re.match(r'^[a-fA-F0-9]{64}$', hash_value):
                        automaton.add_word(hash_value, hash_value)
        
        automaton.make_automaton()
        print(f"✅ Loaded hash database ({len(automaton)} hashes)")
        return automaton
    except Exception as e:
        print(f"❌ Error loading hash dataset: {e}")
        return None

def calculate_file_hash(file_path):
    sha256 = hashlib.sha256()
    try:
        with open(file_path, 'rb') as file:
            while chunk := file.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception as e:
        print(f"❌ Error calculating hash: {e}")
        return None

# ===== (2) PHISHING URL DETECTION =====
def load_phishing_urls(dataset_file="phishinglist.csv"):
    phishing_urls = set()
    try:
        df = pd.read_csv(dataset_file, on_bad_lines='skip')
        url_col = next((col for col in df.columns if 'url' in col.lower()), None)

        if not url_col:
            raise ValueError("No column resembling 'url' found in phishinglist.csv")

        for url in df[url_col]:
            if pd.notna(url):
                clean_url = str(url).strip().lower()
                if not clean_url.startswith(('http://', 'https://')):
                    clean_url = 'http://' + clean_url
                phishing_urls.add(clean_url)

        print(f"✅ Loaded {len(phishing_urls)} phishing URLs for exact matching")
        return phishing_urls
    except Exception as e:
        print(f"❌ Error loading phishing URLs: {e}")
        return set()

def train_phishing_model(dataset_file="phishing.csv"):
    try:
        df = pd.read_csv(dataset_file, on_bad_lines='skip')

        url_col = next((col for col in df.columns if 'url' in col.lower()), None)
        status_col = next((col for col in df.columns if 'status' in col.lower()), None)

        if not url_col or not status_col:
            raise ValueError("phishing.csv must contain URL and status columns")

        feature_cols = [col for col in df.columns if col not in [url_col, status_col]]
        X = df[feature_cols]
        y = df[status_col].apply(lambda x: 1 if str(x).lower() == 'phishing' else 0)

        global PHISHING_FEATURES
        PHISHING_FEATURES = list(X.columns)

        model_file = 'phishing_model.joblib'
        if os.path.exists(model_file):
            model = joblib.load(model_file)
            print("✅ Loaded existing phishing model")
        else:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            joblib.dump(model, model_file)
            print("✅ Trained new phishing model")

        return model
    except Exception as e:
        print(f"❌ Error training ML model: {e}")
        return None

def extract_url_features(url):
    try:
        if not PHISHING_FEATURES:
            raise ValueError("Feature names not loaded")

        features = {col: 0 for col in PHISHING_FEATURES}
        parsed = urlparse(url)

        features['length_url'] = len(url)
        features['length_hostname'] = len(parsed.netloc)
        features['ip'] = 1 if re.match(r'^\d+\.\d+\.\d+\.\d+$', parsed.netloc) else 0
        features['nb_dots'] = url.count('.')
        features['nb_hyphens'] = url.count('-')
        features['nb_slash'] = url.count('/')
        features['nb_www'] = 1 if 'www.' in parsed.netloc.lower() else 0
        features['nb_com'] = 1 if parsed.netloc.endswith('.com') else 0
        features['ratio_digits_url'] = sum(c.isdigit() for c in url) / max(1, len(url))

        return [features[col] for col in PHISHING_FEATURES]
    except Exception as e:
        print(f"❌ Error extracting features: {e}")
        return None

def check_phishing(url):
    if not url or not isinstance(url, str):
        return {"detected": False, "method": "invalid input", "confidence": 0}

    url_lower = url.strip().lower()
    if not url_lower.startswith(('http://', 'https://')):
        url_lower = 'http://' + url_lower

    if url_lower in PHISHING_URLS:
        return {"detected": True, "method": "exact match", "confidence": 100}

    if PHISHING_MODEL:
        features = extract_url_features(url_lower)
        if features:
            try:
                proba = PHISHING_MODEL.predict_proba([features])[0][1]
                return {
                    "detected": proba > 0.5,
                    "method": "machine learning",
                    "confidence": round(proba * 100, 2)
                }
            except Exception as e:
                print(f"⚠️ ML prediction error: {e}")

    return {"detected": False, "method": "no match", "confidence": 0}

# ===== FLASK ROUTES =====
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/faq")
def faq():
    return render_template("FAQ.html")

@app.route("/upload", methods=["POST"])
def scan_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        temp_path = f"temp_{file.filename}"
        file.save(temp_path)
        file_hash = calculate_file_hash(temp_path)
        os.remove(temp_path)

        if not file_hash:
            return jsonify({"error": "Hash calculation failed"}), 500

        is_malicious = file_hash in HASH_AUTOMATON
        return jsonify({
            "status": "success",
            "is_malicious": is_malicious,
            "hash": file_hash,
            "message": "⚠️ Malicious file detected!" if is_malicious else "✅ File is safe."
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/check-url", methods=["POST"])
def check_url():
    data = request.get_json()
    if not data or 'url' not in data or not data['url'].strip():
        return jsonify({"error": "No URL provided"}), 400

    url = data['url']
    result = check_phishing(url)

    if result["detected"]:
        return jsonify({
            "status": "warning",
            "detected": True,
            "method": result["method"],
            "confidence": result["confidence"],
            "message": f"⚠️ Suspicious URL detected ({result['confidence']}% confidence)",
        })
    else:
        return jsonify({
            "status": "safe",
            "detected": False,
            "method": result["method"],
            "confidence": 100 - result["confidence"],
            "message": "✅ URL appears safe)",
        })

# ===== START APP =====
if __name__ == "__main__":
    if not validate_datasets():
        print("❌ Dataset validation failed. Check CSV files.")
        exit(1)

    HASH_AUTOMATON = build_hash_automaton()
    PHISHING_URLS = load_phishing_urls()
    PHISHING_MODEL = train_phishing_model()

    if HASH_AUTOMATON and PHISHING_URLS and PHISHING_MODEL:
        print("🚀 All systems loaded! Starting server...")
        app.run(debug=True)
    else:
        print("❌ Failed to load required components.")