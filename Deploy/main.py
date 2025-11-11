import os
import uuid
from datetime import datetime
from functools import wraps

import cv2
import numpy as np
import tf_keras as keras
from app import app
from flask import flash, redirect, render_template, request, session, url_for
from pymongo import MongoClient, errors
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "inceptionNet_model.h5")

MONGO_DETAILS = os.environ.get(
    "MONGO_DETAILS",
    "mongodb+srv://chandansh82:ansh123@cluster0.can7y.mongodb.net/?retryWrites=true&w=majority",
)
MONGO_DB_NAME = os.environ.get("MONGO_DB_NAME", "deepfake_detection")

mongo_client = MongoClient(MONGO_DETAILS)
mongo_db = mongo_client[MONGO_DB_NAME]
users_collection = mongo_db["users"]
checks_collection = mongo_db["checks"]

try:
    users_collection.create_index("email", unique=True)
except errors.OperationFailure:
    pass

IMG_SIZE = 224
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048

sequence_model = keras.models.load_model(MODEL_PATH)
CLASS_VOCAB = ["FAKE", "REAL"]


@app.before_request
def keep_session_active():
    session.permanent = True


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue.")
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)

    return wrapped_view


def persist_check(user_id, payload):
    document = {
        "user_id": user_id,
        **payload,
    }
    checks_collection.insert_one(document)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if "user_id" in session:
        flash("You're already signed in.")
        return redirect(url_for("detect"))

    next_target = request.args.get("next")

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")
        next_target = request.form.get("next") or next_target

        if not email or not password:
            flash("Please provide both email and password.")
            return redirect(url_for("signup"))

        if password != confirm_password:
            flash("Passwords do not match.")
            return redirect(url_for("signup"))

        if users_collection.find_one({"email": email}):
            flash("An account with that email already exists.")
            return redirect(url_for("signup"))

        password_hash = generate_password_hash(password)
        user_doc = {
            "email": email,
            "password_hash": password_hash,
            "created_at": datetime.utcnow(),
        }
        try:
            result = users_collection.insert_one(user_doc)
        except errors.DuplicateKeyError:
            flash("An account with that email already exists.")
            return redirect(url_for("signup"))

        session["user_id"] = str(result.inserted_id)
        session["user_email"] = email
        flash("Account created successfully. You're now signed in.")
        return redirect(next_target or url_for("detect"))

    return render_template("signup.html", next_target=next_target)


@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        flash("You're already signed in.")
        return redirect(url_for("detect"))

    next_target = request.args.get("next")

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        next_target = request.form.get("next") or next_target

        user = users_collection.find_one({"email": email})
        if not user or not check_password_hash(user.get("password_hash", ""), password):
            flash("Invalid email or password.")
            return redirect(url_for("login", next=next_target))

        session["user_id"] = str(user["_id"])
        session["user_email"] = user["email"]
        flash("Welcome back!")
        return redirect(next_target or url_for("detect"))

    return render_template("login.html", next_target=next_target)


@app.route("/logout")
@login_required
def logout():
    session.clear()
    flash("You have been logged out.")
    return redirect(url_for("home"))


@app.route("/detect", methods=["GET", "POST"])
def detect():
    result = None

    if request.method == "POST":
        if "file" not in request.files:
            flash("Please choose a video file to analyze.")
            return redirect(url_for("detect"))

        file = request.files["file"]
        if file.filename == "":
            flash("Please choose a video file to analyze.")
            return redirect(url_for("detect"))

        original_name = secure_filename(file.filename)
        stored_name = f"{uuid.uuid4().hex}_{original_name}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], stored_name)
        file.save(save_path)

        prediction_data = run_prediction(save_path)
        if prediction_data is None:
            flash("We could not detect a clear face in that video. Try another clip.")
            if os.path.exists(save_path):
                os.remove(save_path)
            return redirect(url_for("detect"))

        prediction, probabilities = prediction_data
        descriptor, percent_text = probability_confidence(
            prediction, probabilities
        )

        analyzed_at = datetime.utcnow()
        result = {
            "id": uuid.uuid4().hex,
            "original_filename": original_name,
            "stored_filename": stored_name,
            "prediction": prediction,
            "video_url": url_for("display_video", filename=stored_name),
            "timestamp": analyzed_at.strftime("%d %b %Y • %H:%M UTC"),
            "confidence_descriptor": descriptor,
            "confidence_percent": percent_text,
        }

        if "user_id" in session:
            persist_check(
                session["user_id"],
                {
                    "original_filename": original_name,
                    "stored_filename": stored_name,
                    "prediction": prediction,
                    "confidence_descriptor": descriptor,
                    "confidence_percent": percent_text,
                    "created_at": analyzed_at,
                },
            )
        else:
            history = session.get("history", [])
            history.insert(0, result)
            session["history"] = history[:20]
        flash("Analysis complete.")

    return render_template(
        "detect.html",
        result=result,
        is_authenticated="user_id" in session,
    )


@app.route("/history")
@login_required
def history():
    user_id = session["user_id"]
    checks_cursor = (
        checks_collection.find({"user_id": user_id})
        .sort("created_at", -1)
        .limit(50)
    )
    checks = []
    for entry in checks_cursor:
        created_at = entry.get("created_at")
        checks.append(
            {
                "id": str(entry.get("_id")),
                "original_filename": entry.get("original_filename"),
                "prediction": entry.get("prediction"),
                "confidence_descriptor": entry.get("confidence_descriptor"),
                "confidence_percent": entry.get("confidence_percent"),
                "timestamp": created_at.strftime("%d %b %Y • %H:%M UTC")
                if isinstance(created_at, datetime)
                else "",
                "video_url": url_for(
                    "display_video", filename=entry.get("stored_filename", "")
                ),
            }
        )
    return render_template("history.html", checks=checks)


@app.route("/display/<filename>")
def display_video(filename):
    return redirect(url_for("static", filename=f"uploads/{filename}"), code=301)


def run_prediction(path):
    frames = load_video(path)
    if frames.size == 0:
        return None
    frame_features, frame_mask = prepare_single_video(frames)
    probabilities = sequence_model.predict([frame_features, frame_mask])[0]
    pred_index = int(probabilities.argmax())
    return CLASS_VOCAB[pred_index], probabilities


def probability_confidence(label, probabilities):
    class_index = CLASS_VOCAB.index(label)
    confidence = probabilities[class_index]
    if confidence >= 0.9:
        descriptor = "very high confidence"
    elif confidence >= 0.7:
        descriptor = "high confidence"
    elif confidence >= 0.5:
        descriptor = "moderate confidence"
    else:
        descriptor = "low confidence"
    return descriptor, f"{confidence:.0%}"


def load_video(path, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(int(total_frames / MAX_SEQ_LENGTH), 1)
    frames = []

    try:
        for frame_cntr in range(MAX_SEQ_LENGTH):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_cntr * skip_frames_window)
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_face_center(frame)
            if frame is None:
                continue
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)
    finally:
        cap.release()

    return np.array(frames)


def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH), dtype="bool")
    frame_features = np.zeros(
        shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
    )

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
        frame_mask[i, :length] = True

    return frame_features, frame_mask


def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )

    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")


feature_extractor = build_feature_extractor()


def crop_face_center(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    if len(faces) == 0:
        return None
    x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
    return frame[y : y + h, x : x + w]


if __name__ == "__main__":
    app.run()