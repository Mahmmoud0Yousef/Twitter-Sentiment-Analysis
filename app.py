from flask import Flask, request, render_template
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the model and files
model = tf.keras.models.load_model(r"D:\INSTANT(Course)\Training Month\Sprint3\Sentiment Analysis\model_sentiment")

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_len = 25

def preprocess_text(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    return padded

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        text = request.form.get("text")
        if text:
            processed = preprocess_text(text)
            prediction_probs = model.predict(processed)
            predicted_class = np.argmax(prediction_probs, axis=1)
            label = label_encoder.inverse_transform(predicted_class)[0]
            confidence = float(np.max(prediction_probs))
            prediction = f"{label} (Confidence: {confidence:.2f})"
        else:
            prediction = "Please enter some text"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
