from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import pickle

app = Flask(__name__)

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path="chatbot_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Load label to answer mapping
with open("label_to_answer.pkl", "rb") as f:
    label_to_answer = pickle.load(f)


def preprocess_text(text):
    sequences = tokenizer.texts_to_sequences([text])
    maxlen = input_details[0]['shape'][1]
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=maxlen, padding='post')
    return padded


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    input_data = preprocess_text(text)
    interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_index = np.argmax(output_data, axis=1)[0]
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]
    answer = label_to_answer.get(predicted_label, "Maaf, saya tidak mengerti.")

    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
