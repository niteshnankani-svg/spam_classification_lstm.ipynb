import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# load model
model = load_model("spam_lstm_model.keras")

# load tokenizer
with open("spam_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# load config
with open("spam_config.pkl", "rb") as f:
    config = pickle.load(f)

max_len = config["max_len"]

def predict_message(text):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=max_len)
    prob = model.predict(pad)[0][0]
    label = "spam" if prob > 0.5 else "ham"
    return prob, label

# test
if __name__ == "__main__":
    msg = input("Enter message: ")
    prob, label = predict_message(msg)
    print(f"Spam probability: {prob:.4f}")
    print(f"Prediction: {label}")
