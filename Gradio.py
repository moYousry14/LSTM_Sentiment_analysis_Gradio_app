import gradio as gr
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import datetime

model = tf.keras.models.load_model("sentiment_model.h5")

with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

maxlen = 100
history = []  #

def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=maxlen)
    pred = model.predict(padded)[0][0]

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if pred > 0.5:
        result = f"😊 Positive ({pred:.2f})"
    else:
        result = f"😠 Negative ({1 - pred:.2f})"
    
    # أضف للتاريخ
    history.append(f"[{timestamp}] {text.strip()} → {result}")
    log = "\n\n".join(reversed(history[-5:]))  
    
    return result, log

# واجهة Gradio
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🍽️ Restaurant Review Sentiment Analyzer")
    gr.Markdown("Enter a review and get the sentiment prediction from our LSTM model.")
    
    with gr.Row():
        input_text = gr.Textbox(lines=3, label="Your Review")
        output_text = gr.Textbox(label="Sentiment Prediction")

    with gr.Row():
        predict_btn = gr.Button("Predict")
        clear_btn = gr.Button("Clear")

    history_box = gr.Textbox(label="🕓 Last Predictions Log", lines=6, interactive=False)

    predict_btn.click(fn=predict_sentiment, inputs=input_text, outputs=[output_text, history_box])
    clear_btn.click(fn=lambda: ("", "", ""), inputs=[], outputs=[input_text, output_text, history_box])

demo.launch()
