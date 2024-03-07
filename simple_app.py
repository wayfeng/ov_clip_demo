from scipy.special import softmax
from tokenizer import tokenize
from utils import preprocess_image, load_clip_model
import gradio as gr
import numpy as np
import pandas as pd

device = 'GPU'
model_id = 'ViT-H-14'
img_encoder_path = f"./models/{model_id.replace('-', '_').lower()}_visual.xml"
txt_encoder_path = f"./models/{model_id.replace('-', '_').lower()}_text.xml"

def classify(image_pil, text):
    image = preprocess_image(image_pil)
    texts = text.split(',')
    tokens = tokenize(texts)
    image_features = _ienc.infer_new_request({"x": image[None]})
    text_features = _tenc.infer_new_request(tokens)
    ifeat = image_features.to_tuple()[0]
    tfeat = text_features.to_tuple()[0]
    ifeat /= np.linalg.norm(ifeat, axis=1, keepdims=True)
    tfeat /= np.linalg.norm(tfeat, axis=1, keepdims=True)

    probs = softmax(100 * ifeat @ tfeat.T)
    data = pd.DataFrame({"text": texts, "prob": [x for x in probs[0]]})
    #print(data)
    return gr.BarPlot(
        data,
        x="text",
        y="prob",
        title="Results",
        tooltip=["text", "prob"],
        y_lim=[0.0, 1.0],
        width=400,
    )

#_tokenizer = SimpleTokenizer()
_ienc, _tenc = load_clip_model(img_encoder_path, txt_encoder_path, device)

demo = gr.Interface(
    classify,
    [
        gr.Image(label="Image", type="pil"),
        gr.Textbox(label="Labels", info="Comma-seperated list of class labels"),
    ],
    gr.BarPlot(),
    examples=[["./assets/cat_dog.jpeg", "cat,dog,deer"]]
)

try:
    demo.launch(debug=True)
except Exception:
    demo.launch(share=True, debug=True)
