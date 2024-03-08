from tokenizer import tokenize
from utils import load_clip_model, preprocess_image
from build_index import load_embeddings, transforms
import gradio as gr
import numpy as np
import os

model_id = os.environ.get('MODEL_ID') or 'ViT-H-14'
#img_encoder_path = f"./models/{model_id.replace('-', '_').lower()}_visual.xml"
txt_encoder_path = f"./models/{model_id.replace('-', '_').lower()}_text.xml"
device = os.environ.get('DEVICE') or 'GPU'
embeddings_path = os.environ.get('EMBEDDINGS') or "./vlm_small.pkl"

#_ienc, _tenc = load_clip_model(img_encoder_path, txt_encoder_path, device)
_, tenc = load_clip_model(None, txt_encoder_path, device)

ds = load_embeddings(embeddings_path)
index = ds.get_index('embedding')

def query_images(text, k=4):
    texts = text.split(',')
    tokens = tokenize(texts)
    text_features = tenc.infer_new_request(tokens)
    tfeat = text_features.to_tuple()[0]
    tfeat /= np.linalg.norm(tfeat, axis=1, keepdims=True)
    _, indices = index.search(tfeat, int(k))
    results = []
    for i in indices:
        results.append(ds[int(i)]['image'])
    return results

with gr.Blocks() as demo:
    with gr.Column():
        text = gr.Textbox(label="prompt")
        k = gr.Slider(minimum=1, maximum=20, step=1, label="output number")
    btn = gr.Button("Query")
    gallery = gr.Gallery(label="results")
    btn.click(fn=query_images, inputs=[text, k], outputs=gallery)

try:
    demo.launch(debug=True)
except Exception:
    demo.launch(share=True, debug=True)
