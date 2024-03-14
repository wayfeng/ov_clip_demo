from tokenizer import tokenize
from utils import load_clip_model
from build_index import load_embeddings, transforms
import argparse
import gradio as gr
import numpy as np

def query_images(text, k=4):
    tokens = tokenize(text)
    text_features = tenc.infer_new_request(tokens)
    tfeat = text_features.to_tuple()[0]
    tfeat /= np.linalg.norm(tfeat, axis=1, keepdims=True)
    _, indices = index.search(tfeat, k)
    results = []
    for i in map(int, indices):
        results.append(ds[i]['image'])
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser("OpenCLIP demo: WebUI", add_help=True)
    parser.add_argument('-d', '--device', default='CPU', help='Device to inference')
    parser.add_argument('-m', '--text_model_path', default='models/vit_h_14_text.xml', help='Path to text encoder model')
    parser.add_argument('-e', '--embeddings_path', default='data/vlm.pkl', help='Path to the embeddings pickle file.')
    parser.add_argument('-n', '--max_queries', default=20, help='Maximum number of queries')
    args = parser.parse_args()
    txt_encoder_path = args.text_model_path
    embeddings_path = args.embeddings_path
    device = args.device
    max_queries = args.max_queries

    _, tenc = load_clip_model(None, txt_encoder_path, device)
    ds = load_embeddings(embeddings_path)
    index = ds.get_index('embedding')

    with gr.Blocks() as demo:
        with gr.Column():
            text = gr.Textbox(label="prompt")
            k = gr.Slider(minimum=1, maximum=max_queries, step=1, label="output number")
        btn = gr.Button("Query")
        gallery = gr.Gallery(label="results")
        btn.click(fn=query_images, inputs=[text, k], outputs=gallery)

    try:
        demo.launch(server_port=17580, debug=True)
    except Exception:
        demo.launch(server_port=17580, share=True, debug=True)
