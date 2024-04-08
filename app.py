from tokenizer import tokenize
from cn_tokenizer import tokenize as cn_tokenize
from utils import load_clip_model
from build_index import load_embeddings, transforms
import argparse
import gradio as gr
import numpy as np

languages = {'English': 'en', '中文': 'cn'}
tenc = {}
ds = {}
index = {}
tokenizer = {'en': tokenize, 'cn': cn_tokenize}
INDEX_COLUMN = 'embedding'

def query_images(text, lang, k=4, show_score=False):
    if lang not in languages:
        return []
    l = languages[lang]
    tokens = tokenizer[l](text)
    text_features = tenc[l].infer_new_request(tokens)
    tfeat = text_features.to_tuple()[0]
    tfeat /= np.linalg.norm(tfeat, axis=1, keepdims=True)
    scores, indices = index[l].search(tfeat, k)
    results = []
    if show_score:
        for s, i in zip(scores, indices):
            results.append((ds[l][int(i)]['image'], f"Score: {s:#.3f}"))
    else:
        for i in map(int, indices):
            results.append(ds[l][i]['image'])
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser("OpenCLIP demo: WebUI", add_help=True)
    parser.add_argument('-d', '--device', default='CPU', help='Device to inference')
    parser.add_argument('-m', '--text_model_path', default='models/en/vit_h_14_text.xml', help='Path to text encoder model')
    parser.add_argument('-mc', '--text_model_path_cn', default='models/cn/vit_h_14_text.xml', help='Path to Chinese text encoder model')
    parser.add_argument('-e', '--embeddings_path', default='data/en/vlm.pkl', help='Path to the embeddings pickle file.')
    parser.add_argument('-ec', '--embeddings_path_cn', default='data/cn/vlm.pkl', help='Path to the Chinese embeddings pickle file.')
    parser.add_argument('-n', '--max_queries', default=20, help='Maximum number of queries')
    args = parser.parse_args()
    max_queries = int(args.max_queries)

    _, tenc['en'] = load_clip_model(None, args.text_model_path, args.device)
    _, tenc['cn'] = load_clip_model(None, args.text_model_path_cn, args.device)
    ds['en'] = load_embeddings(args.embeddings_path)
    ds['cn'] = load_embeddings(args.embeddings_path_cn)
    index['en'] = ds['en'].get_index(INDEX_COLUMN)
    index['cn'] = ds['cn'].get_index(INDEX_COLUMN)

    with gr.Blocks() as demo:
        with gr.Row(equal_height=True):
            lang = gr.Dropdown(label="language", choices=[x for x in languages.keys()], value='English', scale=0)
            text = gr.Textbox(label="prompt", scale=1)
            #score = gr.Checkbox(value=False, scale=0)
        k = gr.Slider(minimum=1, maximum=max_queries, value=max_queries, step=1, label="output number")
        btn = gr.Button("Query")
        gallery = gr.Gallery(label="results")
        btn.click(fn=query_images, inputs=[text, lang, k], outputs=gallery)

        demo.launch(server_port=17580, debug=True)
