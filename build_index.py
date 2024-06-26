import argparse
import faiss
import numpy as np
import pickle
from datasets import load_dataset
from utils import load_clip_model, preprocess_image

INDEX_COLUMN = 'embedding'

def transforms(items):
    items['img'] = [preprocess_image(image) for image in items['image']]
    return items

def build_embeddings(image_path, image_encoder_path, device='CPU'):
    ds = load_dataset('imagefolder', data_dir=image_path, split='train')
    ds.set_transform(transforms)
    ie, _ = load_clip_model(image_encoder_path, '', device, throughputmode=True)
    ireq = ie.create_infer_request()
    def image_embedding(x):
        embedding = ireq.infer({'x': x['img'][None]}).to_tuple()[0]
        embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)
        return {INDEX_COLUMN: embedding[0]}
    ds_emb = ds.map(image_embedding)
    return ds_emb

def load_embeddings(embeddings_path):
    with open(embeddings_path, 'rb') as f:
        ds = pickle.load(f)
        return ds

def save_embeddings(device, image_path, image_model_path, embeddings_path):
    embeddings = build_embeddings(image_path, image_model_path, device)
    embeddings.add_faiss_index(column='embedding', metric_type=faiss.METRIC_INNER_PRODUCT)
    with open(embeddings_path, 'wb') as f:
        pickle.dump(embeddings, f, protocol=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("OpenCLIP demo: build faiss index", add_help=True)
    parser.add_argument('-d', '--device', default='CPU', help='Device to inference')
    parser.add_argument('-i', '--image_path', default='images', help='Path to images')
    parser.add_argument('-m', '--image_model_path', default='models/en/vit_h_14_visual.xml', help='Path to image encoder model')
    parser.add_argument('-mc', '--image_model_path_cn', default='models/cn/vit_h_14_visual.xml', help='Path to Chinese image encoder model')
    parser.add_argument('-e', '--embeddings_path', default='data/en/embeddings.pkl', help='Path to the embeddings pickle file.')
    parser.add_argument('-ec', '--embeddings_path_cn', default='data/cn/embeddings.pkl', help='Path to the Chinese embeddings pickle file.')
    args = parser.parse_args()
    save_embeddings(args.device, args.image_path, args.image_model_path, args.embeddings_path)
    save_embeddings(args.device, args.image_path, args.image_model_path_cn, args.embeddings_path_cn)

