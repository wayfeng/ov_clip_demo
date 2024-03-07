# ov_clip_demo

## Create **embeddings**
``` bash
python build_index.py --device 'GPU' --image_path /path/to/images --image_model_path /path/to/models/vit_h_14_visual.xml --embeddings_path /path/to/embeddings.pkl
```

## Run the app
``` bash
python app.py
```
