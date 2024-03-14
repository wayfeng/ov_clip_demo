# Demo OpenCLIP with OpenVINO

## Setup python virual environment
``` bash
virtualenv .env
pip install -r requirements_app.txt
source .env/bin/activate
```

## Create **embeddings**
``` bash
python build_index.py -d 'GPU' --image_path /path/to/images -m models/vit_h_14_visual.xml -e data/vlm.pkl
```

## Run the app
``` bash
python app.py -d 'GPU'
```
