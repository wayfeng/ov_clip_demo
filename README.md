# Demo OpenCLIP with OpenVINO

## Setup python virual environment
``` bash
virtualenv .env
pip install -r requirements_app.txt
source .env/bin/activate
```

## Prepare models

Please refer to repository [ov_clip](https://github.com/wayfeng/ov_clip.git) on how to convert CLIP models to OpenVINO IRs.

Put open CLIP and Chinese CLIP models as below. Otherwise you'll have to specify path to models when running both *build_index.py* and *app.py*.

``` bash
tree models
models
├── cn
│   ├── vit_h_14_text.bin
│   ├── vit_h_14_text.xml
│   ├── vit_h_14_visual.bin
│   └── vit_h_14_visual.xml
└── en
    ├── vit_h_14_text.bin
    ├── vit_h_14_text.xml
    ├── vit_h_14_visual.bin
    └── vit_h_14_visual.xml
```

## Create **embeddings**
``` bash
mkdir -p data/{cn,en}
python build_index.py -d 'GPU' -i /path/to/images -e data/en/embeddings.pkl -ec data/cn/embeddings.pkl
```

## Run the app
``` bash
python app.py -d 'GPU' -e data/cn/embeddings.pkl -ec data/cn/embeddings.pkl
```
