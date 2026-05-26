CACHE_DIR="cache/"
mkdir -p "$CACHE_DIR/datasets/lm-bff/"
mkdir -p "$CACHE_DIR/models/lm-bff/"

# aria2c -x 16 -s 16 -d "$CACHE_DIR/datasets" https://nlp.cs.princeton.edu/projects/lm-bff/datasets.tar
tar xvf "$CACHE_DIR/datasets/datasets.tar" -C "$CACHE_DIR/datasets/"
python scripts/generate.py --seed 42

python scripts/download_model.py --model roberta-base
python scripts/download_model.py --model legal
python scripts/download_model.py --model mental
python scripts/download_model.py --model codebert
python scripts/download_model.py --model 4m
# gdown --folder "https://drive.google.com/drive/folders/1liy0pWMrLzpa9rZ3TBRkfOw-NtMoksLg" -O "$CACHE_DIR/models/lm-bff/"

