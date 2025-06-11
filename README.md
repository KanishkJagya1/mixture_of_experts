summ_moe/
├── data/                      # raw/downloaded XL-Sum JSON files
├── notebooks/                 # EDA & prototype notebooks
├── src/
│   ├── data/
│   │   ├── loader.py         # loads XLSum via Hugging Face
│   │   ├── preprocess.py     # language tagging, tokenization
│   ├── models/
│   │   ├── experts.py        # defines expert encoders/decoders
│   │   ├── gate.py           # gating network that selects top‑K experts
│   │   ├── moe_encoder.py    # encoder + encoder-MoE
│   │   ├── moe_decoder.py    # decoder + decoder-MoE + final output
│   ├── train.py              # orchestrates training: dataset ➝ forward pass ➝ loss
│   ├── eval.py               # evaluation (ROUGE, language accuracy)
│   ├── streamlit_app.py      # frontend for uploading file & summarizing
├── configs/
│   ├── model.yaml
│   ├── train.yaml
├── requirements.txt
└── README.md
