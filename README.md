# DCF-OTC: Discrete Codebook Fingerprints for Open-Set Traffic Classification

Official implementation of **DCF-OTC**, a representation-driven method for open-set encrypted traffic classification. DCF-OTC quantizes continuous packet-level encoder representations into position-aware codeword sequences, shifting rejection from sample-level confidence scalars to structured behavioral patterns at packet positions.

> Paper: *DCF-OTC: Discrete Codebook Fingerprints for Open-Set Traffic Classification* (IEEE S&P 2027 submission). Experiments cover seven public datasets across three tasks (encrypted malware detection, TLS 1.3 website identification, mobile application identification); DCF-OTC achieves the highest closed-set macro-F1 on all seven datasets and 78.32% average unknown detection rate, compared with 48.54% for the per-dataset best baselines.

## Repository Structure

```
codebook/
├── vocab_gen/                  # traffic vocabulary construction
│   ├── extractIpTcp.py         # extract per-packet header fields from PCAPs
│   └── vocab_gen_all.py        # build the traffic field vocabulary
├── pre-training/
│   ├── preprocess.py           # build the pre-training dataset
│   └── pretrain.py             # masked-prediction pre-training
├── fine-data/
│   ├── get_finedata.py         # extract fine-tuning flows into JSON
│   └── get_finetsv.py          # convert JSON flows into TSV format
└── fine-tuning/
    └── run_jushi_pos_ood12_center_sep.py
                                  # codebook fine-tuning + open-set inference
                                  # (position embeddings, center & separation
                                  #  losses, two-stage rejection)
```

## Pipeline Overview

### 0. Prepare Raw Datasets

Group PCAP files into flows: organize the directory as `category name / flow name / flow`, i.e. one folder per class, containing one `.pcap` file per flow. 

### 1. Pre-training

Learn general field representations from unlabeled flows with masked field prediction:

```bash
# 1. extract fields from raw PCAPs
python vocab_gen/extractIpTcp.py

# 2. build the traffic vocabulary
python vocab_gen/vocab_gen_all.py

# 3. build the pre-training dataset
python pre-training/preprocess.py

# 4. run pre-training
python pre-training/pretrain.py
```

### 2. Fine-tuning and Open-Set Evaluation

Fine-tune the pre-trained encoder with the position-aware codebook, joint loss (`L_cls + lambda1 * L_ctr + lambda2 * L_sep`, default `lambda1 = lambda2 = 0.01`, separation margin `m = 10`), and two-stage rejection:

```bash
# 1. extract fine-tuning flows into JSON
python fine-data/get_finedata.py

# 2. convert JSON flows into TSV
python fine-data/get_finetsv.py

# 3. fine-tune and evaluate under the open-set protocol
python fine-tuning/run_jushi_pos_ood12_center_sep.py
```

The script performs supervised codeword assignment, EMA codebook updates, threshold calibration on the known-class validation set (95th percentile by default), and reports known-class accuracy / macro-F1, known-class acceptance rate, and unknown detection rate. Unknown classes appear only in the test set.

Note: this code is based on [ET-Bert](https://github.com/linwhitehat/ET-BERT) and [UER-py](https://github.com/dbiir/UER-py). Many thanks to the authors.