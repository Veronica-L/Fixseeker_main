# Fixseeker

## Introduction

---

To address the challenging problem of detecting silent vulnerability fixes in open source software, we first conduct an empirical study on the various type of correlations in multiple hunks. Based on our findings, we propose Fixseeker, a graph-based approach that extracts the various correlations between code changes at the hunk level to detect silent vulnerability fixes. 

## **Prerequisites**

---

- Python environment: Python 3.8
- JDK 21
- Joern 2.0.1
- pip install -r requirements.txt

Note: You can install Joern from github  [https://github.com/joernio/joern](https://github.com/joernio/joern)

## Empirical Study

---

After the manual analysis, we summarize four main hunk correlations: Caller-callee dependency, data flow dependency, control flow dependency and pattern replication. The document of manual analysis is in `empirical/data.csv` .

## Dataset

---

The dataset is available at: [https://drive.google.com/file/d/1TUsX9KQ6mm42VeAMZ4A8arThtm8FLW9Y](https://drive.google.com/file/d/1TUsX9KQ6mm42VeAMZ4A8arThtm8FLW9Y/view?usp=drive_link) , Please download and put dataset inside the data folder.

## Replication

---

1. **Extract hunk correlation**

```bash
#Please run:
python correlation.py --dataset_path {dataset_path} --program_language {pl} --balance_type {type}
#such as
python correlation.py --dataset_path data_test.json --program_language c --balance_type  imbalance
```

The correlation result will be stored in `data/correlation/{PROGRAM_LANGUAGE}_{BALANCE_TYPE}.json`

1. **Train the graph-based model**

```bash
#Please run:
python train.py --program_language {pl} --balance_type {type}
#such as:
python train.py --program_language c --balance_type imbalance
```

1. **RQ4: Performance of Fixseeker**

```bash
#Please run:
python evaluate.py --program_language {pl} --balance_type {type} --model_path {model} --data_path {data_path}
#such as:
python evaluate.py --program_language c --balance_type imbalance --model_path model.pt --data_path c_imbalance_test.pt
```

1. **RQ5: Vulnerability Type**

```bash
#Please run:
python evaluate_cwe.py --program_language {pl} --model_path {model} --cwe {cwe_number}
#such as:
python evaluate_cwe.py --program_language c --model_path model.pt --cwe 125
```

1. **RQ6: Feature Analysis**

```bash
#Please run:
python ablation/train.py --program_language {pl} --balance_type {type} --ablation_type {no_edges}
#such as:
python ablation/train.py --program_language c --balance_type imbalance --ablation_type no_edges
```

Note: the `ablation_type` can choose: `no_edges`, `no_call`, `no_ddg`, `no_cfg`, `no_sim`. The evaluation of ablation study can refer to RQ1.