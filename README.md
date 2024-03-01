# REAR
REAR is a **RE**levance-**A**ware **R**etrieval-augmented framework for open-domain question answering. [[paper]](https://arxiv.org/abs/2402.17497)

## 🚀 Quick Start

1. Install dependencies.
    ```bash
    pip install -r requirements.txt
    ```

2. Training.
    ```bash
    bash train.sh meta-llama/Llama-2-7b-hf [your output model dir] [your training data dir] [your deepspeed config file]
    ```

3. Inference.

    First, to generate answers and the reliability scores:

    ```bash
    python rear/inference.py \
        --model_path [your output model dir] \
        --phase reliability \
        --source [your test source file] \
        --outfile [your output data dir]
    ```
    Second, to generate the knowledge-consistency scores:
    ```
    python rear/inference.py \
        --model_path [your output model dir] \
        --phase consistency \
        --source [your output data dir in reliability-generation phase] \
        --outfile [your output data dir]
    ```


## 🌟 Acknowledgement

Please cite the following paper if you find our code helpful.

```bibtex
@misc{wang2024rear,
      title={REAR: A Relevance-Aware Retrieval-Augmented Framework for Open-Domain Question Answering}, 
      author={Yuhao Wang and Ruiyang Ren and Junyi Li and Wayne Xin Zhao and Jing Liu and Ji-Rong Wen},
      year={2024},
      eprint={2402.17497},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```