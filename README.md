# REAR
REAR is a **RE**levance-**A**ware **R**etrieval-augmented framework for open-domain question answering. [[paper]](https://arxiv.org/abs/2402.17497)

The checkpoint is availible on huggingface🤗. [[checkpoint]](https://huggingface.co/RUCAIBox/rear-llama-7b-hf)

## 🚀 Quick Start

1. Install dependencies.
    ```bash
    pip install -r requirements.txt
    ```

2. Run the following codes:
    ```python
    from rear.src.inf import get_vllm_model
    from rear.src.routing import reliability
    get_vllm_model("RUCAIBox/rear-llama-7b-hf")
    dic = {
        "question": "Who won the first Nobel Prize in Physics",
        "ctxs": ["Wilhelm Conrad Röntgen won first Nobel Prize in Physics."]
    }
    res = reliability(dic)
    print(res['response'])
    ```

## 🔍 Training and Inference Scripts

1. Training.
    ```bash
    bash train.sh meta-llama/Llama-2-7b-hf [your output model dir] [your training data dir] [your deepspeed config file]
    ```

2. Inference.

    First, to generate answers and the path-reliability scores:

    ```bash
    python rear/inference.py \
        --model_path [your output model dir] \
        --phase reliability \
        --source [your test source file] \
        --outfile [your output data dir]
    ```
    Second, to generate the knowledge-consistency scores:
    ```bash
    python rear/inference.py \
        --model_path [your output model dir] \
        --phase consistency \
        --source [your output data dir in reliability-generation phase] \
        --outfile [your output data dir]
    ```


## 🌟 Acknowledgement

Please cite the following paper if you find our code helpful.

```bibtex
@article{wang2024rear,
    title={REAR: A Relevance-Aware Retrieval-Augmented Framework for Open-Domain Question Answering},
    author={Wang, Yuhao and Ren, Ruiyang and Li, Junyi and Zhao, Wayne Xin and Liu, Jing and Wen, Ji-Rong},
    journal={arXiv preprint arXiv:2402.17497},
    year={2024}
}
```