# REAR
REAR is a **RE**levance-**A**ware **R**etrieval-augmented framework for open-domain question answering. [[paper]](https://arxiv.org/abs/2402.17497)

The checkpoint is availible on huggingface🤗. [[checkpoint]](https://huggingface.co/RUCAIBox/rear-llama-7b-hf)

The question-and-answer pairs, along with the documents used for inference, are available for download at the following link: [[data]](https://huggingface.co/datasets/yhao-wang/rear-eval).

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
        "question": "Who won the first Noble Prize in Physics",
        "ctxs": [
            "Wilhelm Conrad Röntgen won first Nobel Prize in Physics.",
            "Wilhelm Conrad Röntgen won it for discovery of X-rays",
            "Albert Einstein was awarded the 1921 Nobel Prize in Physics",
            "The Nobel Prize in Physics is a yearly award.",
            "First law of thermodynamics was stated by William"
            ]
        }
    final_answer = reliability(dic)['rely_answer']
    print(final_answer)
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
        --model_path [output model dir] \
        --phase reliability \
        --source [test source file] \
        --outfile [output data dir]
    ```
    Second, to generate the knowledge-consistency scores:
    ```bash
    python rear/inference.py \
        --model_path [your output model dir] \
        --phase consistency \
        --source [your generated path-reliability data dir] \
        --outfile [your output data dir]
    ```
    After running these scripts, if you have provided **"reference"** as the ground truth in the test source data, the **EM** (Exact Match) and **F1** scores will be automatically calculated.


## 🌟 Acknowledgement

Please cite the following paper if you find our code or data helpful.

```bibtex
@article{wang2024rear,
    title={REAR: A Relevance-Aware Retrieval-Augmented Framework for Open-Domain Question Answering},
    author={Wang, Yuhao and Ren, Ruiyang and Li, Junyi and Zhao, Wayne Xin and Liu, Jing and Wen, Ji-Rong},
    journal={arXiv preprint arXiv:2402.17497},
    year={2024}
}
```