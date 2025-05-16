# GGI-attack
This is the official repository for "[Hijacking Large Language Models via Adversarial In-Context Learning](https://arxiv.org/abs/2311.09948)" by [Xiangyu Zhou](www.linkedin.com/in/xiangyu-zhou-71086321a), [Yao Qiang](https://qiangyao1988.github.io/), [Dongxiao Zhu](https://dongxiaozhu.github.io/)

![Illustration of our attack](Illustration.png)

## Abstract
In-context learning (ICL) has emerged as a powerful paradigm leveraging LLMs for specific downstream tasks by utilizing labeled examples as demonstrations (demos) in the precondition prompts. Despite its promising performance, ICL suffers from instability with the choice and arrangement of examples. Additionally, crafted adversarial attacks pose a notable threat to the robustness of ICL. However, existing attacks are either easy to detect, rely on external models, or lack specificity towards ICL. This work introduces a novel transferable attack against ICL to address these issues, aiming to hijack LLMs to generate the target response or jailbreak. Our hijacking attack leverages a gradient-based prompt search method to learn and append imperceptible adversarial suffixes to the in-context demos without directly contaminating the user queries. Comprehensive experimental results across different generation and jailbreaking tasks highlight the effectiveness of our hijacking attack, resulting in distracted attention towards adversarial tokens and consequently leading to unwanted target outputs. We also propose a defense strategy against hijacking attacks through the use of extra clean demos, which enhances the robustness of LLMs during ICL. Broadly, this work reveals the significant security vulnerabilities of LLMs and emphasizes the necessity for in-depth studies on their robustness.

## Getting Started
We use the newest version of PyEnchant and FastChat. These two packages can be installed by running the following command:
```bash
pip3 install pyenchant "fschat[model_worker,webui]"
```

When you install PyEnchant, it typically requires the Enchant library to be installed on your system. you can install it using the following command:
```bash
sudo apt-get install libenchant1c2a
```

To track the loss during the demonstration, we utilize the livelossplot library. Therefore, it's recommended to install this library using pip before proceeding.
```bash
pip install livelossplot
```

## Running the Code
The script to run the baseline method mentioned in our paper is in <kbd style="background-color: #f2f2f2;">/Baseline-attack/scripts/run_text_exp.py</kbd>.

You can also find our method(GGI) in the path <kbd style="background-color: #f2f2f2;">/GGI-attack/demo.ipynb</kbd>.

We additionally provide several demos and queries located at the path /dataset for running the code

## Citation
```bash
@article{qiang2023hijacking,
  title={Hijacking large language models via adversarial in-context learning},
  author={Qiang, Yao and Zhou, Xiangyu and Zhu, Dongxiao},
  journal={arXiv preprint arXiv:2311.09948},
  year={2023}
}
```

