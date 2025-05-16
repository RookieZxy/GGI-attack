# GGI-attack
This is the official repository for "[Hijacking Large Language Models via Adversarial In-Context Learning](https://arxiv.org/abs/2311.09948)" by [Xiangyu Zhou](www.linkedin.com/in/xiangyu-zhou-71086321a), [Yao Qiang](https://qiangyao1988.github.io/), [Dongxiao Zhu](https://dongxiaozhu.github.io/)

![Illustration of our attack](Illustration.png)

## Abstract
In-context learning (ICL) has emerged as a powerful paradigm leveraging LLMs for specific downstream tasks by utilizing labeled examples as demonstrations (demos) in the preconditioned prompts. Despite its promising performance, crafted adversarial attacks pose a notable threat to the robustness of LLMs. Existing attacks are either easy to detect, require a trigger in user input, or lack specificity towards ICL. To address these issues, this work introduces a novel transferable prompt injection attack against ICL, aiming to hijack LLMs to generate the target output or elicit harmful responses. In our {\textit{threat model}}, the hacker acts as a model publisher who leverages a gradient-based prompt search method to learn and append imperceptible adversarial suffixes to the in-context demos via prompt injection. We also propose effective defense strategies using a few shots of clean demos, enhancing the robustness of LLMs during ICL. Extensive experimental results across various classification and jailbreak tasks demonstrate the effectiveness of the proposed attack and defense strategies. This work highlights the significant security vulnerabilities of LLMs during ICL and underscores the need for further in-depth studies. Our code is available at: 

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

