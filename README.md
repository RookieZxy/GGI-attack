# GGI-attack
This is the official repository for "[Hijacking Large Language Models via Adversarial In-Context Learning](https://arxiv.org/abs/2311.09948)" by [Xiangyu Zhou](www.linkedin.com/in/xiangyu-zhou-71086321a), [Yao Qiang](https://qiangyao1988.github.io/), [Dongxiao Zhu](https://dongxiaozhu.github.io/)

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

