# GGI-attack
This is the official repository for "[Hijacking Large Language Models via Adversarial In-Context Learning](https://arxiv.org/abs/2311.09948)" by [Yao Qiang](https://qiangyao1988.github.io/), [Xiangyu Zhou](www.linkedin.com/in/xiangyu-zhou-71086321a), [Dongxiao Zhu](https://dongxiaozhu.github.io/)

## Getting Started
### Installation
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
