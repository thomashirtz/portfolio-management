# portfolio-management

Framework to train agents on a dynamic portfolio management
and allocation environment.
The repository contains three main modules.
- Data management module, used to create MySQL databases using SQLAlchemy and binance-API
- Environment module, that created cryptocurrency market situations using the created databases.
- The Agent module, which try to learn policies on the created environment.

The Soft Actor-Critic is a custom agent made to be able to receive several streams
of data coming from the different cryptocurrencies. It uses an Ensemble of Identical 
Independent Evaluators (EIIE) strategy in order to efficiently learning from the market.

# Sources
[A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem, Zhengyao & al., 2017](https://arxiv.org/abs/1706.10059v2)  
[Adversarial Deep Reinforcement Learning in Portfolio Management, Zhipeng & al., 2018](https://arxiv.org/abs/1808.09940v3)

# Requirements
pandas  
sqlalchemy  
xarray  
gym  
numpy  
scipy  
python-binance  
torch  
torchsummary  

# License
This repository is licensed under the terms of the Apache 2.0 License.
See [LICENSE](/LICENSE) for more information.
