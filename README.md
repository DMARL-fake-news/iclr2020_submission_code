# Modeling Fake News in Social Networks with Deep Multi-Agent Reinforcement Learning

Code for ICLR2020 submission.

## Summmary
The code is found in src/ and is organized into two sub-folders: bias_attack/ and takeover_attack/. Code in each folder is for their respective attack modes. Core elements of this code are:

- `env.py`: Environment for information aggregation game 
- `q_agent.py`: Q learning agent that implements the recurrent neural network, takes action given observations and updates the network weights to minimize the TD error.
- `trainer.py`: A training routine that takes in parameters, sets up the Q agent, trains the neural network and periodically saves snapshots of the network. The routine also allows for the post training evaluation of a saved network.



## Requirements
Install anaconda.
```
$ conda create -n new_venv python=3.6
$ source activate new_venv
$ pip install -r requirements.txt
```

## Running code
Adjust parameters in train or test scripts provided in run/ to desired parameters. Copy script into folder in src/ that corresponds to desired attack mode. Run script. Note that scripts provided are examples. Scripts to reproduce results can be obtained by setting parameters appropriately.
    
## License
Code licensed under the Apache License v2.0
