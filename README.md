# CEIA RL Course Tournament Starter Kit

Example training/testing scripts for our [Soccer-Twos](https://github.com/bryanoliveira/soccer-twos-env) environment.

## Requirements

- Python 3.8
- See [requirements.txt](requirements.txt)

## Usage

- Clone this repository
- Install the requirements: `pip install -r requirements.txt`
- Run `python example_random.py` to watch a random agent play the game
- Run `python example_ray_team_vs_random.py` to train team vs team against a random agent using [Ray RLLib](https://docs.ray.io/en/latest/rllib.html)

You may also run this environment [on Colab](https://colab.research.google.com/drive/1awcOdo8RU9UdaSRKuqUjvaOTF2O17-os?usp=sharing).

### Packing Your Agent

To pack an agent for the target environment you must implement a class that inherits from `soccer_twos.AgentInterface` and implements an `act` method. See `example_player_agent/` or `example_team_agent/` module for reference.

### Testing/Evaluating

Use the environment's rollout tool to test your agent module:

`python -m soccer_twos.watch -m example_player_agent`

You may also run your agent against our [pre-trained baseline (download)](https://drive.google.com/file/d/1WEjr48D7QG9uVy1tf4GJAZTpimHtINzE/view?usp=sharing). Extract the `ceia_baseline_agent` folder to this project's folder and run:

`python -m soccer_twos.watch -m1 example_player_agent -m2 ceia_baseline_agent`
