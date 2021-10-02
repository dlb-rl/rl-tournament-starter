# CEIA RL Course Tournament Starter Kit

Example training/testing scripts for our [Soccer-Twos](https://github.com/bryanoliveira/soccer-twos-env) environment.

## Requirements

- Python 3.8
- [requirements.txt](https://github.com/bryanoliveira/tournament-starter/blob/main/requirements.txt)

## Usage

- Clone this repository
- Install the requirements: `pip install -r requirements.txt`
- Run `python example_random.py` to watch a random agent play the game
- Run `python example_ray.py` to train a simple self-play agent using [Ray RLLib](https://docs.ray.io/en/latest/rllib.html)
- After training an agent with `example_ray.py`, run `python watch.py --agent1-checkpoint PATH --agent2-checkpoint PATH` to watch your agent play against itself.

You may also run this environment [on Colab](https://colab.research.google.com/drive/1awcOdo8RU9UdaSRKuqUjvaOTF2O17-os?usp=sharing).
