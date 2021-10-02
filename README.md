# CEIA RL Course Tournament Starter Kit

## Requirements

- Python 3.8
- [requirements.txt](https://github.com/bryanoliveira/tournament-starter/blob/main/requirements.txt)

## Usage

- Clone this repository
- Run `pip install -r requirements.txt`
- Run `python example_random.py` to watch a random agent play the game
- Run `python example_ray.py` to train a simple self-play agent using [Ray RLLib](https://docs.ray.io/en/latest/rllib.html)
- After training an agent with `example_ray.py`, run `python watch.py --agent1-checkpoint PATH --agent2-checkpoint PATH` to watch your agent play against itself.
