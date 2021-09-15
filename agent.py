import abc


class Agent(abc.ABC):
    @abc.abstractmethod
    def act(self, observation):
        pass
