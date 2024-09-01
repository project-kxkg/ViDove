from abc import ABC, abstractmethod

"""
This is an abstract class for the evaluators. It is used to define the methods that the model should have.
"""


class AbsApiEvaluator(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def load_model(self) -> None:
        pass

    @abstractmethod
    def evaluate(self, input) -> None:
        pass