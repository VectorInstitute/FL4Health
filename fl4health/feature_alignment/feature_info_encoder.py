from abc import ABC, abstractmethod


class FeatureInfoEncoder(ABC):
    @abstractmethod
    def to_json(self) -> str:
        raise NotImplementedError

    @abstractmethod
    @staticmethod
    def from_json(json_str: str) -> "FeatureInfoEncoder":
        raise NotImplementedError
