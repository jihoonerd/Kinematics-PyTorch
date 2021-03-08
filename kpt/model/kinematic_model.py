from abc import ABC, abstractclassmethod, abstractmethod


class KinematicModel(ABC):

    @abstractmethod
    def get_kinematic_chain(self):
        pass