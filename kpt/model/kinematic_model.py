from abc import ABC, abstractclassmethod, abstractmethod


class KinematicModel(ABC):

    @abstractmethod
    def get_kinematic_chain(self):
        pass

    @abstractmethod
    def get_root_pos_rot(self, frame_id: int):
        pass