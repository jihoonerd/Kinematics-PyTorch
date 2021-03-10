from abc import ABC, abstractclassmethod, abstractmethod


class KinematicModel(ABC):

    @abstractmethod
    def get_kinematic_chain(self):
        pass

    @abstractmethod
    def get_root_pos_rot(self, frame_id: int):
        pass
    
    @abstractmethod
    def get_rotation_matrix(self, frame_id: int, joint_name: str):
        pass