import os
import json, yaml
import torch as th
from abc import ABC, abstractmethod
from typing import Union

_read_io_ = {
    "json": json.load,
    "yaml": yaml.safe_load
}
class CameraModel(ABC):

    @abstractmethod
    def project(self, inputs: th.Tensor) -> th.Tensor:
        """
        This method projects points from 3D space
        into image plane due to model notation
        
        As inputs you must provide torch tenosr with size (N, 3)
        which represents the points to project into the image
        plane
        """
        pass

    @abstractmethod
    def re_project(self, inputs: th.Tensor) -> th.Tensor:
        """
        This function re_projects points from 2D image
        plane into 3D space due to model notation

        As inputs you must provide torch tensor with size (N, 2)
        which represents the (u, v) coordinates of pixels into the
        image plane
        """
    
    def _read_config_(self, config: str) -> dict:

        f_type = config.split(".")[1]
        with open(config, "r") as file:
            config = _read_io_[f_type](file)
        
        return config
    

    
