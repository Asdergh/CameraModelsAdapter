import torch as th
import os

from typing import Union
from base import CameraModel
from torch.nn import Module, Parameter



class UcModel(Module, CameraModel):

    def __init__(self, config: Union[str, dict]) -> None:

        super().__init__()

        self.config = config
        if isinstance(config, str):
            self.config = config

        self._alpha_ = Parameter(th.tensor(self.config["alpha"]))
        self.f_x, self.f_y = self.config["focal_len"]
        self.c_x, self.c_y = self.config["camera_center"]

    def project(self, inputs: th.Tensor) -> th.Tensor:

        d = th.linalg.norm(inputs, axis=-1)

        u = (inputs[:, 0] * self.f_x).unsqueeze(dim=-1)
        v = (inputs[:, 1] * self.f_y).unsqueeze(dim=-1)
        coeffs = (1.0 / ((self._alpha_ * d) + ((1 - self._alpha_) * inputs[:, -1]))).unsqueeze(dim=-1)

        plane_points = th.cat([
            u * coeffs,
            v * coeffs
        ], dim=-1)

        shuffle = th.Tensor([self.c_x, self.c_y]).unsqueeze(dim=0).repeat(inputs.size()[0], 1)
        return  plane_points + shuffle

    def re_project(self, inputs):

        mx = (1 - self._alpha_) * ((inputs[:, 0] - self.c_x) / self.f_x).unsqueeze(dim=-1)
        my = (1 - self._alpha_) * ((inputs[:, 1] - self.c_y) / self.f_y).unsqueeze(dim=-1)
        ones = th.ones(mx.size())
        points = th.cat([mx, my, ones], dim=-1)
        
        ru_sqrt = (mx ** 2) + (my ** 2)
        ksi = self._alpha_ / (1 - self._alpha_)
        coeffs = ((ksi + th.sqrt(1 + (1 - ksi**2)*ru_sqrt)) / (1 + ru_sqrt))
        return (coeffs * points) + th.cat([
            th.zeros(coeffs.size()), 
            th.zeros(coeffs.size()), 
            th.ones(coeffs.size()) * ksi
        ], dim=-1)


    def __call__(self, inputs: th.Tensor) -> th.Tensor:

        project_2d3d = self.re_project(inputs)
        project_3d2d = self.project(project_2d3d)   
        return project_3d2d


if __name__ == "__main__":
    
    config = {
        "alpha": 0.0,
        "camera_center": (320, 340),
        "focal_len": (500, 500)
    }

    # test = th.normal(0.0, 1.0, (1000, 3))
    points_test = th.Tensor([0.5, 0.5, 2.0]).unsqueeze(dim=0)

    UCM = UcModel(config=config)
    out = UCM.project(points_test)
    reout = UCM.re_project(out)
    reout = reout.mean()
    reout.backward()
    print(out, reout)
    # loss = th.mean(test - reout)
    # print(loss)
    
        

        
    
    