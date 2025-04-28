import torch as th
import os

from typing import Union
from base import CameraModel
from torch.nn import Module, Parameter



class EucModel(Module, CameraModel):

    def __init__(self, config: Union[str, dict]) -> None:

        super().__init__()

        self.config = config
        if isinstance(config, str):
            self.config = config

        self._alpha_ = Parameter(th.tensor(self.config["alpha"]))
        self._beta_ = Parameter(th.tensor(self.config["beta"]))
        self.f_x, self.f_y = self.config["focal_len"]
        self.c_x, self.c_y = self.config["camera_center"]

    def project(self, inputs: th.Tensor) -> th.Tensor:

        d = th.sqrt(self._beta_ * ((inputs[:, 0] ** 2) + (inputs[:, 1] ** 2)) + (inputs[:, -1] ** 2))
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

        mx = ((inputs[:, 0] - self.c_x) / self.f_x).unsqueeze(dim=-1)
        my = ((inputs[:, 1] - self.c_y) / self.f_y).unsqueeze(dim=-1)
        ru_sqrt = (mx ** 2) + (my ** 2)

        nume = (1 - self._beta_ * (self._alpha_ ** 2) * ru_sqrt)
        denu = (self._alpha_ * th.sqrt(th.abs(1 - ((2 * self._alpha_ - 1) * (self._beta_ * ru_sqrt))))) + (1 - self._alpha_)
        mz = (nume / (denu))

        points = th.cat([mx, my, mz], dim=-1)
    
        return (1 / th.linalg.norm(points)) * points


    def __call__(self, inputs: th.Tensor) -> th.Tensor:

        project_2d3d = self.re_project(inputs)
        project_3d2d = self.project(project_2d3d)   
        return project_3d2d



if __name__ == "__main__":

    config = {
        "alpha": 0.0,
        "beta": 0.0,
        "camera_center": (320, 340),
        "focal_len": (500, 500)
    }

    # test = th.normal(0.0, 1.0, (1000, 3))
    points_test = th.Tensor([0.5, 0.5, 2.0]).unsqueeze(dim=0)

    UCM = EucModel(config=config)
    out = UCM.project(points_test)
    print(out.size(), out)
    reout = UCM.re_project(out)
    reout = reout.mean()
    reout.backward()
    print(out, reout)
    # loss = th.mean(test - reout)
    # print(loss)