import torch as th
import os

from typing import Union
from base import CameraModel
from torch.nn import Module, Parameter



class PinHoleModel(Module, CameraModel):

    def __init__(self, config: Union[str, dict]) -> None:

        super().__init__()

        self.config = config
        if isinstance(config, str):
            self.config = config

        self.f_x, self.f_y = self.config["focal_len"]
        self.c_x, self.c_y = self.config["camera_center"]

    def project(self, inputs: th.Tensor) -> th.Tensor:

        u = (inputs[:, 0] * self.f_x).unsqueeze(dim=-1)
        v = (inputs[:, 1] * self.f_y).unsqueeze(dim=-1)
        plane_points = th.cat([u, v], dim=-1)

        shuffle = th.Tensor([self.c_x, self.c_y]).unsqueeze(dim=0).repeat(inputs.size()[0], 1)
        return  (plane_points + shuffle).to(th.int32)

    def re_project(self, inputs):

        mx = ((inputs[:, 0] - self.c_x) / self.f_x).unsqueeze(dim=-1)
        my = ((inputs[:, 1] - self.c_y) / self.f_y).unsqueeze(dim=-1)
        ones = th.ones(mx.size())
        points = th.cat([mx, my, ones], dim=-1)

        
        return (1 / th.linalg.norm(points, dim=-1)).unsqueeze(dim=-1) * points


    def __call__(self, inputs: th.Tensor) -> th.Tensor:

        project_2d3d = self.re_project(inputs)
        project_3d2d = self.project(project_2d3d)   
        return project_3d2d
    
if __name__ == "__main__":
    
    config = {
        "alpha": 0.0,
        "camera_center": (64, 64),
        "focal_len": (500, 500)
    }

    # test = th.normal(0.0, 1.0, (1000, 3))
    points_test = th.Tensor([0.5, 0.5, 2.0]).unsqueeze(dim=0)
    u_in, v_in = th.meshgrid(
        th.arange(0, 128),
        th.arange(0, 128)
    )
    points = th.flatten(th.stack([u_in, v_in], dim=-1), start_dim=0, end_dim=1)
    UCM = UcModel(config=config)
    reout = UCM.re_project(points)
    out = UCM.project(reout).view(128, 128, 2)

    print(th.min(out))
    print(out.size(), reout.size())
    # loss = th.mean(test - reout)
    # print(loss)
    
        

        
    
    