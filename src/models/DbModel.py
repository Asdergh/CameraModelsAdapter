import os
import torch as th

from typing import Union
from base import CameraModel
from torch.nn import Module, Parameter
from collections import namedtuple


class DbModel(Module, CameraModel):

    def __init__(self, config: Union[str, dict], train: bool=True) -> None:

        super().__init__()
        self.train = train

        self.config = config
        if isinstance(config, str):
            self.config = config

        self.out_tuple = namedtuple("output", [
            "alpha",
            "ksi",
            "focal"
        ])
        
        self._alpha_ = Parameter(th.tensor(self.config["alpha"]))
        self._ksi_ = Parameter(th.tensor(self.config["ksi"]))
        self._focal_ = Parameter(th.Tensor(self.config["focal_len"]))
        self._camera_center_ = Parameter(th.Tensor(self.config["camera_center"]))

    def project(self, inputs: th.Tensor) -> th.Tensor:
        

        uv_points = (inputs[:, :-1] * self._focal_)
        d1 = th.linalg.norm(inputs, dim=-1)
        d2 = th.sqrt((inputs[:, 0] ** 2 + inputs[:, 1] ** 2) + (self._ksi_ * d1 + inputs[:, -1]) ** 2)
        coeff = 1 / ((self._alpha_ * d2 + (1 - self._alpha_) * (self._ksi_ * d2  + inputs[:, -1])))
        
        uv_points *= coeff.unsqueeze(dim=-1)
        return uv_points
        

        
    def re_project(self, inputs):

        mxy = (inputs - self._camera_center_) / self._focal_
        ru_sqrt = (th.linalg.norm(mxy, dim=-1) ** 2).unsqueeze(dim=-1)


        nume = (1 - (self._alpha_ ** 2) * ru_sqrt)
        denu = (self._alpha_ * th.sqrt(th.abs(1 - ((2 * self._alpha_ - 1) * ru_sqrt)))) + (1 - self._alpha_)
        mz = (nume / (denu))

        points = th.cat([mxy, mz], dim=-1)
        nume = (mz * self._ksi_) + th.sqrt(mz ** 2 + (1 - self._ksi_ ** 2) * ru_sqrt)
        denu = ((mz ** 2) + ru_sqrt) + 1e-5
        coeff = nume * denu
        
        ksi_sh = th.ones(coeff.size()) * self._ksi_
        zeros = th.zeros(coeff.size()) 
        shuffle = th.cat([zeros, zeros, ksi_sh], dim=-1)
        
        
        return (coeff * points) + shuffle
    
    def _forward_(self, inputs: th.Tensor) -> th.Tensor:
        project_2d3d = self.re_project(inputs)
        project_3d2d = self.project(project_2d3d) 
        return (project_3d2d, self.out_tuple(
            self._alpha_,
            self._ksi_,
            self._focal_
        ))
    
    
    def __call__(self, inputs: th.Tensor) -> th.Tensor:

        if not self.train:
            with th.no_grad():
                return self._forward_(inputs)

        else:
            return self._forward_(inputs)
            
    


if __name__ == "__main__":

    config = {
        "alpha": 0.098,
        "ksi": 0.098,
        "camera_center": (64, 64),
        "focal_len": (12.12, 12.12)
    }

    # test = th.normal(0.0, 1.0, (1000, 3))
    # points_test = th.Tensor([0.5, 0.5, 2.0]).unsqueeze(dim=0)
    points_test = th.normal(0.0, 1.0, (10, 2))

    UCM = DbModel(config=config, train=False)
    out_grid = UCM(points_test)
    print(out_grid[0].size())
    # out = UCM.project(points_test)
    # print(out.size())
    # print(out.min(), out.max())
    # reout = UCM.re_project(out)
    # print(reout.max(), reout.min())
    # reout = reout.mean()
    # reout.backward()
    # print(out, reout)
    # loss = th.mean(test - reout)
    # print(loss)
