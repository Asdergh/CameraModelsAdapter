import torch as th
import os

from tqdm import tqdm
from typing import Union
from base import CameraModel
from torch.nn import (
    Module, 
    Parameter, 
    Linear,
    MSELoss
)
from torch.optim import Adam



class KbModel(Module, CameraModel):

    def __init__(self, config: Union[str, dict]) -> None:

        super().__init__()

        self.config = config
        if isinstance(config, str):
            self.config = config

        self._k_vector_ = Parameter(th.Tensor([1, ] + [k_coeff for k_coeff in self.config["k_coeffs"]]))
        self.f_x, self.f_y = self.config["focal_len"]
        self.c_x, self.c_y = self.config["camera_center"]
        

    def d_theta_inv(
        self, 
        r: th.Tensor, 
        theta: th.Tensor=None, 
        steps: int=100, 
        lr: float=0.01, 
        verbose=True
    ) -> th.Tensor:

        theta = theta
        loss_mse = MSELoss()
        if theta is None:
            theta = th.zeros(r.size(), requires_grad=True)

        print(theta.size())
        optim = Adam(params=[theta], lr=lr)
        gen = range(steps)    
        if verbose:
            gen = tqdm(
                range(steps),
                colour="RED",
                ascii="=>",
                desc="Calculating Inv Theta"
            )
        
        for _ in gen:
            
            optim.zero_grad()

            theta_deg = th.stack([theta ** i for i in range(self._k_vector_.size()[0])], dim=-1)
            r_theta = th.stack([th.dot(theta_deg_s, self._k_vector_) for theta_deg_s in theta_deg], dim=-1)
            loss = loss_mse(r_theta, r)

            loss.backward()
            optim.step()
        
        return theta
            
    def project(self, inputs: th.Tensor) -> th.Tensor:
        
        r = th.linalg.norm(inputs[:, :-1], dim=-1)
        theta = th.atan2(r, inputs[:, -1])
        theta_deg = th.stack([
            theta ** i 
            for i in range(self._k_vector_.size()[0])
        ], dim=-1)
        d_theta = th.stack([
            th.dot(theta_deg_s, self._k_vector_) 
            for theta_deg_s in theta_deg 
        ], dim=-1)

        u = (d_theta * inputs[:, 0] * self.f_x).unsqueeze(dim=-1)
        v = (d_theta * inputs[:, 1] * self.f_y).unsqueeze(dim=-1)
        plane_points = th.cat([u, v], dim=-1)

        shuffle = th.Tensor([self.c_x, self.c_y]).unsqueeze(dim=0).repeat(inputs.size()[0], 1)
        return  plane_points + shuffle

    def re_project(self, inputs):

        mx = ((inputs[:, 0] - self.c_x) / self.f_x).unsqueeze(dim=-1)
        my = ((inputs[:, 1] - self.c_y) / self.f_y).unsqueeze(dim=-1)
        ru_sqrt = th.sqrt((mx ** 2) + (my ** 2))
        
        
        inv_theta = self.d_theta_inv(r=ru_sqrt.squeeze(dim=-1))
        sin = th.sin(inv_theta).unsqueeze(dim=-1)
        cos = th.cos(inv_theta).unsqueeze(dim=-1)

        points = th.cat([
            sin * (mx / ru_sqrt),
            sin * (my / ru_sqrt),
            cos
        ], dim=-1)
       
        return points
    
    def __call__(self, inputs: th.Tensor) -> th.Tensor:

        project_2d3d = self.re_project(inputs)
        project_3d2d = self.project(project_2d3d)   
        return project_3d2d
    


if __name__ == "__main__":

    config = {
        "k_coeffs": [1, 1, 1, 1],
        "camera_center": (320, 340),
        "focal_len": (500, 500)
    }

    # test = th.normal(0.0, 1.0, (1000, 3))
    points_test = th.Tensor([
        [0.5, 0.5, 2.0],
        [0.5, 0.5, 2.0]
    ])

    UCM = KbModel(config=config)
    out = UCM.project(points_test)
    reout = UCM.re_project(out)
    reout = reout.mean()
    out = out.mean()

    # out.backward()
    # reout.backward()
    print(out, reout)
    # loss = th.mean(test - reout)
    # print(loss)