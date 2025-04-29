import torch as th
import os
import matplotlib.pyplot as plt
plt.style.use("dark_background")

from typing import Union
from base import CameraModel
from torch.nn import (
    Module, 
    Parameter, 
    functional as F
)
from collections import namedtuple
from torchvision.transforms import Resize
from torchvision.io import read_image


class DbModel(Module, CameraModel):

    def __init__(
        self,
        c_x: float, c_y: float,
        f_x: float=None, f_y: float=None,
        alpha: float=None, ksi: float=None,
        train: bool=True
    ) -> None:

        super().__init__()
        self.params_collection = namedtuple("ModelParams", [
            "f_x", "f_y",
            "c_x", "c_y",
            "alpha", "ksi"
        ])
        self.params = self.params_collection(
            th.tensor(f_x), th.tensor(f_y),
            th.tensor(c_x), th.tensor(c_y),
            th.tensor(alpha), th.tensor(ksi)
        )
        
    def project(self, inputs: th.Tensor) -> th.Tensor:
        
        d1 = th.linalg.norm(inputs, dim=-1)
        d2 = th.sqrt((inputs[..., 0] ** 2 + inputs[..., 1] ** 2) + (self.params.ksi * d1 + inputs[..., -1]) ** 2)

        u = (inputs[..., 0] * self.params.f_x).unsqueeze(dim=-1)
        v = (inputs[..., 1] * self.params.f_y).unsqueeze(dim=-1)
        coeffs = (1.0 / ((self.params.alpha * d2) + ((1 - self.params.alpha) * (self.params.ksi * d1 + inputs[..., -1])))).unsqueeze(dim=-1)

        plane_points = th.cat([
            u * coeffs,
            v * coeffs
        ], dim=-1)

        shuffle = th.Tensor([self.params.c_x, self.params.c_y])
        return  plane_points + shuffle

    def re_project(self, inputs):

        mx = ((inputs[..., 0] - self.params.c_x) / self.params.f_x).unsqueeze(dim=-1)
        my = ((inputs[..., 1] - self.params.c_y) / self.params.f_y).unsqueeze(dim=-1)
        ru_sqrt = (mx ** 2) + (my ** 2)

        nume = (1 - (self.params.alpha ** 2) * ru_sqrt)
        denu = (self.params.alpha * th.sqrt(th.abs(1 - ((2 * self.params.alpha - 1) * ru_sqrt)))) + (1 - self.params.alpha)
        mz = nume / (denu + 1e-5)

        points = th.cat([mx, my, mz], dim=-1)
        nume = (mz * self.params.ksi) + th.sqrt(mz ** 2 + (1 - self.params.ksi ** 2) * ru_sqrt)
        denu = ((mz ** 2) + ru_sqrt) + 1e-5
        coeff = nume * denu

        print(f"""
            mx: {th.isnan(mx).any()},
            my: {th.isnan(my).any()},
            ru_sqrt: {th.isnan(ru_sqrt).any()},
            nume: {th.isnan(nume).any()},
            denu: {th.isnan(denu).any()},
            mz: {th.isnan(mz).any()},
            points: {th.isnan(points).any()},
            coeff: {th.isnan(coeff).any()}
        """)
        
        ksi_sh = th.ones(coeff.size()) * self.params.ksi
        zeros = th.zeros(coeff.size()) 
        shuffle = th.cat([zeros, zeros, ksi_sh], dim=-1)

        return (coeff * points)[..., :] + shuffle

    def _forward_(self, image: th.Tensor, out_3d: bool, warping_mode: str="bilinear") -> th.Tensor:

        w, h = image.size()[-2:]
        grid = th.stack(th.meshgrid(
            th.arange(0, h),
            th.arange(0, w),
            indexing="xy"
        ), dim=-1).to(th.float32)

        project_2d3d = self.re_project(grid)
        project_3d2d = self.project(project_2d3d) 
        project_3d2d = ((project_3d2d / project_3d2d.max()) * 2) - 1
        
        warped_image = F.grid_sample(
            input=image.unsqueeze(dim=0),
            grid=project_3d2d.unsqueeze(dim=0),
            mode=warping_mode,
            padding_mode='zeros',
            align_corners=False
        ).squeeze(dim=0)
        print(warped_image.min(), warped_image.max(), warped_image.mean())
        
        if out_3d:
            return (warped_image, project_2d3d, self.params)
        
        return (warped_image, self.params)
        
    
    def __call__(self, inputs: th.Tensor, out_3d: bool=False) -> th.Tensor:

        if not self.train:
            with th.no_grad():
                return self._forward_(inputs, out_3d)

        else:
            return self._forward_(inputs, out_3d)
    


if __name__ == "__main__":

    config = {
        "alpha": 0.6139778006995379,
        "ksi": -0.168984435418813,
        "c_x": 964.1528743111107,
        "c_y": 607.8084846801218,
        "f_x": 511.47805067922064,
        "f_y": 511.22309238518983
    }


    path = "C:\\Users\\1\\Desktop\\PythonProjects\\CameraModelAdapter\\input0.jpeg"
    resize = Resize((1920, 1200))
    image = (read_image(path) / 255.0).to(th.float32)
    image = resize(image)

    camera_model = DbModel(**config)
    out_image, points3d, params = camera_model(image, out_3d=True)
    print(out_image.size())
    _, axis = plt.subplots(ncols=3)
    axis[0].imshow(image.permute(1, 2, 0))
    axis[1].imshow(out_image.permute(1, 2, 0))
    axis[2].contourf(points3d[..., 0], points3d[..., 1], points3d[..., 2], cmap="jet")
    plt.show()
 

    # print(out.size(), reout.size())
    # reout = camera_model.project(out)
    # print(reout.size())
    # loss = th.mean(test - reout)
    # print(loss)