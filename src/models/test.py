import torch as th

from torch.nn import functional as F
from torch.nn.functional import grid_sample, affine_grid
from torchvision.io import read_image
from torchvision.transforms import Resize
from models import *
from db_test import DbModel
from KbTest import KannalaBrandtProjection

import matplotlib.pyplot as plt
plt.style.use("dark_background")



path = "C:\\Users\\1\\Desktop\\PythonProjects\\CameraModelAdapter\\input0.jpeg"
DS = DbModel(config={
    "alpha": 0.6139778006995379,
    "ksi": -0.1689844354188135,
    "camera_center": (964.1528743111107, 607.8084846801218),
    "focal_len": (511.47805067922064, 511.22309238518983)
})

# KB = KbModel(config={
#     "k_coeffs": [-0.00179801, -0.00179801, -0.00110941, 9.25435e-07],
#     "camera_center": (964.153, 607.808),
#     "focal_len": (615.482, 615.175)
# })
fx, fy = 615.482, 615.175
cx, cy = 964.153, 607.808
KB = KannalaBrandtProjection(
    fx, fy, 
    cx, cy,
    k1=-0.00179801, k2=-0.0017980,
    k3=-0.00110941, k4=9.25435e-07
)

resize = Resize((1200, 1920))
image = (read_image(path) / 255.0).to(th.float32)
image = resize(image)

u, v = th.meshgrid(
    th.arange(0, 1200),
    th.arange(0, 1920),
    indexing="ij"
)

grid = th.stack([u, v], dim=-1)
print(grid.size())
grid_ft = th.flatten(grid, start_dim=0, end_dim=1)
# out_grid_ft, params = camera_model(grid_ft)
pcd = DS.re_project(grid_ft).view(1200, 1920, 3)
plane_projection = KB.ray_to_pixel(pcd)

# print(params)
# # out_grid_ft.mean().backward()
# print("GREATE")
# out_grid = plane_projection.view(1920, 1200, 2).to(th.float32)
# out_grid /= out_grid.max()
# out_grid *= 2
# out_grid -= 1

# print(th.unique(out_grid))
# print(out_grid_ft[:23])
# print(th.min(out_grid), th.max(out_grid), th.mean(out_grid))
# print(out_grid.size(), image.size())
# new_grid_image = grid_sample(
#     image.unsqueeze(dim=0), 
#     out_grid.permute(1, 0, 2).unsqueeze(dim=0),
#     mode="nearest",
#     padding_mode='zeros',
#     align_corners=False
# )

# new_grid_image = new_grid_image.squeeze(dim=0)
# print(new_grid_image)

w, h = plane_projection.size()[-2:]
# print(w, h)
# print(th.tensor([w, h]).size())
_, axis = plt.subplots(ncols=2)
normalized = (plane_projection / plane_projection.max()) * 2 - 1
warped = F.grid_sample(
    image.unsqueeze(0), 
    normalized.unsqueeze(0),
    mode='bilinear',
    padding_mode='zeros',
    align_corners=False
).squeeze(dim=0).permute(1, 2, 0)
axis[0].imshow(image.permute(1, 2, 0))
axis[1].imshow(warped.detach())
plt.show()
# unprojected_points = camera_model.re_project(grid_ft).detach()
# figure = plt.figure()
# space = figure.add_subplot(projection="3d")
# pcd = pcd.detach()
# space.scatter(
#     pcd[:, 0], 
#     pcd[:, 1], 
#     pcd[:, 2], 
#     s=0.12,
#     c=pcd[:, 2],
#     cmap="jet"
# )

plt.show()