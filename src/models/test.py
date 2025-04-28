import torch as th
from torch.nn.functional import grid_sample, affine_grid
from torchvision.io import read_image
from torchvision.transforms import Resize
from models import *

import matplotlib.pyplot as plt
plt.style.use("dark_background")



path = "C:\\Users\\1\\Downloads\\lol.jpeg"
camera_model = UcModel(config={
    "alpha": 0.0098,
    "camera_center": (64, 64),
    "focal_len": (20.12, 20.12)
})

resize = Resize((128, 128))
image = (read_image(path) / 256.0).to(th.float32)
image = resize(image)

u, v = th.meshgrid(
    th.arange(0, 128),
    th.arange(0, 128)
)
grid = th.stack([u, v], dim=-1)
grid_ft = th.flatten(th.stack([u, v], dim=-1), start_dim=0, end_dim=1)
out_grid_ft = camera_model(grid_ft)
out_grid = out_grid_ft.view(128, 128, 2).to(th.float32)
out_grid /= 126.0
out_grid *= 2
out_grid -= 1


# print(th.unique(out_grid))
print(out_grid_ft[:23])
print(th.min(out_grid), th.max(out_grid), th.mean(out_grid))

new_grid_image = grid_sample(
    image.unsqueeze(dim=0), 
    out_grid.unsqueeze(dim=0),
    mode="bicubic"
)
new_grid_image = new_grid_image.squeeze(dim=0)
# print(new_grid_image)

_, axis = plt.subplots(ncols=2)
axis[0].imshow(image.permute(1, 2, 0))
axis[1].imshow(new_grid_image.permute(2, 1, 0).detach())


unprojected_points = camera_model.re_project(grid_ft).detach()
figure = plt.figure()
space = figure.add_subplot(projection="3d")
space.scatter(
    unprojected_points[:, 0], 
    unprojected_points[:, 1], 
    unprojected_points[:, 2], 
    s=0.12,
    c=unprojected_points[:, 2],
    cmap="jet"
)

plt.show()


