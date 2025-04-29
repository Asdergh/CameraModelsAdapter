import torch as th
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
plt.style.use("dark_background")

from torchvision.io import read_image
from torchvision.transforms import Resize
from torch.nn import Parameter


class KannalaBrandtProjection(th.nn.Module):
    def __init__(self, fx, fy, cx, cy, k1=0.0, k2=0.0, k3=0.0, k4=0.0):
        super().__init__()
        
        self.fx = th.tensor(fx, dtype=th.float32)
        self.fy = th.tensor(fy, dtype=th.float32)
        self.cx = th.tensor(cx, dtype=th.float32)
        self.cy = th.tensor(cy, dtype=th.float32)
        

        self.k1 = th.tensor(k1, dtype=th.float32)
        self.k2 = th.tensor(k2, dtype=th.float32)
        self.k3 = th.tensor(k3, dtype=th.float32)
        self.k4 = th.tensor(k4, dtype=th.float32)
        
    def pixel_to_ray(self, uv):
        

        x = (uv[..., 0] - self.cx) / self.fx
        y = (uv[..., 1] - self.cy) / self.fy
        r = th.sqrt(x**2 + y**2)
        theta = r  
        for _ in range(5):
            theta = theta - (theta + self.k1*theta**3 + self.k2*theta**5 + 
                           self.k3*theta**7 + self.k4*theta**9 - r) / \
                          (1 + 3*self.k1*theta**2 + 5*self.k2*theta**4 + 
                           7*self.k3*theta**6 + 9*self.k4*theta**8)
    
        mask = r > 1e-8
        x[mask] = x[mask] * th.sin(theta[mask]) / r[mask]
        y[mask] = y[mask] * th.sin(theta[mask]) / r[mask]
        z = th.cos(theta)
        
        return th.stack([x, y, z], dim=-1) 
    
    def ray_to_pixel(self, xyz):
    
        x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
        theta = th.atan2(th.sqrt(x**2 + y**2), z)
        
        r = theta + self.k1*theta**3 + self.k2*theta**5 + \
            self.k3*theta**7 + self.k4*theta**9
        
        u = self.fx * (x * r / (th.sqrt(x**2 + y**2) + 1e-8)) + self.cx
        v = self.fy * (y * r / (th.sqrt(x**2 + y**2) + 1e-8)) + self.cy
        
        return th.stack([u, v], dim=-1)
    
    def forward(self, image):
        
        h, w = image.shape[-2:]
        uv_grid = th.stack(th.meshgrid(
            th.arange(w, device=image.device),
            th.arange(h, device=image.device),
            indexing='xy'
        ), dim=-1).float() 
        
        rays = self.pixel_to_ray(uv_grid)
        new_uv = self.ray_to_pixel(rays)  
        normalized = new_uv / new_uv.max() * 2 - 1
        warped = F.grid_sample(
            image.unsqueeze(0), 
            normalized.unsqueeze(0),
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )
        
        return warped.squeeze(0)


if __name__ == "__main__":
    # Параметры камеры (пример для fisheye)
    fx, fy = 615.482, 615.175
    cx, cy = 964.153, 607.808
    kb_model = KannalaBrandtProjection(
        fx, fy, 
        cx, cy,
        k1=-0.00179801, k2=-0.0017980,
        k3=-0.00110941, k4=9.25435e-07
    )
    
    path = "C:\\Users\\1\\Desktop\\PythonProjects\\CameraModelAdapter\\input0.jpeg"
    resize = Resize((1200, 1920))
    image = (read_image(path) / 255.0).to(th.float32)
    image = resize(image)
    
    # Обработка
    warped_img = kb_model(image)
    # warped_img.backward()
    # print(warped_img.mean())
    _, axis = plt.subplots(ncols=2)
    axis[0].imshow(image.permute(1, 2, 0))
    axis[1].imshow(warped_img.permute(1, 2, 0).detach())
    plt.show()

