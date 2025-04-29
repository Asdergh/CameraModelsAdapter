import numpy as np
import os 
import cv2


def slice_image(in_path: str, out_path: str) -> None:
    image = cv2.imread(in_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image[:, :1920]
    print(image.shape)
    cv2.imwrite(out_path, image)


if __name__ == "__main__":

    in_path = "C:\\Users\\1\\Downloads\\input.jpeg"
    out_path = "input0.jpeg"
    slice_image(in_path=in_path, out_path=out_path)
    

    