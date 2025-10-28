import PIL.Image as Image
from typing import Union
import numpy as np

class utils():
    @staticmethod
    def perturb_image(xs, img_pil: Image.Image) -> Union[list, np.ndarray]: 
        if xs.ndim < 2:
            xs = np.array([xs])
        batch = len(xs)
        img = np.array(img_pil)
        xs = xs.astype(int)
        if batch > 1:
            imgs = [img.copy() for _ in range(batch)]
            count = 0
            for x in xs:
                pixels = np.split(x, len(x) / 3)
                for pixel in pixels:
                    x_pos, y_pos, r = pixel
                    imgs[count][x_pos, y_pos, :] = r
                count += 1
        elif batch == 1:
            imgs = img
            for x in xs:
                pixels = np.split(x, len(x) / 3)
                for pixel in pixels:
                    x_pos, y_pos, r = pixel
                    imgs[x_pos, y_pos, :] = r
        return imgs