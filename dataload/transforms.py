import numpy as np

class win:
    def __init__(self,win_min,win_max):
        self.win_min = win_min
        self.win_max = win_max

    def __call__(self,image):
         image = np.clip(image, self.win_min, self.win_max)
         return image


class norm:
    def __init__(self,end,start):
        self.end = end
        self.start = start

    def __call__(self,image):
        image = (image - self.end) / (self.start - self.end)
        image = image * 2 - 1
        return image

