import torch
import numpy as np
from TransformLayer import ColorJitterLayer

def grayscale(imgs,device):
    # imgs: b x c x h x w
    b, c, h, w = imgs.shape
    # frames = c // 3
    
    # imgs = imgs.view([b,frames,3,h,w])
    # imgs = imgs[:, 0, :, :] * 0.2989 + imgs[:, 1, :, :] * 0.587 + imgs[:, 2, :, :] * 0.114 
    
    imgs = imgs.view([b,3,h,w])
    imgs = imgs[:, 0, ...] * 0.2989 + imgs[:, 1, ...] * 0.587 + imgs[:, 2, ...] * 0.114 
    

    # imgs = imgs.type(torch.uint8).float()
    # assert len(imgs.shape) == 3, imgs.shape
    imgs = imgs[:, None, :, :]
    imgs = imgs * torch.ones([1, 3, 1, 1], dtype=imgs.dtype).float().to(device) # broadcast tiling
    return imgs


def color_jitter(obs):
  # device = torch.device('cpu')
  in_stacked_x = obs.to(device)
  # in_stacked_x= in_stacked_x / 255.0
  # in_stacked_x = in_stacked_x.reshape(-1,3,64,64)
  # start = time()
  randconv_x = transform_module(obs)
  # return (randconv_x)

def random_cutout(imgs, min_cut,max_cut):
    """
        args:
        imgs: shape (B,C,H,W)
        out: output size (e.g. 84)
    """
    n, c, h, w = imgs.shape
    w1 = np.random.randint(min_cut, max_cut, n)
    h1 = np.random.randint(min_cut, max_cut, n)
    
    cutouts = np.empty((n, c, h, w), dtype=imgs.dtype)
    for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
        cut_img = img.copy()
        cut_img[:, h11:h11 + h11, w11:w11 + w11] = 0
        #print(img[:, h11:h11 + h11, w11:w11 + w11].shape)
        cutouts[i] = cut_img
    return cutouts
