class Center_Crop(object):
    def __init__(self, 
                 *_args, 
                 **_kwargs):
        self.crop_size = 64
    
    def do_augmentation(self, image):
        h, w = image.shape[1], image.shape[2]
        new_h, new_w = self.crop_size, self.crop_size

        top = (h - new_h)//2
        left = (w - new_w)//2
        image = image[:, top:top + new_h, left:left + new_w, :]
        # print('returning image as:', image[0,:,:,:])
        return image.copy()
    
    def change_randomization_params(self, index_):
        index_ = index_
        
    def change_randomization_params_all(self):
        index_ = 0
    
    def print_parms(self):
        print('nothing')


# Augmentation code that we use
class RandGray(object):
    def __init__(self,  
                 batch_size, 
                 p_rand=0.5,
                 *_args, 
                 **_kwargs):
        
        self.p_gray = p_rand
        self.batch_size = batch_size
        self.random_inds = np.random.choice([True, False], 
                                            batch_size, 
                                            p=[self.p_gray, 1 - self.p_gray])
        
    def grayscale(self, imgs):
        # imgs: b x h x w x c
        # the format is incorrect
        b, c, h, w = imgs.shape # format changed, hlynur
        imgs = imgs[:, 0, :, :] * 0.2989 + imgs[:, 1, :, :] * 0.587 + imgs[:, 2, :, :] * 0.114 
        imgs = np.tile(imgs.reshape(b,-1,h,w), (1, 3, 1, 1)) # .astype(np.uint8)
        return imgs

    def do_augmentation(self, images):
        # images: [B, C, H, W]
        bs, channels, h, w = images.shape
        # print(images.shape)
        if self.random_inds.sum() > 0:
            print(self.random_inds)
            # print(sum(self.random_inds))
            # print(images[self.random_inds].shape)
            images[self.random_inds] =  self.grayscale(images[self.random_inds])

        return images
    
    def change_randomization_params(self, index_):
        self.random_inds[index_] = np.random.choice([True, False], 1, 
                                                    p=[self.p_gray, 1 - self.p_gray])
        
    def change_randomization_params_all(self):
        self.random_inds = np.random.choice([True, False], 
                                            self.batch_size, 
                                            p=[self.p_gray, 1 - self.p_gray])
        
    def print_parms(self):
        print(self.random_inds)

def imshow(img):
    """ show an image """
    plt.figure(figsize=(10,8))
    plt.imshow(np.transpose(img, (1, 2, 0)))


# Att the path to our repo raw file of TransformLayer
from TransformLayer import ColorJitterLayer
def random_color_jitter(imgs):
    """
        inputs np array outputs tensor, HLYNUR: this is wrong we input a tensor
    """
    b,c,h,w = imgs.shape
    imgs = imgs.view(-1,3,h,w)
    transform_module = nn.Sequential(ColorJitterLayer(brightness=0.4, 
                                                contrast=0.4,
                                                saturation=0.4, 
                                                hue=0.5, 
                                                p=0.5,
                                                batch_size=num_envs, 
                                                stack_size=1))

    imgs = transform_module(imgs).view(b,c,h,w)
    return imgs


# The original color jitter for all 
def color_jitter(obs):

  transform_module = nn.Sequential(ColorJitterLayer(brightness=0.4, 
                                                  contrast=0.4,
                                                  saturation=0.4, 
                                                  hue=0.5, 
                                                  p=0.5, 
                                                  batch_size=num_envs,
                                                  stack_size=1))
  # device = torch.device('cpu')
  in_stacked_x = obs.to(device)
  # in_stacked_x= in_stacked_x / 255.0
  # in_stacked_x = in_stacked_x.reshape(-1,3,64,64)
  # start = time()
  randconv_x = transform_module(obs)
  # return (randconv_x)


# Grayscaling all images 
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


# random cut outputs
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
