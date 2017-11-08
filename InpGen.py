import numpy as np
import cv2
import copy

class InputGenerator:
    back_name = "background.png"
    obj_name = "tomato.png"

    def __init__(self,size = (150,150),batch_size = 16,ratio_range=(0.2,0.6),imdir="./"):
        b = cv2.imread(imdir+self.back_name)
        b= cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
        self.bs =batch_size 
        self.size = size
        self.back = cv2.resize(b,size)
        self.obj = cv2.imread(imdir+self.obj_name)
        self.obj= cv2.cvtColor(self.obj, cv2.COLOR_BGR2RGB)
        self.maxh = int(ratio_range[1]*size[1])
        self.maxw = int(ratio_range[1]*size[0])
        self.minh = int(ratio_range[0]*size[1])
        self.minw = int(ratio_range[0]*size[0])

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        ims = []
        msks = []
        params = []
        for b in range(self.bs):
            h = np.random.randint(self.minh, high=self.maxh)
            w = np.random.randint(self.minw, high=self.maxw)
            y = np.random.randint(1, high=self.size[1]-h)
            x = np.random.randint(1, high=self.size[0]-w)
            i,m = self.gen_im((x,y,w,h))
            ims.append(i)
            msks.append(m)
            params.append([x,y,w,h])
        return np.array(ims)/255.,np.array(msks),np.array(params)
        
    def gen_im(self,params):
        x,y,w,h = params
        #place obj at (x,y) with size (w,h) 
        obj = cv2.resize(self.obj,(params[2],params[3]))
        mask = np.zeros((self.back.shape))
        im = copy.copy(self.back)
        for i in range(h-1):
            for j in range(w-1):
                im[i+y][j+x] = obj[i][j]
                mask[i+y][j+x] = 1
        return im,mask
        



