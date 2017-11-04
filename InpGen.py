import cv2

class InputGenerator:
    back_name = "background.png"
    obj_name = "object.png"
    __init__(size = (150,150),ratio_range=(0.2,0.6),imdir="./"):
        b = cv2.imread(imdir+back_name)
        self.back = cv2.resize(b,size)
        self.obj = cv2.imread(imdir+obj_name)
        
    def gen_im(self,params):
        #params:[x,y,w,h]
        #place obj at (x,y) with size (w,h) 
        self.obj = cv2.resize(self.obj,(params[2],params[3]))
        



