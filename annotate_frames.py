import time
import torch 
from torch.autograd import Variable
import cv2 
from scripts.util import *
from scripts.darknet import Darknet
from scripts.preprocess import prep_image
import argparse
import random
#from social_distancing_latest import distancing
import math
import os

class Detection:
    cfgfile = "./cfg/yolov3.cfg"
    weightsfile = "yolov3.weights"
    classes = load_classes('./data/coco.names')
    num_classes = 80
    bbox_attrs = 5 + num_classes
    args = None
    confidence = None
    nms_thesh = None
    CUDA = False
    model = None
    inp_dimensions = None
    colors = list()
    
    # def map_point(self, point):
    #     x = int(((self.matrix[0][0] * point[0]) + (self.matrix[0][1]*point[1]) + self.matrix[0][2])/((self.matrix[2][0] * point[0]) + (self.matrix[2][1]*point[1]) + self.matrix[2][2]))
    #     y = int(((self.matrix[1][0] * point[0]) + (self.matrix[1][1]*point[1]) + self.matrix[1][2])/((self.matrix[2][0] * point[0]) + (self.matrix[2][1]*point[1]) + self.matrix[2][2]))
    #     return (x,y)
    
    # def cal_distance(self,point1,point2):
    #     temp = ((point1[0]-point2[0])**2) + ((point1[1]-point2[1])**2)
    #     dist = temp**(1/2)
    #     print(point1 , point2)
    #     print(dist)
    #     return dist*0.3

    def color_generator(self):
        ''' Generate a color pallete for different objects.
        '''

        for i in range(0, 80):
            temp = list()
            b = random.randint(0, 255)
            g = random.randint(0, 255)
            r = random.randint(0, 255)
            temp.append(b)
            temp.append(g)
            temp.append(r)
            self.colors.append(temp)

    def plot_results(self, output, img , imge_name):
        ''' 
        Draw the bounding box and results on the frame.
        '''
        a = []
        print(img.shape[1])
        for x in output:

            if float(x[5]) <= 0 or float(x[5]) > 1:
                continue

            cls = int(x[-1])
            if int(cls)<0 or int(cls)>80:
                continue

            c1 = tuple(x[1:3].int())
            c2 = tuple(x[3:5].int())

            x1, y1 = c1[0].item(), c1[1].item()
            x2, y2 = c2[0].item(), c2[1].item()
                
            label = "{0}".format(self.classes[cls])
            score = str("{0:.3f}".format(float(x[5])))
            color = self.colors[cls]
            if label == 'person':
                
                # cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                # cv2.rectangle(img, (x1, y1), (x1 + (len(label) + len(score)) * 10, 
                #                 y1 - 10) , color, -1, cv2.LINE_AA)
                # cv2.putText(img, label + ':' + score, (x1, y1), 
                #             cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
                cx = (x1 + x2)/2
                cy = (y1 + y2)/2
                w = x2 - x1
                h = y2 -y1

                a.append(["person " + score ,x1,y1,x2,y2])

            if label == 'car':
                
                # cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                # cv2.rectangle(img, (x1, y1), (x1 + (len(label) + len(score)) * 10, 
                #                 y1 - 10) , color, -1, cv2.LINE_AA)
                # cv2.putText(img, label + ':' + score, (x1, y1), 
                #             cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
                cx = (x1 + x2)/2
                cy = (y1 + y2)/2
                w = x2 - x1
                h = y2 -y1

                a.append(["car " + score ,x1,y1,x2,y2])
            
            
        mf = open(self.save_path + imge_name[:-4] + '.txt', 'w')
        text = ''
        for i in range(0,len(a)): 
            append_string = str(a[i][0]) + ' ' + str(a[i][1])+ ' '+str(a[i][2])+ ' '+str(a[i][3])+ ' '+str(a[i][4])+'\n'
            text += append_string
        mf.write(text)
        mf.close()
        #cv2.imwrite(self.save_path + str(self.frame_number) + '.jpg', img)
        self.frame_number += 1
        return img

    def argsparser(self):
        '''
        Argument parser for command line arguments.
        '''

        parser = argparse.ArgumentParser(description='YOLO v3 Cam Demo')
        parser.add_argument('--confidence', dest='confidence', help='Object Confidence to filter predictions', default=0.25, type=float)
        parser.add_argument('--nms_thresh', dest='nms_thresh', help='NMS Threshhold', default=0.4, type=float)
        parser.add_argument('--reso', dest='reso', help=
                            'Input resolution of the network. Increase to increase accuracy. Decrease to increase speed',
                            default=416, type=int)
        parser.add_argument('--source', dest='source', default=0, help='Input video source', type=str)
        parser.add_argument('--skip', dest='skip', default=1, help='Frame skip to increase speed', type=int)
        parser.add_argument('--path', dest='path to the images folder', default="./images", help='path to the images folder', type=str)
        
        return parser.parse_args()
    
    def run(self):
        '''
        Method to run the detection.
        '''

        # cap = cv2.VideoCapture('result3.avi')
        # assert cap.isOpened(), 'Cannot capture source'
        img_path = self.args.path

        frames = 0
        start = time.time()  

        all_images = os.listdir(img_path)
        for i in range(0,len(all_images)):
            ret = True
        
            frame = cv2.imread(img_path+ all_images[i])
            #ret, frame = cap.read()
            #ret, frame = cap.read()
           
            #frame = cv2.resize(frame,(1280,720))
            if ret:
                
                img, orig_im, dim = prep_image(frame, self.inp_dimensions)
                im_dim = torch.FloatTensor(dim).repeat(1, 2)                        
            
                if self.CUDA:
                    im_dim = im_dim.cuda()
                    img = img.cuda()
            
                output = self.model(Variable(img), self.CUDA)
                output = write_results(output, self.confidence, self.num_classes, nms = True, nms_conf = self.nms_thesh)

                if not type(output) == int:

                    output[:,1:5] = torch.clamp(output[:,1:5], 0.0, float(self.inp_dimensions))/self.inp_dimensions

                    im_dim = im_dim.repeat(output.size(0), 1)
                    output[:,[1,3]] *= frame.shape[1]
                    output[:,[2,4]] *= frame.shape[0]
            
                    orig_im = self.plot_results(output, orig_im , all_images[i])

                orig_im = cv2.resize(orig_im,(1280,720))
                #self.out.write(orig_im)
            
                cv2.imshow(self.windowName, orig_im)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                frames += 1
                print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)), end='\r')
    
    def __init__(self):
        '''
        Intitialize method to run on class object creation.
        '''
        self.frame_number = 0
        self.save_path = './annotations/'
        #pts1 = np.float32([[512, 704], [596, 219],[980, 223], [1220, 625]])
        #pts2 = np.float32([[126, 1508], [0, 0], [836,0], [700, 1326]])
        # pts1 = np.float32([[368, 706], [748, 590],[878, 757], [447, 992]])
        # pts2 = np.float32([[0, 0], [1219, 0], [1219,610], [0, 610]])
        pts1 = np.float32([[707, 318], [852, 299],[980, 387], [829, 405]])
        pts2 = np.float32([[116, 0], [616, 0], [680,697], [0, 657]])
        self.matrix = cv2.getPerspectiveTransform(pts1, pts2)
        
        self.args = self.argsparser()

        self.CUDA = torch.cuda.is_available()
        if self.CUDA:
            print('Device Used: ', torch.cuda.get_device_name(0))
            print('Capability: ', torch.cuda.get_device_capability(0))

        self.confidence = float(self.args.confidence)
        self.nms_thesh = float(self.args.nms_thresh)

        self.model = Darknet(self.cfgfile)
        self.model.load_weights(self.weightsfile)
        self.model.net_info["height"] = self.args .reso

        self.inp_dimensions = int(self.model.net_info["height"])

        assert self.inp_dimensions % 32 == 0, 'Input not a multiple of 32'
        assert self.inp_dimensions > 32, 'Input must be larger than 32'


        if self.CUDA:
            self.model.cuda()

        self.model.eval()
        #self.social_distacing = distancing()
        self.color_generator()
        self.windowName = "Object Detection"
        cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)
        ##cv2.setWindowProperty(self.windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        #self.out = cv2.VideoWriter('result2.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10,(1280,720))
if __name__ == '__main__':

    detection = Detection()
    detection.run()
    