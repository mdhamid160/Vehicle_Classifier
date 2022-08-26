import cv2
import time
import logging
import warnings
import torch
import subprocess
import numpy as np
from threading import Thread, Lock
import traceback
from logging.handlers import TimedRotatingFileHandler
warnings.filterwarnings("ignore")



class CameraStream(object):
    def __init__(self, src=0):
        #self.stream = cv2.VideoCapture("%s"%RSTP_protocal)                       #for live camera feeding
        self.stream = cv2.VideoCapture("%s" % "./1.mp4")                          #for test video feeding
        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = Lock()
        

    def start(self):
        if self.started:
            print("already started!!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.started:
            (grabbed, frame) = self.stream.read()
            self.read_lock.acquire()
            self.grabbed, self.frame = grabbed, frame
            self.read_lock.release()
            time.sleep(.005)

    def read(self):
        try:
            self.read_lock.acquire()
            frame = self.frame.copy()
            self.read_lock.release()
            return frame
        except:
            pass

    def stop(self):
        self.started = False
        logger.info("entered in stop function")         
        self.thread.join(timeout=1)
        logger.info("exit from stop function")

    def __exit__(self, exc_type, exc_value, traceback):
        self.stream.release()        
        
def box_normal_to_pixel(box,dim):    
        width, height = dim[0], dim[1]
        box_pixel = [int(box[1]), int(box[0]), int(box[3]), int(box[2])]
        return np.array(box_pixel)
    

def check_ping(camera_ip):
    response = subprocess.Popen("ping"+ " "+ camera_ip )
    response=response.wait(timeout=30)   
    return response

def display(image_np,lane_name,dim):
    cv2.namedWindow("%s_Test"%lane_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow("%s_Test"%lane_name,(416,416))
    cv2.imshow("%s_Test"%lane_name,image_np)
    

def is_in_box(l_x1, l_y1, l_x2, l_y2, x3, y3):
    slope = (l_y2 - l_y1) / (l_x2 - l_x1)
    c = (l_y1 - (slope * l_x1))
    y = (slope * x3) + c
    if y <= y3:
        return True
    else:
        return False    
    
         
def myfunction():   
    
    full_image=[]
    veh_id=[]
    tyre=[]
    wait_count = 0
    current_counter = 0
               
    while video_capture:
        time.sleep(.01)
        current_counter +=1
        if current_counter%2 == 0 :
            frame1 = video_capture.read()
            h,w = frame1.shape[0:2]    # 720*1280
            frame1 = cv2.resize(frame1,(640,int(h*(640/w))))
            image_np = frame1.copy()
            dim = image_np.shape[0:2]
                                                                                                                           
            #model Actual prediction
            pred_model = model(image_np)
            boxes = pred_model.xyxy[0][:,:4].cpu()
            scores = pred_model.xyxy[0][:,4].cpu()
            classes = pred_model.xyxy[0][:,5].cpu()  
            cls = classes.tolist()

    
            dets=[]
            veh_class=[]
            tyre_dets = []
            double_tyre = []
            for i, value in enumerate(cls):
                val_3, val_4 = boxes[i][0], boxes[i][1]
                other_lane_roi = is_in_box(363, 231, 397, 355, int(val_3), int(val_4))
    
                if scores[i] > 0.85:                            
                    if int(value) == 0 or int(value) == 1 or int(value) == 2 or int(value) == 4 or int(value) == 5 or int(value) == 6 or int(value) == 8 or int(value) == 9 or int(value) == 11 or int(value) == 13:                                            
                        (y1, x1) = (boxes[i][0], boxes[i][1])
                        (y2, x2) = (boxes[i][2], boxes[i][3])
                        if other_lane_roi:
                            dets.append([x1,y1,x2,y2])
                            veh_class.append(int(value)+1)
                            print(veh_class)
                if scores[i] > .60:
                    if int(value) == 3:
                        (y3, x3) = (boxes[i][0], boxes[i][1])
                        (y4, x4) = (boxes[i][2], boxes[i][3])
                        tyre_dets.append([x3,y3,x4,y4])
                if scores[i] > .60:
                    if int(value) == 10:
                        (y5, x5) = (boxes[i][0], boxes[i][1])
                        (y6, x6) = (boxes[i][2], boxes[i][3])
                        double_tyre.append([x5,y5,x6,y6])
                    
                                                       
            dets = np.asarray(dets)

            double_tyre = np.asarray(double_tyre)

            tyre_dets= np.asarray(tyre_dets)                            
                                     
            if len(dets)!= 0 :
                status  = False
                for i in range(len(dets)):
                    nor_cord=box_normal_to_pixel(dets[i],dim)
                    
                    if len(dets) > 0:
                        status = False
                        wait_count = 0
                        veh_id.append(veh_class[i])    
                        if len(full_image) < 80:
                            cv2.rectangle(image_np,(nor_cord[0],nor_cord[1]),(nor_cord[2],nor_cord[3]),(0,0,255),2)
                            full_image.append(cv2.cvtColor(image_np,cv2.COLOR_BGR2RGB))
                            count+=1
                        logger.info("vehicle detected")                                
                    else:
                        status  = True
                        wait_count = wait_count+1
            else:
                status  = True
                wait_count = wait_count+1

            if len(tyre_dets)!= 0:
                status = False
                for y in range(len(tyre_dets)):
                    temp_cord=box_normal_to_pixel(tyre_dets[y],dim)
                    cv2.rectangle(image_np,(temp_cord[0],temp_cord[1]),(temp_cord[2],temp_cord[3]),(0,255,255),2)
                    cv2.putText(image_np,"axle-%s"%str(int(y)+1),(temp_cord[0],int(temp_cord[1])-2),cv2.FONT_HERSHEY_COMPLEX,.5, (0,204,0),1)
                if len(full_image) < 80:
                    full_image.append(cv2.cvtColor(image_np,cv2.COLOR_BGR2RGB))

            if len(double_tyre)!= 0:
                status = False
                for z in range(len(double_tyre)):
                    tem_cord=box_normal_to_pixel(double_tyre[z],dim)
                    height=int(tem_cord[3]-tem_cord[1])
                    width= int(tem_cord[2]-tem_cord[0])                        
                    tyre.append(12)
                    veh_id.append(7)
                    cv2.rectangle(image_np,(int(tem_cord[0])-1,int(tem_cord[1])-1),(int(tem_cord[2])+1,int(tem_cord[3])+1),(0,255,0),2)
                    cv2.putText(image_np,"Multi-axle",(tem_cord[0],int(tem_cord[1])-2),cv2.FONT_HERSHEY_COMPLEX,.5, (0,204,0),1)
                if len(full_image) < 80:
                    full_image.append(cv2.cvtColor(image_np,cv2.COLOR_BGR2RGB))
                        
                        
            if status and wait_count > 2:
                if len(veh_id) > 0  :
                    veh_class = veh_id
                    
                    if veh_class == 7 or veh_class == 2 or veh_class== 5:
                        skip_time_value = 5
                        if len(tyre) > 1:                                   
                            logger.info("double_tyre is  detected %d times "%(len(tyre)))
                            veh_class = 6
                    else:
                        skip_time_value = skip_time_value
                    logger.info('veh class is :{}'.format(veh_class))
    
                    tyre=[]                            
                    veh_id = []                            
                    full_image=[]                            
                                                                                                                                  
                else:
                    full_image=[]
                    veh_id = []
                    tyre=[]
                                            
            display(image_np,'Test',dim)
            time.sleep(.01)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
                            
    
if __name__ == '__main__':


    # format the log entries
    formatter = logging.Formatter('%(asctime)s-%(message)s',datefmt='%H:%M:%S')
    handler = TimedRotatingFileHandler("logs", when='midnight',backupCount = 30)
    handler.setFormatter(formatter)
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    
                                                 
    # Model Initialization

    model_name = r'./yolov5-master/best.pt'
    yolo_file_path = r'./yolov5-master'
    model = torch.hub.load(yolo_file_path, 'custom', path=model_name, source='local',force_reload=True)
    print("Model loaded Successfully")
    start_time = time.time()

    
    while True :
        video_capture = CameraStream().start()        
        while video_capture.stream.isOpened():
            try:            
                myfunction()
            except Exception as e :
                logger.info("Main function is failing please check error is %s"%e)        
        e ="None frame encounterd or Video feed from camera is not available"
        
        video_capture.stop()
        logger.info("video capture has been stoped")
        cv2.destroyAllWindows()
        logger.info("window has been destroyed")
        traceback.print_exc()
            
            
   

   
