# --time**2022.8.13
# --** worker:CSDN大气层煮月亮
# --** email:2642898145@qq.com
 
import cv2
import time
 
# import torch
import warnings
import numpy as np
 
from PIL import Image
from project_yolov7det import yolov7Pt_infer 
from draw import ImageDrawer
from loguru import logger
 
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
 
import configparser
import sys
class VideoTracker(object):
    def __init__(self, cam=-1, video_path='', save_path='', use_frame=[0, -1], display=True):
        self.display = display
        self.use_frame = use_frame
        self.video_path = video_path
        self.cam = cam
        if self.cam != -1:
            print("Using webcam :" + str(self.cam))
            self.vdo = cv2.VideoCapture(self.cam)
        else:
            print("Using video :" + str(self.video_path))
            self.vdo = cv2.VideoCapture()
 
        self.save_path = save_path
        self.frame_interval = 1
        self.use_cuda = True
 
        #use_cuda = self.use_cuda and torch.cuda.is_available()
        #if not use_cuda:
            #warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)
        self.det = yolov7Pt_infer(*self.get_ptModel_config())
        self.drawTool = ImageDrawer()

    def __enter__(self):
        if self.cam != -1:
            ret, frame = self.vdo.read()
            assert ret, "Error: Camera error"
            self.im_width = frame.shape[0]
            self.im_height = frame.shape[1]
            self.count_frame = int(-1)
        else:
            assert os.path.isfile(self.video_path), "Path error"
            self.vdo.open(self.video_path)
            self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.count_frame = int(self.vdo.get(cv2.CAP_PROP_FRAME_COUNT))
            assert self.vdo.isOpened()
 
        if self.save_path != '':
            os.makedirs(self.save_path, exist_ok=True)
 
            # path of saved video and results
            self.save_video_path = os.path.join(self.save_path, "results.avi")
 
            # create video writer
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(self.save_video_path, fourcc, 24, (self.im_width, self.im_height))
 
            # logging
            logger.info("Save results to {}".format(self.save_path))
 
        return self
 
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)
 
        # 释放视频资源
        if hasattr(self, 'vdo') and self.vdo.isOpened():
            self.vdo.release()
            print("Video resource released.")
 
        # 释放视频写入器资源
        if hasattr(self, 'writer'):
            self.writer.release()
            print("Video writer resource released.")
 
    def run(self):
        idx_frame = 0
        all_costTime = 0
 
        while self.vdo.grab():
            idx_frame += 1
 
            start_iter_frame_id = int(self.count_frame * self.use_frame[0])
            end_iter_frame_id = int(self.count_frame * self.use_frame[1])
            # 展示用
            self.show_count_frames = end_iter_frame_id
 
            if idx_frame % self.frame_interval:
                continue
                
            if idx_frame < start_iter_frame_id:
                continue
 
            if idx_frame > end_iter_frame_id:
                break
 
            start = time.time()
            ref, ori_im = self.vdo.retrieve()
 
            if ref is True:
                cv2.imwrite("test.png", ori_im)
                # start your code from here
                im0, source, preds = self.det.infer(ori_im.copy())
                im0 = self.drawTool.draw_polygon(im0)
                for xmin,ymin,xmax,ymax,score,cls in preds:
                    xmin = int(xmin)
                    ymin = int(ymin)
                    xmax = int(xmax)
                    ymax = int(ymax)
                    im0 = self.drawTool.draw_bounding_box(im0, xmin, ymin, xmax, ymax)
                # -----------end-----------
                if self.display:
                    cv2.imshow("frame", im0)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
 
                if self.save_path:
                    self.writer.write(im0)
 
                # logging
                end = time.time() - start if time.time() - start != 0 else 1
                all_costTime += end - start
                if self.display:
                    if self.cam != -1:
                        logger.info("frame schedule:<{}/-1> ({:.2f} ms), fps: {:.03f}"
                                .format(idx_frame, end - start, 1 / (end - start)))
                    else:
                        logger.info("frame schedule:<{}/{}> ({:.2f} ms), fps: {:.03f}"
                                .format(idx_frame, self.show_count_frames , end - start, 1 / (end - start)))
                else:
                    self.print_progress_bar(idx_frame, self.show_count_frames, prefix='Progress:', suffix='Complete', length=50)
 
        logger.info("ALL_COST_TIME:{:.3f}s".format(all_costTime))
 
    def print_progress_bar(self, iteration, total, prefix='', suffix='', length=50, fill='█'):
        """
        Args:
        iteration (int): 当前迭代次数
        total (int): 总迭代次数
        prefix (str): 进度条前缀文本
        suffix (str): 进度条后缀文本
        length (int): 进度条长度
        fill (str): 用于填充进度条的字符
        """
        percent = ("{0:.1f}").format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + '-' * (length - filled_length)
        sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
        sys.stdout.flush()
 
    def pil_to_cv2(self, pil_image):
        """
        Convert a PIL Image to an OpenCV image (NumPy array).
        Args:
        pil_image (PIL.Image): The image in PIL format.
        Returns:
        numpy.ndarray: The image in OpenCV format.
        """
        # Convert the PIL image to a NumPy array
        pil_image_np = np.array(pil_image)
        # Convert RGB to BGR
        cv2_image = cv2.cvtColor(pil_image_np, cv2.COLOR_RGB2BGR)
        return cv2_image
    
    def cv2_to_pil(self, cv2_image):
        """
        Convert an OpenCV image (NumPy array) to a PIL Image.
        Args:
        cv2_image (numpy.ndarray): The image in OpenCV format.
        Returns:
        PIL.Image: The image in PIL format.
        """
        # OpenCV uses BGR by default, whereas PIL uses RGB, so we need to convert the image
        cv2_image_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        # Convert the NumPy array to a PIL image
        pil_image = Image.fromarray(cv2_image_rgb)
        return pil_image

    def get_ptModel_config(self):
        config = configparser.ConfigParser()

        # 读取 config.ini 文件
        config.read('config.ini',encoding='utf-8')

        # 获取 [config] 部分的内容
        weights = config.get('config', 'weights')
        imgsz = config.getint('config', 'imgsz')
        conf_thres = config.getfloat('config', 'conf_thres')
        iou_thres = config.getfloat('config', 'iou_thres')
        device = config.get('config', 'device')
        save_conf = config.getboolean('config', 'save_conf')
        nosave = config.getboolean('config', 'nosave')
        classes_str = config.get('config', 'classes')
        classes = None if classes_str.lower() == 'none' else [int(x) for x in classes_str.split(' ')]
        agnostic_nms = config.getboolean('config', 'agnostic_nms')
        augment = config.getboolean('config', 'augment')
        update = config.getboolean('config', 'update')
        no_trace = config.getboolean('config', 'no_trace')
        
        return (weights, imgsz, conf_thres, iou_thres, device, save_conf, nosave, classes, agnostic_nms, augment, update, no_trace)
    