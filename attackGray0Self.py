#!/usr/bin/env python 
# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from PIL import Image
from typing import Union
import matplotlib.pyplot as plt 
import seaborn as sns 
import torch
import torch.nn as nn
from utils import *
from model import Model
#from differential_evolution_simulator import differential_evolution as de
from differential_evolution3 import differential_evolution
from differential_evolution3_color_space import differential_evolution as de_color
from differential_evolution_MPCS import differential_evolution as de_MPCS
from mmocr.apis import TextDetInferencer, TextRecInferencer
import editdistance
import pickle

#from Evolution_Simulator import *
seed = 1  # 随机种子
np.random.seed(seed)  # Numpy module.
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True


class Record_Dict(object):
    def __init__(self, character, max_record, pixel_num):
        self.max = max_record
        self.record = {}
        self.idxs = {}
        for c in character:
            # 初始化值为-1
            self.record[c] = np.full((max_record, pixel_num, 3), -1.)
            self.idxs[c] = 0    # 初始化数组指针，指向下一个存储像素的位置

    def get_record(self, c):
        return self.record[c].copy()

    def save_record(self, c, pixel):
        idx = self.idxs[c]
        if idx >= self.max:    # 已经达到存储上限
            return -1
        n = pixel.size // 3
        pixel = pixel.reshape((n, 3))
        np.copyto(self.record[c][idx], pixel)
        self.idxs[c] += 1

    def save_relative(self, c, pixel, img_size, index, gt_len):
        '''
        这个函数用于替换save_record函数。
        首先根据字符串长度将图像沿宽边N等分，得到N个分区后，计算扰动像素在对应字符分区上的相对位置，并记录。
        记录像素: p = (h, w_relative, v)
        '''
        pixel = pixel.copy()
        _,_,_,W = img_size
        idx = self.idxs[c]
        if idx >= self.max:    
            return -1
        n = pixel.size // 3
        pixel = pixel.reshape((n, 3))

        W_partition_len = W / gt_len
        for p in pixel:
            p[1] = (p[1] % W_partition_len)
            p[1] = p[1] / W_partition_len
        np.copyto(self.record[c][idx], pixel)

        self.idxs[c] += 1

    def to_file(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        path = os.path.join(save_dir, 'Record.pickle')
        with open(path, 'wb') as file:
            pickle.dump(self.record, file)
        return 1

    def load_from_file(self, path):
        with open(path, 'rb') as file:
            self.record = pickle.load(file) 
        return 1

class Attacker(object):
    def __init__(self, c_para):
        self.img_size = (c_para.batch_size, c_para.input_channel, c_para.imgH, c_para.imgW)
        self.batch_size = c_para.batch_size
        # self.popsize = c_para.popsize
        self.binary = c_para.binary
        self.disp = c_para.disp
        self.dynamic = c_para.dynamic
        self.alpha = c_para.alpha
        self.device = c_para.device
        self.batch_max_length = c_para.batch_max_length
        self.Transformation = c_para.Transformation
        self.FeatureExtraction = c_para.FeatureExtraction
        self.SequenceModeling = c_para.SequenceModeling
        self.Prediction = c_para.Prediction

        self.converter = self._load_converter(c_para)
        self.criterion = self._load_base_loss(c_para)
        #self.model = self._load_model(c_para)
        
        self.detector = self._load_detector(c_para)
        self.record_dict = c_para.record_dict
        #self.evolution_simulator = c_para.evolution_simulator
        self.test = False
        self.popnum = c_para.popnum
        self.early_stop = c_para.early_stop
        self.perturb_all = False if c_para.perturb_all == 0 else True
        self.color_pop_size = c_para.color_pop_size
        self.model = self._load_mmocr(c_para)

    @staticmethod
    def _load_base_loss(c_para):
        #criterion = nn.BCELoss().to(c_para.device)
        #criterion = _editdistance
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(c_para.device)
        
        #criterion = nn.CrossEntropyLoss()
        return criterion
    @staticmethod
    def _load_model(c_para):
        if not os.path.exists(c_para.saved_model):
            raise FileNotFoundError("cannot find pth file in {}".format(c_para.saved_model))
        # load model
        with torch.no_grad():
            model = Model(c_para)
            model = torch.nn.DataParallel(model).to(c_para.device)
            model.load_state_dict(torch.load(c_para.saved_model, map_location=torch.device(c_para.device)))
        for name, para in model.named_parameters():
            para.requires_grad = False
        return model
    @staticmethod
    def _load_mmocr(c_para):
        model = TextRecInferencer(model=c_para.mmocr_model,device=c_para.device)
        return model
    @staticmethod
    def _load_detector(c_para):
        detector = TextDetInferencer(model='Textsnake',device=c_para.device)
        return detector
    @staticmethod
    def _load_converter(c_para):
        converter = TextConverter(c_para.labels)
        c_para.num_class = len(converter.character)
        return converter
    def _check_img_size(self, img, size=(1, 1, 32, 100)):
        if isinstance(img, np.ndarray):
            img = cv2.resize(img, (size[3], size[2]), cv2.INTER_CUBIC)
            img = (img.reshape(size) / 255 - 0.5) / 0.5
            return torch.Tensor(img).to(self.device)
        return img
    def text_detection(self, img):
        if isinstance(img, Image.Image):
            img = np.array(img)
        result = self.detector(img, progress_bar=False)
        return result
    def openocr_pred_one(self, img, lower_case=True):
        img = img.astype(np.uint8)
        img = Image.fromarray(img)
        pred = self.openocr(img_numpy=img)[0]
        text_result = pred['text']
        if lower_case:
            text_result = text_result.lower()
        score_result = pred['score']
        score_result = [score_result] * len(text_result)
        return [text_result], [score_result]
    def openocr_pred_batch(self, imgs, lower_case=True):
        imgs = [Image.fromarray(img) for img in imgs]
        preds = self.openocr(img_numpy_list=imgs)
        text_results = []
        score_results = []
        for pred in preds:
            text_result = pred['text']
            if lower_case:
                text_result = text_result.lower()
            score_result = pred['score']
            score_result = [score_result] * len(text_result)
            text_results.append(text_result)
            score_results.append(score_result)
        return text_results, score_results
    def mmocr_pred_one(self, img, lower_case=True):
        pred = self.model(img, return_datasamples=True, progress_bar=False)
        if lower_case:
                text_result = pred['predictions'][0].pred_text.item.lower()
        else:
                text_result = pred['predictions'][0].pred_text.item
        score_result = pred['predictions'][0].pred_text.score
        return [text_result], [score_result]
    def mmocr_pred_batch(self, imgs, lower_case=True):
        '''
        b: batch size
        '''
        text_result = []
        score_result = []
        pred = self.model(imgs, return_datasamples=True, progress_bar=False)['predictions']
        for i in range(len(imgs)):
            # text_result.append(pred[i].pred_text.item)
            # score_result.append(pred[i].pred_text.score)
            text_result.append(pred[i].pred_text.item.lower() if lower_case else pred[i].pred_text.item)
            score_result.append(pred[i].pred_text.score)
        return text_result, score_result
    def mmocr_pred(self, img, popsize=1, lower_case=True):
        pred_one = self.mmocr_pred_one
        pred_batch = self.mmocr_pred_batch
        if torch.is_tensor(img):
            if img.ndim == 3:
                img = img.squeeze(0).permute(1,2,0).cpu().numpy()
                text_result, score_result = pred_one(img, lower_case)
            elif img.ndim == 4:
                b,c,h,w = img.shape
                if b == 1:
                    img = img.squeeze(0).permute(1,2,0).cpu().numpy()
                    text_result, score_result = pred_one(img, lower_case)
                else:
                    imgs = img.permute(0,2,3,1).cpu().numpy()
                    imgs = [imgs[i] for i in range(b)]
                    text_result, score_result = pred_batch(img, lower_case)
        elif isinstance(img, Image.Image):
            img = np.array(img)
            text_result, score_result = pred_one(img, lower_case)
        elif isinstance(img, np.ndarray):
            text_result, score_result = pred_one(img, lower_case)
        elif isinstance(img, list):
            text_result, score_result = pred_batch(img, lower_case)
        else:
            print('Image type not supported!')
            return -1

        return text_result, score_result
  
    def perturb_raw_image(self, xs, img_pil: Image.Image) -> Union[list, np.ndarray]: 
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
    
    def perturb_raw_image_rgbx(self, xs, img_pil: Image.Image) -> Union[list, np.ndarray]: 
        if xs.ndim < 2:
            xs = np.array([xs])
        batch = len(xs)
        img = np.array(img_pil)
        xs = xs.astype(int)
        if batch > 1:
            imgs = [img.copy() for _ in range(batch)]
            count = 0
            for x in xs:
                pixels = np.split(x, len(x) / 5)
                for pixel in pixels:
                    x_pos, y_pos, r, g, b = pixel
                    imgs[count][x_pos, y_pos, 0] = r
                    imgs[count][x_pos, y_pos, 1] = g
                    imgs[count][x_pos, y_pos, 2] = b
                count += 1
        elif batch == 1:
            imgs = img
            for x in xs:
                pixels = np.split(x, len(x) / 5)
                for pixel in pixels:
                    x_pos, y_pos, r, g, b = pixel
                    imgs[x_pos, y_pos, 0] = r
                    imgs[x_pos, y_pos, 1] = g
                    imgs[x_pos, y_pos, 2] = b
        return imgs

    def predict_classes(self, xs, img_rgb, img_gray, label, target_class, popsize, value_num=3):
        if value_num==3:
            imgs_perturbed = self.perturb_raw_image(xs, img_rgb)
        elif value_num == 5:
            imgs_perturbed = self.perturb_raw_image_rgbx(xs, img_rgb)
        text_for_pred = torch.LongTensor(self.batch_size*popsize, self.batch_max_length + 1).fill_(0).to(self.device)
        preds, scores = self.mmocr_pred(imgs_perturbed)

        

        # 字符串编码
        target_code = self.converter.encode(target_class)
        
        # 转换为概率分布，预测字符的概率为score，非预测字符的概率全部相同，为(1-score)/(N-1)。
        
        CTC_targets = torch.LongTensor(target_code)
        all_cost = []
        for pred, score in zip(preds, scores):
            if len(pred) == 0 or len(pred) != len(score):
                cost = 0
                all_cost.append(cost)
                continue
            S = len(target_class)
            T = len(pred)
            C = len(self.converter.dict) + 1

            pred_code = self.converter.encode(pred)
            CTC_preds = torch.zeros((T, C), dtype=torch.float64)  
            for i, p in enumerate(pred_code):
                CTC_preds[i][p] = score[i]
                others_score = 1 / C
                for j in range(C):
                    if j != p:
                        CTC_preds[i][j] = others_score
            CTC_preds = CTC_preds.log_softmax(1)
            cost = self.criterion(CTC_preds, CTC_targets, input_lengths=[T], target_lengths=[S])
            cost = np.array(cost.cpu().detach().numpy())
            cost = 10 - cost
            # cost = cost if target_class == pred else -cost
            all_cost.append(cost)
        all_cost = np.array(all_cost)

        return all_cost 

    def attack_success(self, x, img_rgb, img_gray, label, target_class, value_num = 3):
        if value_num == 3:
            attack_image = self.perturb_raw_image(x, img_rgb)
        elif value_num == 5:
            attack_image = self.perturb_raw_image_rgbx(x, img_rgb)
        [sim_pred], score = self.mmocr_pred(attack_image, self.Prediction)
        # print('*********************')
        # print("Origin: %s" % label, "New: %s" % sim_pred)
        if (label != target_class and sim_pred == target_class) or (label == target_class and sim_pred != target_class):
            return True

    def popinit_multipop(self, inits, pixels_num, record_pixels, img_gray, x1, x2):
        '''
        将种群初始化限制在各个区域内
        '''
        record_proportion = 0.5
        N, M, p = inits.shape   # N为种群个数, M为单个种群大小, p为单个个体像素数*3
        max_record_num = int(M * record_proportion)
        b, c, h, w = img_gray.shape
        stride = w // N
        for j, popi in enumerate(inits):
            w_low = j * stride
            w_high = (j+1) * stride
            if record_pixels == None:
                record_len = 0
            else:
                record_len = len(record_pixels[j])
            record_idx = 0  #用于循环计数
            for init in popi:
                pixel_idx = 0
                for i in range(pixels_num):
                    if record_idx < max_record_num and record_idx < record_len: # 先用record pixels填充,剩下的随机初始化
                        init[i * 3 + 0] = record_pixels[j][record_idx][pixel_idx][0]
                        init[i * 3 + 1] = record_pixels[j][record_idx][pixel_idx][1]
                        # init[i * 3 + 2] = record_pixels[record_idx][2]    # 像素值先不记录
                        pixel_idx += 1
                    else:
                        init[i * 3 + 0] = np.random.randint(0, h)  # init[i * 3 + 0] = np.random.random() * 32
                        init[i * 3 + 1] = np.random.randint(w_low, w_high)  # init[i * 3 + 1] = np.random.random() * 100
                        
                    if self.binary == -2:  # 初始化全随机
                        init[i * 3 + 2] = np.random.randint(0, 256)
                    if self.binary == -1:
                        init[i * 3 + 2] = np.random.normal(128, 127)
                        if init[i * 3 + 2] > 255:
                            init[i * 3 + 2] = 255
                        if init[i * 3 + 2] < 0:
                            init[i * 3 + 2] = 0
                    elif self.binary == 1:
                        init[i * 3 + 2] = np.random.choice([0, 255], size=1, replace=True, p=None)
                    elif self.binary == 2:
                        if img_gray[0, 0, int(init[i * 3 + 0]), int(init[i * 3 + 1])] > 0:
                            init[i * 3 + 2] = 0
                        else:
                            init[i * 3 + 2] = 255
                    elif self.binary == 0:
                        init[i * 3 + 2] = 0
                    elif self.binary == 255:
                        init[i * 3 + 2] = 255
                record_idx += 1
        return
    
    def untarget_attack_baseline(self, img_rgb_pil, img_gray, gt, BiggestLocation=None, target_class=None, pixels=1, maxiter=75, popsize=400):        
        if BiggestLocation == None:
            x1, x2 = 0, 99
        else:
            x1, x2 = BiggestLocation[0], BiggestLocation[1]

        bounds = [(0, 31), (x1, x2), (0, 255)] * pixels
        popmul = max(1, popsize // len(bounds))
        inits = np.zeros([popmul * len(bounds), len(bounds)])
        for init in inits:
            for i in range(pixels):
                init[i * 3 + 0] = np.random.randint(0, 32)  # init[i * 3 + 0] = np.random.random() * 32
                init[i * 3 + 1] = np.random.randint(x1, x2+1)  # init[i * 3 + 1] = np.random.random() * 100
                if self.binary == -2:  # 初始化全随机
                    init[i * 3 + 2] = np.random.randint(0, 256)
                if self.binary == -1:
                    init[i * 3 + 2] = np.random.normal(128, 127)
                    if init[i * 3 + 2] > 255:
                        init[i * 3 + 2] = 255
                    if init[i * 3 + 2] < 0:
                        init[i * 3 + 2] = 0
                elif self.binary == 1:
                    init[i * 3 + 2] = np.random.choice([0, 255], size=1, replace=True, p=None)
                elif self.binary == 2:
                    if img_gray[0, 0, int(init[i * 3 + 0]), int(init[i * 3 + 1])] > 0:
                        init[i * 3 + 2] = 0
                    else:
                        init[i * 3 + 2] = 255
                elif self.binary == 0:
                    init[i * 3 + 2] = 0
                elif self.binary == 255:
                    init[i * 3 + 2] = 255

        # popsize = popmul*3*pixels
        predict_fn = lambda xs, popsize: self.predict_classes(xs, img_rgb_pil, img_gray, gt, target_class, popsize)  # 待最小化的函数
        callback_fn = lambda x, convergence: self.attack_success(x, img_rgb_pil, img_gray, gt, target_class)

        attack_result = differential_evolution(img_gray, predict_fn, bounds, maxiter=maxiter, atol=-1, init=inits,
                        args=(self.dynamic, self.binary, self.alpha[0], self.alpha[1]), popsize=popmul, recombination=1,
                              disp=self.disp, polish=False, callback=callback_fn)
        attack_resultx = attack_result.x
        attack_image = self.perturb_raw_image(attack_resultx, img_rgb_pil)
        [pre_label],score = self.mmocr_pred(attack_image, self.Prediction)
        if (gt == target_class and pre_label != gt) or (gt != target_class and pre_label == target_class):
            return 1, attack_resultx, attack_result.nfev, pre_label
        return 0, [None], [None], [None]
  
    def popinit_color(self, xs, img, popsize=10):
        x_len = len(xs)
        N = x_len // 3  # pixel num
        inits = np.zeros([popsize, N * 5])
        for i,init in enumerate(inits):
            for j in range(N):
                init[0 + j*5] = xs[0 + j*3]
                init[1 + j*5] = xs[1 + j*3]
                if i == 0:  # 保留原攻击像素值
                    init[2 + j*5] = xs[2 + j*3]
                    init[3 + j*5] = xs[2 + j*3]
                    init[4 + j*5] = xs[2 + j*3]
                else:
                    init[2 + j*5] = np.random.randint(0, 256)
                    init[3 + j*5] = np.random.randint(0, 256)
                    init[4 + j*5] = np.random.randint(0, 256)
        return inits
    
      
    def untarget_attack_CC(self, img_rgb_pil, img_gray, gt, BiggestLocation=None, target_class=None, pixels=1, maxiter=75, popsize=400, idx=0):        
        '''
        将图片水平均分为popnum个区域，每个pop初始限制在各个区域内.
        popsize为单个种群容量，popnum为种群数
        '''
        len_g = len(gt)
        x1, x2 = BiggestLocation[0], BiggestLocation[1]
        parameter_num = pixels * 3
        bounds = [(0, x1), (0, x2), (0, 255)] * pixels
        if self.popnum == -1:
            popnum = len(gt)
            popsize = popsize // popnum
        else:
            popnum = self.popnum
        inits = np.zeros([popnum, popsize, parameter_num])

        _,_,H,W = img_gray.shape
        # 获取记录像素
        if self.record_dict.max == 0:
            record_pixels = None
        else:
            # 根据gt字符个数的区域划分来记录像素
            record_pixels = [[] for c in gt]
            partition_len = W / len_g
            for i,c in enumerate(gt):
                record = self.record_dict.get_record(c)
                for r in record:
                    if r[0][0] != -1:   # 若记录不为空
                        for p in r:     # 遍历记录中所有像素点
                            new_p = int((i + p[1]) * partition_len)
                            if new_p == H:    # 边界值
                                p[1] = new_p - 1
                            else:
                                p[1] = new_p
                        record_pixels[i].append(r)
                    else:
                        pass
        
        # 初始化种群
        self.popinit_multipop(inits, pixels, record_pixels, img_gray, x1, x2)
        result = self.text_detection(img_rgb_pil)
        
        # 回调函数
        predict_fn = lambda xs, popsize: self.predict_classes(xs, img_rgb_pil, img_gray, gt, target_class, popsize)  # 待最小化的函数
        callback_fn = lambda x, convergence: self.attack_success(x, img_rgb_pil, img_gray, gt, target_class)
        img_rgb_np = np.array(img_rgb_pil)
        # 寻找对抗像素位置
        attack_result = de_MPCS(img_gray, predict_fn, bounds, maxiter=maxiter, atol=-1, init=inits,
                        args=(self.dynamic, self.binary, self.alpha[0], self.alpha[1]), popsize=popsize, recombination=1,
                              disp=self.disp, polish=False, callback=callback_fn,early_stop=self.early_stop, 
                              perturb_all=len_g if self.perturb_all else 0, idx=idx, img_rgb=img_rgb_np)
        nfev = attack_result.nfev
        if self.perturb_all:
            # 优化颜色空间
            color_pop_size = self.color_pop_size     #种群大小
            
            len_x = len(attack_result.x)
            N = len_x // (3 * pixels)
            color_init = self.popinit_color(attack_result.x, img_rgb_pil, popsize=color_pop_size)
            color_predict_fn = lambda xs, popsize: self.predict_classes(xs, img_rgb_pil, img_gray, gt, target_class, popsize, 5)
            color_callback_fn = lambda x, convergence: self.attack_success(x, img_rgb_pil, img_gray, gt, target_class, 5)
            rgb_bounds = [(0, x1), (0, x2), (0, 255), (0, 255), (0, 255)] * pixels * N
            optimized_result = de_color(img_gray, color_predict_fn, rgb_bounds, maxiter=0, atol=-1, init=color_init,
                        args=(self.dynamic, self.binary, self.alpha[0], self.alpha[1]), popsize=color_pop_size, recombination=1,
                              disp=self.disp, polish=False, callback=color_callback_fn) # maxiter记得改回来！
            nfev += optimized_result.nfev
            attack_resultx = optimized_result.x
            attack_image = self.perturb_raw_image_rgbx(attack_resultx, img_rgb_pil)           
        else:
            attack_resultx = attack_result.x
            attack_image = self.perturb_raw_image(attack_resultx, img_rgb_pil)

        [pre_label],score = self.mmocr_pred(attack_image, self.Prediction)
        if (gt == target_class and pre_label != gt) or (gt != target_class and pre_label == target_class):  #攻击成功
            img_size = self.img_size
            len_p = len(pre_label)
            len_g = len(gt)
            save = self.record_dict.save_relative

            if len_p == len_g:   # 字符识别错误
                for i,(pc,gc) in enumerate(zip(pre_label, gt)):
                    if pc != gc:
                        save(gc, attack_resultx, img_size, i, len_g)
            elif len_p < len_g:  # 字符缺失
                i, j = 0, 0
                while i < len_g and j < len_p:
                    if gt[i] == pre_label[j]:
                        i += 1
                        j += 1
                    else:
                        save(gt[i], attack_resultx, img_size, i, len_g)
                        i += 1
                # 处理剩余的字符
                while i < len_g:
                    save(gt[i], attack_resultx, img_size, i, len_g)
                    i += 1
            elif len_p > len_g:   
                # 多识别字符,考虑为字符分裂的情况。如：'ani' -> 'arii'。
                i,j = 0,0
                while i < len_g and j < len_p:
                    if gt[i] == pre_label[j]:
                        i += 1
                        j += 1
                    elif i == len_g - 1:
                        # 到达最后一个字符
                        save(gt[i], attack_resultx, img_size, i, len_g)
                        i += 1
                        j += 1
                    else:
                        save(gt[i], attack_resultx, img_size, i, len_g)  
                        if j+2 < len_p:
                            if gt[i+1] == pre_label[j+2]:   # 字符分裂
                                j += 3
                                i += 2    
                            else:   #字符误识别
                                i += 1
                                j += 1 
                        else:   # 到达最大分裂个数,之后都按识别错误处理
                            i += 1
                            j += 1        
            else:
                pass
            return 1, attack_resultx, nfev, pre_label
        return 0, [None], [None], [None]
        
      