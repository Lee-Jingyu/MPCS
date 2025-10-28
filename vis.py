from scipy.io import loadmat
import os
import numpy as np
import torch
from dataset0_1 import RawDataset_own, AlignCollate_own, tensor2im, save_image
import argparse
import torch.backends.cudnn as cudnn
import editdistance
from attackGray0Self import Record_Dict
import string
from attackGray0Self import Attacker
from PIL import Image
import progressbar
from scipy import ndimage
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
character_dict = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]_`~'

mat_path = 'Data/TIFS_results(seed=1)/pop2/SAR/popnum=20_popsize=30_CrossMutate=0_All=0/IC13_1015/pixels=7/max_record=0/maxiters=[10]/eta=-0.4_lambda=5/_binary=2_popsize=30_maxiter=10_dynamic=2ADE1015Gray.mat'
mmocr_model = 'CRNN'
dataset = 'CUTE80'

def process_gt(gt):
    gt = ((gt + 1) * 0.5) * (-1)
    return gt



def cal(args):
    args.binary = args.binarys[0]
    args.mmocr_model = mmocr_model
    args.dataset = dataset
    args.root = 'Data/datasets/' + args.datafile + args.dataset

    AlignCollate_demo = AlignCollate_own(imgH=args.imgH, imgW=args.imgW, keep_ratio_with_pad=args.PAD)
    H,W = args.imgH,args.imgW
    demo_data = RawDataset_own(root=args.root, opt=args)
    demo_loader = torch.utils.data.DataLoader(demo_data, batch_size=args.batch_size, shuffle=False, num_workers=int(args.workers), collate_fn=AlignCollate_demo, pin_memory=True)
    attacker = Attacker(args)
    args.cnt = 0
    EDps = []
    EDs = []
    N = len(demo_loader)
    bar = progressbar.ProgressBar(maxval=N).start()
    for imageRGB, imageGray, image_buf in demo_loader:
        bar.update(args.cnt)
        args.cnt = args.cnt + 1
        mask = np.load(f'Data/EnergyMaps/CRNN/CUTE80/{args.cnt}.npy')
        mask = process_gt(mask)
        img_rgb_pil = Image.open(image_buf[0]).convert('RGB')
        img_rgb_pil = img_rgb_pil.resize((128, 32))
        img_rgb = np.array(img_rgb_pil)
        img_rgb = np.float32(img_rgb) / 255
        masked_img = show_cam_on_image(img_rgb, mask, use_rgb=True, image_weight=0.7)
        out_img = Image.fromarray(masked_img.astype(np.uint8))
        out_img.save(f'vis/final_raw_scoremask/{args.cnt}.png')
        img_rgb_pil.save(f'vis/raw_image/{args.cnt}.png')
    bar.finish()
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile', type=str, default='lmdb/evaluation/',
                        help=" 'lmdb/training/','lmdb/validation/','lmdb/evaluation/' ")
    parser.add_argument('--dataset', type=str, default='CUTE80',
                        help="training: ST"
                             "evaluation: 'CUTE80','IC03_867','IC13_857','IC15_2077','IIIT5k_3000','SVT','SVTP' ")
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--saved_models', help="path to saved_model to evaluation",
                        default=[#'../../Data/saved_models/None-VGG-BiLSTM-CTC.pth',
                                 '../../Data/saved_models/None-ResNet-None-CTC.pth',
                                 #'../../Data/saved_models/TPS-ResNet-BiLSTM-CTC.pth',
                                 #'../../Data/saved_models/TPS-ResNet-BiLSTM-Attn.pth'
                                 ])
    parser.add_argument('--pixels_range', default=[1], nargs='+', type=int, help='The number of pixels that can be perturbed.')
    parser.add_argument('--maxiters', default=[10], nargs='+', type=int, help='The maximum number of iteration in the DE algorithm.')
    parser.add_argument('--popsize_range', default=[600], type=int, nargs='+', help='3的整数倍，The number of adverisal examples in each iteration.')
    parser.add_argument('--binarys', type=int, default=[2],
                        help='-1:扰动灰度是0-255, 1:扰动灰度是0 and 255, 2:扰动灰度是0 or 255'
                             '0:扰动灰度是0, 255:扰动灰度是255')
    parser.add_argument('--disp', type=bool, default=1, help='0,1')
    parser.add_argument('--dynamic', type=int, default=2, help='0: 复现代码的DE, 1: 动态DDE, 2: AD2E, 3:自己的DE')
    parser.add_argument('--alphas', type=int, default=[[-0.4, 5]],
                        help='CRNN: [-0.4, 11], Attn: [-1, 11]')
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-LABEL-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', default=0, action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default=character_dict, help='character LABEL')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, default='None', help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, default='VGG', help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, default='BiLSTM', help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, default='No', help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512, help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    parser.add_argument('--mmocr_model', help="mmocr_model to use for inference", default='CRNN')
    parser.add_argument('--ScoreLossWeight', help="ScoreLossWeight", default=100)
    parser.add_argument('--max_record', type=int, default=0, help="max record pixels for each character")
    parser.add_argument('--attack_method', type=str, default='simple', 
                        help="1.baseline; 2.simple; 3.v1")
    parser.add_argument('--checkpoint_path', type=str, default=None, help='If specified, turn test mode on.')
    parser.add_argument('--popnum', type=int, default=1, help='Each pop has the size of popsize_range. When popnum is -1, then it will be set to the length of GT.')
    parser.add_argument('--cross_mutate', type=int, default=0, help='0: No cross mutate; 1: activate cross mutate')
    parser.add_argument('--perturb_all', type=int, default=1, help='0: Only attack one character. 1: Try to attack all characters.')
    parser.add_argument('--color_pop_size', type=int,default=600, help='popsize for color space attack')
    parser.add_argument('--mat_path', type=str)
    args = parser.parse_args()
    """ vocab / character number configuration """
    args.device = device
    args.labels = character_dict
    cudnn.benchmark = True
    cudnn.deterministic = True
    args.num_gpu = torch.cuda.device_count()
    if args.sensitive:
        args.character = string.printable[:-6]  # same with ASTER setting (use 94 char).
    if args.rgb:
        args.input_channel = 3
    cudnn.benchmark = True
    # args.root = '../../Data/datasets/' + args.datafile + args.dataset
    # for args.alpha in args.alphas:
    #     args.results_dir = '../../Data/TIFS_results(seed=1)/OnePixelGrayADE_' + \
    #                        args.dataset + '_eta=' + str(args.alpha[0]) + '_lambda=' + str(args.alpha[1]) + '/'
    args.root = 'Data/datasets/' + args.datafile + args.dataset
    args.record_dict = Record_Dict(args.character, args.max_record, args.pixels_range[0])
    #args.evolution_simulator = Evolution_Simulator(args.imgH, args.imgW)
    for args.alpha in args.alphas:
        args.results_dir = 'Data/TIFS_results(seed=1)/' + args.attack_method + '/' + args.mmocr_model + '/popnum=' + str(args.popnum) + '_popsize=' + str(args.popsize_range[0]) + '_CrossMutate=' + str(args.cross_mutate) + '_All=' + str(args.perturb_all) + '/' + args.dataset + '/pixels=' + str(args.pixels_range[0]) + '/max_record=' + str(args.max_record) + '/maxiters=' + str(args.maxiters) + '/eta=' + str(args.alpha[0]) + '_lambda=' + str(args.alpha[1]) + '/'
        # print(vars(args))
        cal(args)
