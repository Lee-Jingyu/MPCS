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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
character_dict = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]_`~'



def read_mat(mat_path):
    nfev_mean = []
    success_rate = []
    num_success = []
    num_correct = []
    EDs_mean = []
    L0_mean = []
    SR_max = 0
    l_max = 0
    i_max = 0
    i = 0
    data = loadmat(mat_path) 
    num_success.append(data['num_success'][0][0])
    success_rate.append(data['success_rate'][0][0])
    nfev_mean.append(data['nfev'][0][0])
    EDs_mean.append(data['EDs_mean'][0][0])
    L0_mean.append(data['L0'][0][0])
    num_correct.append(data['num_correct'][0][0])
    l2 = (data['l2'][0])

    l2_mean = sum(l2) / len(np.nonzero(l2)[0])

    #print('===========================================')
    # print('lambda: ' + l)
    # print('num_success: ' + str(num_success[i]))
    # print('success_rate: ' + str(success_rate[i]))
    #print('mean nfev: ' + str(nfev_mean[i]))
    # print('EDs_mean: ' + str(EDs_mean[i]))
    # print('L0_mean: ' + str(L0_mean[i]))
    print('L2: ' + str(l2_mean))
    # print('num_correct: ' + str(num_correct[i]))
    #print('===========================================')
    return success_rate[i]

def cal(args):
    args.binary = args.binarys[0]
    args.root = 'Data/datasets/' + args.datafile + args.dataset

    data = loadmat(args.mat_path)
    advs = data['imageRGBadv_all']
    success_index = data['success_index'][0]

    AlignCollate_demo = AlignCollate_own(imgH=args.imgH, imgW=args.imgW, keep_ratio_with_pad=args.PAD)
    H,W = args.imgH,args.imgW
    demo_data = RawDataset_own(root=args.root, opt=args)
    demo_loader = torch.utils.data.DataLoader(demo_data, batch_size=args.batch_size, shuffle=False, num_workers=int(args.workers), collate_fn=AlignCollate_demo, pin_memory=True)
    attacker = Attacker(args)
    args.cnt = 0
    EDps = []
    EDs = []
    adv_LABELs = []
    correct_LABELS = []
    gt_lens = []
    N = len(demo_loader)
    bar = progressbar.ProgressBar(maxval=N).start()
    for imageRGB, imageGray, image_buf in demo_loader:
        bar.update(args.cnt)
        args.cnt = args.cnt + 1
        LABEL = demo_data.image_label_list[args.cnt - 1]
        gt_lens.append(len(LABEL))
        if success_index[args.cnt - 1] == 0.:
            continue
        img_adv = advs[args.cnt-1].astype(np.uint8)
        imageRGB = imageRGB[0].to(device)
        temp = (imageRGB * 0.5 + 0.5) * 255
        [pre_LABEL], score = attacker.mmocr_pred(temp)
        pre_LABEL = pre_LABEL.replace('<ukn>', '')
        
        if pre_LABEL.lower() != LABEL.lower():
            continue
        
        [adv_LABEL], adv_score = attacker.mmocr_pred(img_adv)
        adv_LABEL = adv_LABEL.replace('<ukn>', '')
        if adv_LABEL == pre_LABEL:
            continue
        adv_LABELs.append(adv_LABEL)
        correct_LABELS.append(LABEL)
        ED = editdistance.eval(adv_LABEL, LABEL)
        l = len(LABEL)
        EDp = ED / l
        EDps.append(EDp)
        EDs.append(ED)
    bar.finish()
    mean_EDp = sum(EDps) / len(EDps)
    mean_ED = sum(EDs) / len(EDs)
    print(f'PR: {mean_EDp}')
    ASR = read_mat(args.mat_path)
    GPR = ASR * mean_EDp
    print(f'SR: {ASR}')
    print(f'ED: {mean_ED}')
    avg_gt_len = sum(gt_lens) / len(gt_lens)
    print(f'Avg GT len: {avg_gt_len}')
    correct_LABEL_lens = [len(l) for l in correct_LABELS]
    avg_correct_label_len = sum(correct_LABEL_lens) / len(correct_LABEL_lens)
    print(f'avg_correct_label_len: {avg_correct_label_len}')

    import json
    content = {
        'adv_LABELs': adv_LABELs,
        'correct_LABELs': correct_LABELS
    }
    with open('LABELs.json', 'w', encoding='utf-8') as f:
    # 将Python对象写入JSON文件，indent=4表示缩进为4个空格，使文件内容更易读
        json.dump(content, f, ensure_ascii=False, indent=4)
    


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
    parser.add_argument('--early_stop', type=int, default=0, help='0 or 1')
    parser.add_argument('--openocr_config', type=str, default='models/igtr_syn_model_log/svtr_base_igtr_syn.yml', help='Openocr config')
    parser.add_argument('--use_openocr', type=bool, default=True)
    parser.add_argument('--mat_path', type=str, help='The mat file path.')
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
