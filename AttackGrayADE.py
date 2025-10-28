import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import argparse
import torch.backends.cudnn as cudnn
import scipy.io as sio
import string
import matplotlib
import torch.utils.data
matplotlib.use('Agg')
import time
from dataset0_1 import RawDataset_own, AlignCollate_own, tensor2im, save_image
from attackGray0Self import Attacker
from PIL import Image
import editdistance
from attackGray0Self import Record_Dict
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

character_dict = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]_`~'

def demo_own(args):
    for args.saved_model in args.saved_models:
        for args.binary in args.binarys:
            crnn_type = args.saved_model.split('/')[-1].split('.')[0]
            args.Transformation = crnn_type.split('-')[0]
            args.FeatureExtraction = crnn_type.split('-')[1]
            args.SequenceModeling = crnn_type.split('-')[2]
            #args.Prediction = crnn_type.split('-')[3]
            attacker = Attacker(args)
            print("*********************************")
            print("config:", args)
            print("*********************************")
            name0 = args.root.split('/')[-1]
            AlignCollate_demo = AlignCollate_own(imgH=args.imgH, imgW=args.imgW, keep_ratio_with_pad=args.PAD)
            H,W = args.imgH,args.imgW
            demo_data = RawDataset_own(root=args.root, opt=args)
            demo_loader = torch.utils.data.DataLoader(demo_data, batch_size=args.batch_size, shuffle=False, num_workers=int(args.workers), collate_fn=AlignCollate_demo, pin_memory=True)
            nSamples = len(demo_data)
            for args.pixels in args.pixels_range:
                for args.maxiter in args.maxiters:
                    for args.popsize in args.popsize_range:
                        args.cnt = 0
                        num_correct = 0
                        num_success = 0
                        success_index = np.zeros(nSamples)
                        success_rate = 0
                        L0_all = []
                        imageRGB_all = np.zeros([nSamples, H, W, 3])
                        imageRGBadv_all = np.zeros([nSamples, H, W, 3])
                        delta_all = np.zeros([nSamples, H, W, 3])
                        nfevs = np.zeros(nSamples)
                        EDs = np.zeros(nSamples)
                        # l1 = np.zeros(nSamples)
                        l2 = np.zeros(nSamples)
                        # linf = np.zeros(nSamples)
                        start = time.time()
                        save_path = args.results_dir + '_binary=' + \
                                    str(args.binary) \
                                    + '_dynamic=' + str(args.dynamic) + '/'
                        # save_pathRGB = save_path + 'OriGray/'
                        save_pathCorr = save_path + 'GrayCorr/'
                        save_pathAdv = save_path + 'GrayAdv/'
                        # if not os.path.exists(save_pathRGB):
                        #     os.makedirs(save_pathRGB)
                        if not os.path.exists(save_pathCorr):
                            os.makedirs(save_pathCorr)
                        if not os.path.exists(save_pathAdv):
                            os.makedirs(save_pathAdv)
                        for imageRGB, imageGray, image_buf in demo_loader:  # 设置batch_size 强制为1, 读取rgb三通道图片
                            args.cnt = args.cnt + 1
                            
                            # if args.cnt not in [3,4,5,6,8,10,11,12,13,16,18,20,22,23,28,29,30,31,33,36,37,38,40]:
                            #     continue
                            # if args.cnt not in [70]:
                            #     continue
                            print("")
                            print("== Dynamic={} | Binary={} | Dataset: {} | Model: {}==".format(args.dynamic, args.binary, args.dataset, args.mmocr_model))
                            print("== pixels | {} in {} ==".format(args.pixels, args.pixels_range))
                            print("== popsize {} in {} | maxiter {} in {} | Processing {}/{} ==".format(args.popsize, args.popsize_range, args.maxiter, args.maxiters, args.cnt, nSamples))
                            LABEL = demo_data.image_label_list[args.cnt - 1]
                            imageRGB, imageGray = imageRGB[0].to(device), imageGray.to(device)
                            #pre_LABEL = attacker.model_pred(imageGray, args.Prediction)  # 'Attn', 'CTC'
                            img_rgb_pil = Image.open(image_buf[0]).convert('RGB')
                            img_rgb_pil = img_rgb_pil.resize((args.imgW, args.imgH))
                            pre_LABEL, score = attacker.mmocr_pred(img_rgb_pil)
                            print('LABEL:', LABEL, 'pre_LABEL:', pre_LABEL)
                            # save_image(tensor2im(imageRGB), save_pathRGB + str(args.cnt) + '_' + LABEL + '_' + pre_LABEL + '.png')
                            temp = (imageRGB.permute(1, 2, 0) * 0.5 + 0.5) * 255  # 3*32*100变为32*100*3，RGB
                            imageRGB_all[args.cnt - 1, :, :, :] = temp.cpu()
                            imageRGBadv = temp.clone()
                            if pre_LABEL[0].lower() != LABEL.lower():
                                print("Failure recognition！")
                            else:
                                num_correct = num_correct + 1
                                save_image(tensor2im(imageRGB), save_pathCorr + str(args.cnt) + '_' + LABEL + '_' + pre_LABEL[0] + '.png')
                                target_class = LABEL
                                print("== pixels {} in {} | popsize {} in {} Attacking {}/{}==".format(args.pixels, args.pixels_range, args.popsize, args.popsize_range, args.cnt, nSamples))
                                if args.attack_method == 'baseline':
                                    ATK = attacker.untarget_attack_baseline
                                elif args.attack_method == 'simple':
                                    ATK = attacker.untarget_attack_simple
                                elif args.attack_method == 'v1':
                                    ATK = attacker.untarget_attack_v1
                                elif args.attack_method == 'v2':
                                    ATK = attacker.untarget_attack_v2
                                    attacker.load_checkpoint(args.checkpoint_path) if args.checkpoint_path else 0
                                elif args.attack_method == 'pop2':
                                    ATK = attacker.untarget_attack_pop2
                                elif args.attack_method == 'CC':
                                    ATK = attacker.untarget_attack_CC
                                elif args.attack_method == 'CC_v2':
                                    ATK = attacker.untarget_attack_CC_v2
                                
                                flag, perturb, nfev, predicted_class = ATK(img_rgb_pil, imageGray, LABEL, target_class=target_class, pixels=args.pixels, maxiter=args.maxiter, popsize=args.popsize, BiggestLocation=(args.imgH-1, args.imgW-1), idx=args.cnt-1)
                                num_success += flag
                                success_rate = float(num_success) / num_correct
                                if args.attack_method == 'pop2' and args.perturb_all != 0:
                                    L0 = len(perturb) // 5
                                else:
                                    L0 = args.pixels_range[0]
                                L0_all.append(L0)
                                if flag:
                                    print('Sucess Attack!')
                                    success_index[args.cnt - 1] = 1
                                    xs = perturb.astype(int)
                                    if args.perturb_all == 0:
                                        pixels = np.split(xs, len(xs) / 3)
                                        t = []
                                        for pixel in pixels:
                                            x_pos, y_pos, r = pixel
                                            r_o = int(imageRGBadv[x_pos, y_pos, 0])
                                            # g_o = int(imageRGBadv[x_pos, y_pos, 1])
                                            # b_o = int(imageRGBadv[x_pos, y_pos, 2])
                                            imageRGBadv[x_pos, y_pos, 0] = r / 1.0
                                            imageRGBadv[x_pos, y_pos, 1] = r / 1.0
                                            imageRGBadv[x_pos, y_pos, 2] = r / 1.0
                                            t.extend([x_pos, y_pos, r - r_o])
                                    else:
                                        pixels = np.split(xs, len(xs) / 5)
                                        for pixel in pixels:
                                            x_pos, y_pos, r, g, b = pixel
                                            r_o = int(imageRGBadv[x_pos, y_pos, 0])
                                            # g_o = int(imageRGBadv[x_pos, y_pos, 1])
                                            # b_o = int(imageRGBadv[x_pos, y_pos, 2])
                                            imageRGBadv[x_pos, y_pos, 0] = r / 1.0
                                            imageRGBadv[x_pos, y_pos, 1] = g / 1.0
                                            imageRGBadv[x_pos, y_pos, 2] = b / 1.0
                                    print('nfev:', nfev)
                                    nfevs[args.cnt - 1] = nfev
                                    ed = editdistance.eval(predicted_class, LABEL)
                                    EDs[args.cnt -1] = ed
                                    imageRGBadv_all[args.cnt - 1, :, :, :] = imageRGBadv.cpu()
                                    delta_all[args.cnt - 1, :, :, :] = (imageRGBadv - temp).cpu()
                                    # l1[args.cnt - 1] = np.linalg.norm(delta_all[args.cnt-1, :, :, :], ord=1)  # 三维不行
                                    l2[args.cnt - 1] = np.linalg.norm(delta_all[args.cnt - 1, :, :, :]) / 255  # 在（0，1）之间，公平比较的话要乘以2
                                    delta_tensor = torch.tensor(delta_all[args.cnt - 1, :, :, :])
                                    l2_tensor = torch.norm(delta_tensor, p=2)
                                    # linf[args.cnt - 1] = np.linalg.norm(delta_all[args.cnt - 1, :, :, :], ord=np.inf)   # 三维不行
                                    save_image(tensor2im((imageRGBadv.permute(2, 0, 1) / 255 - 0.5) / 0.5), save_pathAdv + str(args.cnt) + '_' + LABEL + '_' + predicted_class + '.png')
                                else:
                                    print('Fail Attack!')
                                nfevs_mean = sum(nfevs) / len(np.nonzero(nfevs)[0])
                                EDs_mean = sum(EDs) / len(np.nonzero(EDs)[0])
                                print("total: %d | pixels: %d num_success rate: %.4f (%d/%d) nfevs_mean: %.4f EDs_mean: %.4f" % (nSamples, args.pixels, success_rate, num_success, num_correct, nfevs_mean, EDs_mean))
                                end_time = time.time()
                                img_time = int(end_time - start) / args.cnt
                                ETA = (len(demo_loader) - args.cnt) * img_time
                                print('***********************************************************************')
                                print(f'ETA: {ETA//86400} day, {(ETA%86400)//3600} hours, {(ETA%3600)//60} minutes, {(ETA%60)} s.')
                                print('***********************************************************************')
                        end = time.time()
                        times = end - start
                        
                        nfevs_mean = sum(nfevs)/len(np.nonzero(nfevs)[0])
                        EDs_mean = sum(EDs) / len(np.nonzero(EDs)[0])
                        L0_mean = sum(L0_all) / len(L0_all)
                        l2 = np.float16(l2)
                        success_index = np.float16(success_index)
                        imageRGB_all = np.float16(imageRGB_all)
                        imageRGBadv_all = np.float16(imageRGBadv_all)
                        delta_all = np.float16(delta_all)
                        sio.savemat(save_path + 'ADE' + str(nSamples) + 'Gray.mat',
                        mdict={'nSamples': nSamples, 'nfev': nfevs_mean, 'EDs_mean': EDs_mean, 'l2': l2, 'L0': L0_mean, 'times': times,  # 'l1': l1, 'linf': linf,
                        'success_index': success_index, 'success_rate': success_rate, 'num_success': num_success, 'num_correct': num_correct,
                        'imageRGB_all': imageRGB_all, 'imageRGBadv_all': imageRGBadv_all, 'delta_all': delta_all})
                        print(name0 + '_' + str(args.pixels))
                        print("total: %d pixels: %d num_success rate: %.4f (%d/%d) mean nfevs: %.4f mean EDs: %.4f mean L0: %.4f"
                              % (nSamples, args.pixels, success_rate, num_success, num_correct, nfevs_mean, EDs_mean, L0_mean))
                        print("total seconds: ", times, "total hours: ", times / 60 / 60)
                        print(save_path + 'ADE' + str(nSamples) + 'Gray.mat')
            
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
    parser.add_argument('--popsize_range', default=[600], type=int, nargs='+', help='The number of adverisal examples in each iteration.')
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
    parser.add_argument('--early_stop', type=int, default=0, help='0 or 1')
    parser.add_argument('--perturb_all', type=int, default=1, help='0: Only attack one character. 1: Try to attack all characters.')
    parser.add_argument('--color_pop_size', type=int,default=600, help='popsize for color space attack')
    parser.add_argument('--openocr_config', type=str, default='models/igtr_syn_model_log/svtr_base_igtr_syn.yml', help='Openocr config')
    parser.add_argument('--use_openocr', type=bool, default=False)
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
        args.results_dir = 'Data/SavedResults/' + args.attack_method + '/' + args.mmocr_model + '/popnum=' + str(args.popnum) + '_popsize=' + str(args.popsize_range[0]) + '_colorpop=' + str(args.color_pop_size) + '_All=' + str(args.perturb_all) + '/' + args.dataset + '/pixels=' + str(args.pixels_range[0]) + '/maxiters=' + str(args.maxiters) + '/early_stop=' + str(args.early_stop) + '/'
        # print(vars(args))
        demo_own(args)