import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import cv2

class CTCLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'blank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = ['[blank]'] + dict_character  # dummy '[blank]' token for CTCLoss (index 0)

    def encode(self, text, batch_max_length=25):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]

        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]
        text = ''.join(text)
        text = [self.dict[char] for char in text]

        return (torch.IntTensor(text).to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        index = 0
        for l in length:
            t = text_index[index:index + l]
            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
            index += l
        return texts


class AttnLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.character = list_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i

    def encode(self, text, batch_max_length=25):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts


class TextConverter(object):
    """ Convert between text and index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'blank' token required by CTCLoss
            self.dict[char] = i + 1
        self.character = dict_character

    def encode(self, text, popsize=1, batch_max_length=25):
        """ convert text-label into text-index.
        """
        if popsize > 1:
            encoded_sequence = []
            for t in text:  
                encoded_sequence.append([self.dict[char] for char in t.lower()])

            # 填充数组使长度对齐，以转换为tensor
            maxlen = max(len(j) for j in encoded_sequence)
            encoded_sequence = [e + [-1]*(maxlen - len(e)) for e in encoded_sequence]

        else:
            text = ''.join(text)
            encoded_sequence = [self.dict[char] for char in text.lower()]

        x = encoded_sequence
        return x

    def decode(self, text_index, length):
        """ Not implemented yet """
        # texts = []
        # for index, l in enumerate(length):
        #     text = ''.join([self.character[i] for i in text_index[index, :]])
        #     texts.append(text)
        # return texts



class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res

def pop2img(pop, H, W, iter, img_idx, image=None):
    pixel_num = len(pop[0]) // 3
    img = np.zeros((H,W), dtype=np.uint8)
    for x in pop:
        for i in range(pixel_num):
            h = int(x[i*3])
            w = int(x[i*3+1])
            img[h][w] += 1
    if image is not None:
        # 在原图上绘制
        mask = img * 0.2
        image = np.float32(image) / 255
        masked_img = show_cam_on_image(image, mask, use_rgb=True, image_weight=0.5)
        dir_path = f'vis/pop_img/{img_idx}/'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        path = f'{dir_path}pop_{iter}.png'
        out_img = Image.fromarray(masked_img.astype(np.uint8))
        out_img.save(path)
    else:
        # 直接绘制
        dir_path = f'vis/pop/{img_idx}/'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        path = f'{dir_path}pop_{iter}.png'
        plt.imshow(img, cmap='viridis', origin='lower')
        plt.colorbar()
        plt.xlabel('Column')
        plt.ylabel('Row')
        plt.savefig(path)
        plt.close()
        print('pop visualized.')