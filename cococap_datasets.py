"""
COCO Image Captioning - Custom COCO Datasets
"""

import numpy as np
import torch
import torch.utils.data.Dataset as Dataset
from pycocotools.coco import COCO
from tqdm import tqdm
import nltk
nltk.download('punkt')

from PIL import Image
import os


class CoCoDataset_DevMode(Dataset):
    """
    Custom dataset used for training routine (batch_size >= 1)
    Implementation modified from Udacity's Computer Vision Nanodegree
    """
    def __init__(self, transform, batch_size, vocab_file, annotations_file, img_folder):
        """
        Args
            transform: (func) data transform
            batch_size: (int) batch size
            vocab_file: (str) path to the existing vocab file
            annotations_file: (str) COCO annotations file (training or validation)
            img_folder: path to the images
        """
        self.transform = transform
        self.batch_size = batch_size
        self.vocab = vocab_file
        self.img_folder = img_folder
        # initialize COCO
        self.coco = COCO(annotations_file)
        # annotation ids
        self.ids = list(self.coco.anns.keys())
        # tokenize all captions
        print('Obtaining caption lengths...')
        all_tokens = [nltk.tokenize.word_tokenize(str(self.coco.anns[self.ids[index]]['caption']).lower()) for index in tqdm(np.arange(len(self.ids)))]
        # create a list of lengths of the captions
        self.caption_lengths = [len(token) for token in all_tokens]

    def __getitem__(self, idx):
        """
        Get the idx-th item from the dataset
        Returns
            image: processed image
            caption: tokenized caption (including special tokens)
        """
        ann_id = self.ids[idx]
        caption = self.coco.anns[ann_id]['caption']
        img_id = self.coco.anns[ann_id]['image_id']
        path = self.coco.loadImgs(img_id)[0]['file_name']
        # Convert image to tensor and pre-process using transform
        image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
        image = self.transform(image)
        # tokenize
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(self.vocab(self.vocab.start_word))
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab(self.vocab.end_word))
        caption = torch.Tensor(caption).long()
        # return pre-processed image and caption tensors
        return image, caption

    def get_data_indices(self):
        # choose a length of the caption
        sel_length = np.random.choice(self.caption_lengths)
        # find all availalbe captions with this length
        all_indices = np.where([self.caption_lengths[i] == sel_length for i in np.arange(len(self.caption_lengths))])[0]
        # select batch_size captions among them
        indices = list(np.random.choice(all_indices, size=self.batch_size))
        return indices

    def __len__(self):
        return len(self.ids)



class CoCoDataset_BLEUMode(Dataset):
    """
    Custom dataset used for calculating BLEU scores (batch_size = 1)
    Implementation modified from Udacity's Computer Vision Nanodegree
    """
    def __init__(self, transform, batch_size, vocab_file, annotations_file, img_folder):
        """
        Args
            transform: data transform
            batch_size: batch size
            vocab_file: path to the existing vocab file
            annotations_file: annotations file (for training dataset)
            img_folder: path to the images
        """
        self.transform = transform
        self.batch_size = batch_size
        self.vocab = vocab_file
        self.img_folder = img_folder
        # initialize COCO
        self.coco = COCO(annotations_file)
        # annotation ids
        self.ids = list(self.coco.anns.keys())
        # tokenize all captions
        print('Obtaining caption lengths...')
        all_tokens = [nltk.tokenize.word_tokenize(str(self.coco.anns[self.ids[index]]['caption']).lower()) for index in tqdm(np.arange(len(self.ids)))]
        # create a list of lengths of the captions
        self.caption_lengths = [len(token) for token in all_tokens]

    def __getitem__(self, idx):
        """
        Get the idx-th item from the dataset
        Returns
            image: processed image
            caption: tokenized caption (including special tokens)
        """
        ann_id = self.ids[idx]
        caption = self.coco.anns[ann_id]['caption']
        img_id = self.coco.anns[ann_id]['image_id']
        path = self.coco.loadImgs(img_id)[0]['file_name']
        # Convert image to tensor and pre-process using transform
        image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
        image = self.transform(image)
        # Convert caption to tensor of word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(self.vocab(self.vocab.start_word))
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab(self.vocab.end_word))
        caption = torch.Tensor(caption).long()
        # return pre-processed image and caption tensors
        return image, caption, img_id

    def get_data_indices(self):
        # choose a length of the caption
        sel_length = np.random.choice(self.caption_lengths)
        # find all availalbe captions with this length
        all_indices = np.where([self.caption_lengths[i] == sel_length for i in np.arange(len(self.caption_lengths))])[0]
        # select batch_size captions among them
        indices = list(np.random.choice(all_indices, size=self.batch_size))
        return indices

    def __len__(self):
        return len(self.ids)



class CoCoDataset_CaptionMode(Dataset):
    """
    Custom dataset used for generating captions and displaying the image (batch_size = 1)
    Implementation modified from Udacity's Computer Vision Nanodegree
    """
    def __init__(self, transform, batch_size, vocab_file, annotations_file, img_folder):
        """
        Args
            transform: data transform
            batch_size: batch size
            vocab_file: path to the existing vocab file
            annotations_file: annotations file (for training dataset)
            img_folder: path to the images
        """
        self.transform = transform
        self.batch_size = batch_size
        self.vocab = vocab_file
        self.img_folder = img_folder
        # annotation file
        test_info = json.loads(open(annotations_file).read())
        self.paths = [item['file_name'] for item in test_info['images']]

    def __getitem__(self, idx):
        """
        Get the idx-th item from the dataset
        Returns
            orig_image: original image
            image: preprocessed image
        """
        path = self.paths[idx]
        # Convert image to tensor and pre-process using transform
        PIL_image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
        orig_image = np.array(PIL_image)
        image = self.transform(PIL_image)
        # return original image and pre-processed image tensor
        return orig_image, image

    def __len__(self):
        return len(self.paths)
