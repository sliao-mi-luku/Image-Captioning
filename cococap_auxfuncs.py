"""
COCO Image Captioning - helper functions
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import json


def get_word_list_and_sentence(token_list, vocab):
    """
    Given a list of token (ex. [1, 1024, 222, 2]):
        1. remove the <start> token
        2. remove the <end> token and all its following tokens
    And finally return a list of words (ex. ["Hello", "world"]) and the complete sentence (ex. "Hello world")
    Args
        token_list: (list) a list of token integers
        vocab: (obj) the vocalulary object
    Returns
        word_list: (list) a list of words
        sentence: (str) a str of the words joined by spaces
    """
    word_list = []
    for tok in token_list:
        # skip the <start> token
        if tok == 0:
            continue
        # break if it's an <end> token
        if tok == 1:
            break
        # look up the word
        word = vocab.idx2word[tok]
        word_list.append(word)
    sentence = " ".join(word_list)
    return word_list, sentence



def random_sample_testdata(dataloader, encoder, decoder, device):
    """
    Random sample an image from the test data and generate the caption along with it
    Args
        dataloader: (Pytorch dataloader) the test dataloader (batch_size = 1)
    """
    encoder.eval()
    decoder.eval()
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    # sample an image
    orig_image, image_t = next(iter(dataloader))
    image_t = image_t.to(device)

    # plot the original image
    plt.imshow(orig_image[0])
    plt.axis('off')

    # caption prediction
    with torch.no_grad():
        features_t = encoder(image_t).unsqueeze(1)
        token_list = decoder.sample(features_t)

    decoded_word_list, decoded_sentence = get_word_list_and_sentence(token_list)

    print(decoded_sentence)



def image_captioning_custom_image(img_path, encoder, decoder, device):
    """
    Generate a caption for a custom image
    Args
        img_path: (str) path to the image
    """
    encoder.eval()
    decoder.eval()
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    # image preprocessing
    orig_image = np.array(Image.open(img_path).convert('RGB'))
    # plot the original image
    plt.imshow(orig_image)
    plt.axis('off')

    # caption prediction
    image_t = transform_eval(Image.open(img_path).convert('RGB'))
    image_t = image_t.to(device)
    with torch.no_grad():
        features_t = encoder(image_t).unsqueeze(1)
        token_list = decoder.sample(features_t)
    decoded_word_list, decoded_sentence = get_word_list_and_sentence(token_list)
    print(decoded_sentence)
