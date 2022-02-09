"""
COCO Image Captioning - model evaluation functions
"""

import torch
import sys

from torchtext.data.metrics import bleu_score

from cococap_auxfuncs import get_word_list_and_sentence


def eval_BLEU(encoder, decoder, dataloader, vocab, device):
    """
    Evaluate the model on a chosen dataset to calculate the overall BLEU score
    Args
        encoder: (Pytorch model) encoder
        decoder: (Pytorch model) decoder
        dataloader: (Pytorch dataloader) single_dataloader_train OR single_dataloader_val
        vocab: (obj) the vocabulary object
        device: gpu or cpu
    Returns
        avg_bleu: (float) average BLEU score
        bleu_list: (list) list of BLEUs scores for all data
    """
    # turn on eval mode and move to GPU
    encoder.eval()
    decoder.eval()

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # dicts to store candidate and reference sentences
    dict_candidates = dict()
    dict_references = dict()

    # create a list to store all BLEU scores
    bleu_list = []

    cnt = 0

    # load (processed image,  caption) one by one (batch_size is 1)
    for image_t, caption_t, img_id in dataloader:

        image_t = image_t.to(device)
        caption_t = caption_t.to(device)

        img_id = img_id.tolist()[0]  # int

        cnt += caption_t.size(0)

        with torch.no_grad():
            # encode
            feature_t = encoder(image_t).unsqueeze(1)
            # decode
            token_list = decoder.sample(feature_t)

            # convert token list to word list
            decoded_word_list, decoded_sentence = get_word_list_and_sentence(token_list, vocab)
            if decoded_sentence not in dict_candidates.get(img_id, []):
                dict_candidates[img_id] = dict_candidates.get(img_id, []) + [decoded_word_list]

            # convert captions to word list
            ref_word_list, ref_sentence = get_word_list_and_sentence(caption_t.tolist()[0], vocab)
            if ref_sentence not in dict_references.get(img_id, []):
                dict_references[img_id] = dict_references.get(img_id, []) + [ref_word_list]

        stats = "[{}/{}] Calculating BLEU scores...".format(cnt, len(dataloader.dataset))
        # same line print out
        print('\r' + stats, end="")
        sys.stdout.flush()

        if cnt == len(dataloader.dataset):
            print('\r' + stats)
            break

    # calculate BLEU
    bleu_candidates = []
    bleu_references = []
    for img_id in dict_candidates.keys():

        for cancadate in dict_candidates[img_id]:
            bleu_candidates.append(cancadate)
            bleu_references.append(dict_references[img_id])

    bleu1 = bleu_score(bleu_candidates, bleu_references, max_n=1, weights=[1.0])
    bleu2 = bleu_score(bleu_candidates, bleu_references, max_n=2, weights=[0.5, 0.5])
    bleu3 = bleu_score(bleu_candidates, bleu_references, max_n=3, weights=[0.33, 0.33, 0.33])
    bleu4 = bleu_score(bleu_candidates, bleu_references, max_n=4, weights=[0.25, 0.25, 0.25, 0.25])

    bleu = [bleu1, bleu2, bleu3, bleu4]

    return bleu, bleu_candidates, bleu_references
