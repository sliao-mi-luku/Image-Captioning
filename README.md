# Image Captioning

*Last updated: 02/04/2022*

*This project is developed from Udacity's Computer Vision Nanodegree*

## Project Summary

1.
2.
3.
4.

## Future Work

1. Use **Vision Transformer** to serve as the encoder to extract features from the images
2. Use **Transformer** to serve as the decoder to generate captions
3. Use SOTA **VisualBERT** or **ViLBERT** for end-to-end image captioning

## Dataset

I use the Microsoft Common Objects in Content (MS COCO) Dataset (Ver. 2014)

Link to the MS COCO Dataset: https://cocodataset.org/#home

[![COCO-homepage.png](https://i.postimg.cc/K8JTC9jW/COCO-homepage.png)](https://postimg.cc/rDRzrNSG)
<p align="center">
    MS COCO Dataset (https://cocodataset.org/#home)
</p>

**How to Download MS COCO Dataset**

This projects use the [COCO API](https://github.com/cocodataset/cocoapi) provided by the MS COCO. Instructions in detail can be found on their websites. Here is a brief instruction:

1. In your work directory, create a folder `opt`
2. In the `opt` folder, run the bash command `git clone https://github.com/cocodataset/cocoapi.git`
3. Download `2014 Train/Val annotations [241MB]` from [MS COCO download page](https://cocodataset.org/#download)
4. Extract the zip file `annotation_trainval2014.zip` into the `opt/cocoapi` folder
5. (Checkpoint) You should see a folder `annotations` inside the `opt/cocoapi` folder
6. Download `2014 Testing Image info [1MB]` from [MS COCO download page](https://cocodataset.org/#download)
7. Extract the zip file `image_info_test2014.zip` into the `opt/cocoapi` folder
8. (Checkpoint) You should see a file `image_info_test2014.json` inside the `opt/cocoapi/annotations` folder
9. In your work directory, create a folder `images`
10. Download `2014 Train images [83K/13GB]` from [MS COCO download page](https://cocodataset.org/#download)
11. Download `2014 Val images [41K/6GB]` from [MS COCO download page](https://cocodataset.org/#download)
12. Download `2014 Test images [41K/6GB]` from [MS COCO download page](https://cocodataset.org/#download)
13. Extract all 3 downloaded zip files (steps 10-12) into the `images` folder
14. (Checkpoint) You should see 3 folders (`train2014`, `val2014`, `test2014`) inside the `images` folder


## Model

The model consists of an encoder an a decoder. The encoder extract semantic information from the input image to generate a feature vector.

### Encoder

I use a pre-trained ResNet-50 network to extract the features from an image. I removed the last fc layer, flattened the final output and pass through a dense layer to obtain a feature vector of size `embed_size`

### Decoder

I use an LSTM netork as the decoder. I train the model from scratch.


## Hyperparameters

#### Vocaburary frequency threshold

The parameter `vocab_freq_threshold` set the minimum number that a vocabulary needs to appear in the training corpus to enter our vocabulary dictionary. The larger the `vocab_freq_threshold`, the bigger the vocabulary dictionary.

## Evaluation

I use the **BLEU-4 Score** to evaluate the model performance


## References

1. Udacity's Computer Vision Nanodegree
2. COCO API: https://github.com/cocodataset/cocoapi
3. This notebook tells how to download the COCO Dataset https://colab.research.google.com/github/rammyram/image_captioning/blob/master/Image_Captioning.ipynb
4. Google's paper using LSTM for image captioning https://arxiv.org/pdf/1411.4555.pdf
