# SegNet and Bayesian SegNet Tutorial

This repository contains all the files for you to complete the 'Getting Started with SegNet' and the 'Bayesian SegNet' tutorials here:
http://mi.eng.cam.ac.uk/projects/segnet/tutorial.html

Please note that if following this instruction set, that the folder names __have been modified__.

For example, `Scripts` is now `scripts`, and the training scripts are located under `scripts/training`. The CamVid data is located under `data/CamVid`. And lastly, the models have all been sorted into `inference_models` and `training_models`.

## Caffe-SegNet

SegNet requires a modified version of Caffe to run. Please see the [`caffe-segnet-cudnn7`](https://github.com/navganti/caffe-segnet-cudnn7/tree/7ffea61d08ef7dd153a5c207bfee42882115b104) submodule within this repository, and follow the installation instructions.

## Getting Started

To start, you can use the `scripts/segnet_inference.py` script. It is recommended to use this with the `segnet_cityscapes.prototxt` model, and Timo Sämann's trained weights, which are available for download [here](http://mi.eng.cam.ac.uk/~agk34/resources/SegNet/segnet_iter_30000_timo.caffemodel).

The inference script can be used as follows:

```
python scripts/inference.py inference_models/SegNet/segnet_cityscapes.prototxt \
/PATH/TO/segnet_iter_30000_timo.caffemodel data/test_segmentation.avi [--cpu]
```

The script uses OpenCV's [VideoCapture](https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-videocapture) to parse the data. An example video file has been provided for testing, `data/test_segmentation.avi`.

The easiest way to specify your own segmentation data is via a video file, such as an `.mp4` or `.avi`. Else, you must be sure to specify a folder of images with the format required for VideoCapture.

The --cpu flag indicates whether or not to run segmentation using the CPU. The default is to use the GPU.

## Example Models

A number of example models for indoor and outdoor road scene understanding can be found in the [SegNet Model Zoo](https://github.com/navganti/SegNet/blob/master/inference_models/segnet_model_zoo.md).

## Publications

For more information about the SegNet architecture:

http://arxiv.org/abs/1511.02680
Alex Kendall, Vijay Badrinarayanan and Roberto Cipolla "Bayesian SegNet: Model Uncertainty in Deep Convolutional Encoder-Decoder Architectures for Scene Understanding." arXiv preprint arXiv:1511.02680, 2015.

http://arxiv.org/abs/1511.00561
Vijay Badrinarayanan, Alex Kendall and Roberto Cipolla "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation." PAMI, 2017.

## License

This software is released under a creative commons license which allows for personal and research use only. For a commercial license please contact the authors. You can view a license summary here:
http://creativecommons.org/licenses/by-nc/4.0/


## Contact

Alex Kendall

agk34@cam.ac.uk

Cambridge University
