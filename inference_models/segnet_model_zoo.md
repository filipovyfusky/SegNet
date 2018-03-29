# SegNet Model Zoo
This page lists a number of example SegNet models in the SegNet Model Zoo. NOTE: all Bayesian SegNet models can be tested as SegNet models (for example by using the webcam demo) by removing the line ```sample_weights_test: true``` on all Dropout layers, and setting batch size of 1.

### CamVid

These models have been trained for road scene understanding using the [CamVid dataset](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/).

 - SegNet model file: `inference_models/SegNet/CamVid/segnet_camvid.prototxt`. [Weights](http://mi.eng.cam.ac.uk/~agk34/resources/SegNet/segnet_weights_driving_webdemo.caffemodel).
 - SegNet Basic model file: `inference_models/SegNet/CamVid/segnet_basic_camvid.prototxt`. [Weights](http://mi.eng.cam.ac.uk/~agk34/resources/SegNet/segnet_basic_camvid.caffemodel).
 - Bayesian SegNet model file: `inference_models/BayesianSegNet/CamVid/bayesian_segnet_camvid.prototxt`. [Weights](http://mi.eng.cam.ac.uk/~agk34/resources/SegNet/bayesian_segnet_camvid.caffemodel).
 - Bayesian SegNet Basic model file: `inference_models/BayesianSegNet/CamVid/bayesian_segnet_basic_camvid.prototxt`.[Weights](http://mi.eng.cam.ac.uk/~agk34/resources/SegNet/bayesian_segnet_basic_camvid.caffemodel).

### SUN

These models have been trained for indoor scene understanding using the [SUN RGB-D dataset](http://rgbd.cs.princeton.edu/).

 - SegNet model file: `inference_models/SegNet/SUN/segnet_sun.prototxt`. [Weights](http://mi.eng.cam.ac.uk/~agk34/resources/SegNet/segnet_sun.caffemodel).
 - SegNet model file: `inference_models/BayesianSegNet/SUN/bayesian_segnet_sun.prototxt`. [Weights](http://mi.eng.cam.ac.uk/~agk34/resources/SegNet/segnet_sun.caffemodel).

The model definition file used for training can be found here: `training_models/SegNet/standard/SUN/train_segnet_sun.prototxt`.

We have also trained a model for a 224x224 pixel input:

 - SegNet low resolution model file: `inference_models/SegNet/SUN/segnet_sun_low_resolution.prototxt`. [Weights](http://mi.eng.cam.ac.uk/~agk34/resources/SegNet/segnet_sun_low_resolution.caffemodel).

### Pascal VOC

These models have been trained on the [Pascal VOC 2012 dataset ](http://host.robots.ox.ac.uk/pascal/VOC/).

 - SegNet model file: `inference_models/SegNet/PASCAL/segnet_pascal.prototxt`. [Weights](http://mi.eng.cam.ac.uk/~agk34/resources/SegNet/segnet_pascal.caffemodel).
 - Bayesian SegNet model file: `inference_models/BayesianSegNet/PASCAL/bayesian_segnet_pascal.prototxt`. [Weights](http://mi.eng.cam.ac.uk/~agk34/resources/SegNet/segnet_pascal.caffemodel).

This model is based on the Dropout enc-dec variant and is designed for a 224x224 pixel input.

### CityScapes

This model contains finetuned weights from the original `segnet_camvid` model using an 11 class version of the [CityScapes dataset](https://www.cityscapes-dataset.com/) trained by *Timo Sämann*, Aschaffenburg University of Applied Sciences.

 - SegNet model file: `inference_model/SegNet/CityScapes/segnet_cityscapes.prototxt`. [Weights](http://mi.eng.cam.ac.uk/~agk34/resources/SegNet/segnet_iter_30000_timo.caffemodel).

## License

These models are released under a creative commons license which allows for personal and research use only. For a commercial license please contact the authors. You can view a license summary here:
http://creativecommons.org/licenses/by-nc/4.0/

## Contact

Alex Kendall
agk34@cam.ac.uk
Cambridge University
