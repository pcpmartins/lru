# Label recognition utility

Experimental utility for label detection using multiple detection methods. Based on the [(openCV demo)](https://github.com/opencv/opencv/blob/master/samples/cpp/videocapture_starter.cpp). This code is just a proof of a concept. We want to use multiple detection/recognition methods as a way to boost performance.

![figure 1](/images/label_detection_system.png)
*figure 1 - System description*

## Installation

* There is a Visual Studio 2015 project
* OpenCV 3.2.0
* Tesseract binaries for windows x64

## Possible inputs

The program captures frames from a video file, image sequence (01.jpg, 02.jpg ... 10.jpg) or camera connected to your computer.

* To capture from a camera pass the device number. To find the device number, try ls /dev/video*
example: involucros.exe 0, in this case we are assigning device 0.

* You may also pass a video file instead of a device number
example: involucros.exe video.avi

* You can also pass the path to an image sequence and OpenCV will treat the sequence just like a video.
example: involucros.exe right%%02d.jpg

## Keyboard shortcuts

* q, esc : quit
* spacebar : capture frame for text and color comparison.
* r : reset values
* t : text detection toggle
* c : cascade classification toggle

## Detection methods

We use multiple detection algorithms. The first, cascade classifiers is applied to every frame. The other two methods depends on the capture of an isolated frame using the backspace key.

### Cascade classifiers

We use the [Cascade Trainer GUI](http://amin-ahmadi.com/cascade-trainer-gui/) a program that can be used to train, test and improve cascade classifier models. It uses a graphical interface to set the parameters and make it easy to use OpenCV tools for training and testing classifiers (Haar, LBP and HOG features).
There are 5 classifiers XML files in the data/rotulos/ folder (cascade1.xml to cascade5.xml). It is possible to train new models and overwrite this XML classifiers.
In the main window,The detection of the 5 objects its signaled with circles of different colors. If the detection is consistent during a short period of time (default is 3 seconds) the item will lock in the first column of the detection board at the top left of the window.

#### Datasets and parameters

* At the moment the positive images consist on one startinmg image that was multiple times rotated 5 degrees resulting in a total of 72 images.

* The negative samples are 176 random images.

* Haar features, all orientations, default parameters.

### OCR and semantic detection

Its used the [Tesseract Open Source OCR Engine](https://github.com/tesseract-ocr/tesseract), installing the pre-built binaries for windows available at [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki). After the recognition process words are compared with word definitions for each of the 5 items. If there is a match the corresponding item will lock in the second column of the detection board.
The raw text output of the ocr engine is also visible the console.

#### to do

* Own dictionary of terms
* Advanced semantic logic

### Color histogram comparison

We compare any new sample image with our dataset of labels comparing the respective [color histograms](https://docs.opencv.org/3.0-beta/doc/tutorials/imgproc/histograms/histogram_comparison/histogram_comparison.html) using the correlation metric(real values inside [0,1]. The higher the metric, the more accurate the match.

### Denoising

To apply [denoising techniques](http://funvision.blogspot.pt/2016/10/denoising-opencv-image-in-c-video-is.html) is fundamental because noise, usually associated to low quality camera captures can hamper greatly any of the detection algorithms we use. Important parameter h controls the filter strength for the luminance component and hcolor have the same effect in the color componenets. Defaults are both set to 3.

### Whishlist

* SURF+Flann+Homography
* Multi-class classifiers