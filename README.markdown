# YOLO with Core ML and MPSNNGraph

This is the source code for my blog post [YOLO: Core ML versus MPSNNGraph](http://machinethink.net/blog/yolo-coreml-versus-mps-graph/).

YOLO is an object detection network. It can detect multiple objects in an image and puts bounding boxes around these objects. [Read my other blog post about YOLO](http://machinethink.net/blog/object-detection-with-yolo/) to learn more about how it works.

![YOLO in action](YOLO.jpg)

Previously, I implemented YOLO in Metal using the [Forge library](https://github.com/hollance/Forge). Since then Apple released Core ML and MPSNNGraph as part of the iOS 11 beta. So I figured, why not try to get YOLO running on these two other technology stacks too?

In this repo you'll find:

- **TinyYOLO-CoreML:** A demo app that runs the Tiny YOLO neural network on Core ML.
- **TinyYOLO-NNGraph:** The same demo app but this time it uses the lower-level graph API from Metal Performance Shaders.
- **Convert:** The scripts needed to convert the original DarkNet YOLO model to Core ML and MPS format.

To run the app, just open the **xcodeproj** file in Xcode 9 and run it on a device with iOS 11 or better installed.

> **NOTE:** Running these kinds of neural networks eats up a lot of battery power. The app puts a limit on the number of times per second it runs the neural net. You can change this in `setUpCamera()` by changing the line `videoCapture.fps = 5` to a larger number.

## Converting the models

> **NOTE:** You don't need to convert the models yourself. Everything you need to run the demo apps is included in the Xcode projects already. 

If you're interested in how the conversion was done, there are three conversion scripts:

### YAD2K

The original network is in [Darknet format](https://pjreddie.com/darknet/yolo/). I used [YAD2K](https://github.com/allanzelener/YAD2K) to convert this to Keras. Since [coremltools](https://pypi.python.org/pypi/coremltools) currently requires Keras 1.2.2, the included YAD2K source code is actually a modified version that runs on Keras 1.2.2 instead of 2.0.

First, set up a virtualenv with Python 3:

```
virtualenv -p /usr/local/bin/python3 yad2kenv
source yad2kenv/bin/activate
pip3 install tensorflow
pip3 install keras==1.2.2
pip3 install h5py
pip3 install pydot-ng
pip3 install pillow
```

Run the yad2k.py script to convert the Darknet model to Keras:

```
cd Convert/yad2k
python3 yad2k.py -p ../tiny-yolo-voc.cfg ../tiny-yolo-voc.weights model_data/tiny-yolo-voc.h5
```

To test the model actually works:

```
python3 test_yolo.py model_data/tiny-yolo-voc.h5 -a model_data/tiny-yolo-voc_anchors.txt -c model_data/pascal_classes.txt 
```

This places some images with the computed bounding boxes in the `yad2k/images/out` folder.

### coreml.py

The **coreml.py** script takes the `tiny-yolo-voc.h5` model created by YAD2K and converts it to `TinyYOLO.mlmodel`. Note: this script requires Python 2.7 from `/usr/bin/python` (i.e. the one that comes with macOS).

To set up the virtual environment:

```
virtualenv -p /usr/bin/python2.7 coreml
source coreml/bin/activate
pip install tensorflow
pip install keras==1.2.2
pip install h5py
pip install coremltools
```

Run the `coreml.py` script to do the conversion (the paths to the model file and the output folder are hardcoded in the script):

```
python coreml.py
```

### nngraph.py

The **nngraph.py** script takes the `tiny-yolo-voc.h5` model created by YAD2K and converts it to weights files used by `MPSNNGraph`. Requires Python 3 and Keras 1.2.2.
