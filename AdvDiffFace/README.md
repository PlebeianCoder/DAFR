# AdvFaceUtil
A collection of tools for the creation of adversarial face accessories.
This project includes datasets for loading images, pre-made facial recognition
systems, an interactive demonstration program to test out adversarial methods
and a component-based ecosystem for running various project components.

## Installation
To install the project, create a Python 3.9 or above virtual environment either using
Pip or Conda. Using Pip we create a virtual environment using:
```
python -m venv venv
```
To install the dependencies run (on the virtual environment):
```
python -m pip install -r requirements.txt
```
_Note: dlib is one of the possible options for the alignment of images before passing
into the facial recognition systems. To install Dlib using CUDA, you may need to install from source._

_Note: This module uses [**PyFaceAR**](https://github.com/The-Adversaries/PyFaceAR) for 3D face augmentation in the demonstration software.
To install, please check out that repository._

## Execution
There are three scripts which can be executed, detailed below.

### Training Script
The training script, which is used to train a facial recognition system can be executed by running:

```
python -m advfaceutil.train NETWORK "/path/to/dataset" "/path/to/researchers" "/path/to/base_weights.pth" "/path/to/output" "DATASET" "SIZE"
```

The usage of this file is as follows:
```
usage: train.py [-h] [-researchers RESEARCHERS] [-log-level {CRITICAL,ERROR,WARNING,INFO,DEBUG}] {vgg,openface,cosface_resnet,cosface_swinnet,arcface_resnet,arcface_swinnet} ...

Run a component

positional arguments:
  {vgg,openface,cosface_resnet,cosface_swinnet,arcface_resnet,arcface_swinnet}
                        Component help
    vgg                 vgg help
    openface            openface help
    cosface_resnet      cosface_resnet help
    cosface_swinnet     cosface_swinnet help
    arcface_resnet      arcface_resnet help
    arcface_swinnet     arcface_swinnet help

optional arguments:
  -h, --help            show this help message and exit
  -researchers RESEARCHERS
                        A comma separated list of researchers names to replace the defaults with. For example: "Name1,Name2"
  -log-level {CRITICAL,ERROR,WARNING,INFO,DEBUG}
```

After providing the facial recognition system:
```
usage: train.py vgg [-h] [-training-image-limit TRAINING_IMAGE_LIMIT] [-testing-image-limit TESTING_IMAGE_LIMIT]
                    dataset_directory researchers_directory weights_file output_directory {PUBFIG,LFW} {SMALL,LARGE}

positional arguments:
  dataset_directory     The directory for dataset images
  researchers_directory
                        The directory for the researchers images
  weights_file          The directory for the researchers images
  output_directory      The output directory
  {PUBFIG,LFW}          The dataset to use
  {SMALL,LARGE}         The size of the dataset to use

optional arguments:
  -h, --help            show this help message and exit
  -training-image-limit TRAINING_IMAGE_LIMIT
                        The number of training images for each class
  -testing-image-limit TESTING_IMAGE_LIMIT
                        The number of testing images for each class
```

To view the possible command line arguments, use `python -m advfaceutil.train --help` to output the usages
or `python -m advfaceutil.train vgg --help` to output the usages of a specific facial recognition system.

### Alignment Script
The alignment script can process either an image file or a directory of images and can create a new
directory containing the aligned images. This script can also use face augmentation to batch process
images and apply a face effect. The usage (given by `python -m advfaceutil.align --help`) is as follows:

```
usage: align.py [-h] [--overlay OVERLAY] [-visualise] [-crop CROP] [-face-predictor FACE_PREDICTOR] [-face-processor {DLIB,MEDIAPIPE}] [-overlay-model OVERLAY_MODEL]
                [-log-level {CRITICAL,ERROR,WARNING,INFO,DEBUG}]
                input output

positional arguments:
  input                 Specify the image or directory containing images to align
  output                Specify the file name if aligning just an image or the directory if aligning a directory

optional arguments:
  -h, --help            show this help message and exit
  --overlay OVERLAY     Specify the overlay image to place on top
  -visualise            Whether to output a visualisation image
  -crop CROP            How much to crop by default = 96
  -face-predictor FACE_PREDICTOR
                        The path to the face predictor model
  -face-processor {DLIB,MEDIAPIPE}
                        Which face processor to use
  -overlay-model OVERLAY_MODEL
                        The path to the model to use for augmentation. This can only be used if MediaPipe is selected as the face processor.
  -log-level {CRITICAL,ERROR,WARNING,INFO,DEBUG}
```

### Accessory Name2chmark
The usage of the accessory Name2chmark script (given by `python -m advfaceutil.Name2ch --help`) is as follows:
```
usage: Name2ch.py [-h] [-ao] [-om OVERLAY_MODEL] [-wp WEIGHTS_PATH] [-tp TRANSFER_PATH] [-sau] [-sal] [--no-class-image-limit] [--class-image-limit CLASS_IMAGE_LIMIT] [-srs] [-wc WORKERS]
                [-to TARGET_CLASS] [-b Name2CHMARK] [--log-level {CRITICAL,ERROR,WARNING,INFO,DEBUG}] [--verbose] [--researchers RESEARCHERS]
                dataset_directory researchers_directory output_directory {VGG,OPENFACE,COSFACE_RESNET,COSFACE_SWINNET,ARCFACE_RESNET,ARCFACE_SWINNET} {PUBFIG,LFW} {SMALL,LARGE} {DLIB,MEDIAPIPE}
                accessory_path non_adversarial_accessory_path base_class

Name2chmarking pipeline

positional arguments:
  dataset_directory     The directory for the dataset images
  researchers_directory
                        The directory for the researchers images
  output_directory      The output directory
  {VGG,OPENFACE,COSFACE_RESNET,COSFACE_SWINNET,ARCFACE_RESNET,ARCFACE_SWINNET}
                        The recognition architecture to use
  {PUBFIG,LFW}          The dataset to use
  {SMALL,LARGE}         The size of the dataset to use
  {DLIB,MEDIAPIPE}      The face processor to use to augment the images
  accessory_path        The path to the accessory to add to the images
  non_adversarial_accessory_path
                        The path to the non-adversarial accessory
  base_class            The class that the accessory was designed for

optional arguments:
  -h, --help            show this help message and exit
  -ao, --additive-overlay
                        This is optional and is only used if the face processor is dlib
  -om OVERLAY_MODEL, --overlay-model OVERLAY_MODEL
                        This is optional and is only used if the face processor is mediapipe
  -wp WEIGHTS_PATH, --weights-path WEIGHTS_PATH
                        The base weights for the recognition model
  -tp TRANSFER_PATH, --transfer-path TRANSFER_PATH
                        The transfer weights for the recognition model
  -sau, --save-augmented-images
                        Optionally save the augmented images
  -sal, --save-aligned-images
                        Optionally save aligned images
  --no-class-image-limit
                        Choose not to limit the number of images per class
  --class-image-limit CLASS_IMAGE_LIMIT
                        Optionally limit the number of images per class
  -srs, --save-raw-statistics
                        Optionally save the raw statistics
  -wc WORKERS, --workers WORKERS
                        Optionally set the number of workers to use to execute the Name2chmark. Using 0 will causethe Name2chmark to execute sequentially. Not setting this will use as many processes as
                        possible
  -to TARGET_CLASS, --target-class TARGET_CLASS
                        The class that we are supposed to target
  -b Name2CHMARK, --Name2chmark Name2CHMARK
                        Specify a Name2chmark to use. For example: 'Baseline' will use the baseline Name2chmark.
  --log-level {CRITICAL,ERROR,WARNING,INFO,DEBUG}
  --verbose, -v
  --researchers RESEARCHERS
                        A comma separated list of researchers names to replace the defaults with. For example: "Name1,Name2"
```

### Demonstration Software
To run the demonstration software, simply run `python -m advfaceutil.demo` whilst connected
to your virtual environment. This will launch a new window which opens the webcam
and can allow the user to load in textures that can be placed on someone's face.

#### Dlib vs MediaPipe
This project contains two different methods of face augmentation and alignment.
Dlib is the more traditional approach and uses 68 landmarks to find a face without estimating
depth information. Therefore, augmentation with Dlib consists of performing an affine transformation
that places the loaded image onto a face.

MediaPipe on the other-hand uses 468 landmarks to give a better estimation of the face geometry
and includes a depth-estimation. Therefore, with a bit more work using [**PyFaceAR**](https://github.com/The-Adversaries/PyFaceAR),
we are able to 3D render a model onto a face. For us, this allows us to create adversarial face masks
that wrap around an individuals face in a much more realistic manner than possible with Dlib.

### Survey Script
The survey script can be used to generate augmented images to be used in the inconspicuousness survey.
By providing a directory of accessories, each accessory is placed on the face of a person from the dataset
and saved to the given output directory. The usage (given by `python -m advfaceutil.survey --help`) is as follows:

```
usage: Generate augmented images for a survey. [-h] [-n NUMBER] [-c ADD_CLASS]
                                               [-sp] [-p ADD_PATH]
                                               [-fp {DLIB,MEDIAPIPE}]
                                               [-om OVERLAY_MODEL]
                                               [--log-level {CRITICAL,ERROR,WARNING,INFO,DEBUG}]
                                               [--verbose]
                                               [--researchers RESEARCHERS]
                                               accessory_directory
                                               output_directory
                                               dataset_directory
                                               researchers_directory
                                               {PUBFIG,LFW} {SMALL,LARGE}

positional arguments:
  accessory_directory   The directory containing the accessories to apply to
                        faces
  output_directory      The directory to save the augmented images
  dataset_directory     The directory containing the dataset images
  researchers_directory
                        The directory containing the researchers images
  {PUBFIG,LFW}          The dataset to use
  {SMALL,LARGE}         The size of the dataset

optional arguments:
  -h, --help            show this help message and exit
  -n NUMBER, --number NUMBER
                        The number of classes to augment for each accessory
  -c ADD_CLASS, --add-class ADD_CLASS
                        The class names to augment for each accessory. If
                        given, this will override the number of classes
                        argument.
  -sp, --same-paths     Whether to use the same paths for each accessory
  -p ADD_PATH, --add-path ADD_PATH
                        The paths to the images to augment for each accessory.
                        If given, this will override the number of classes and
                        the class names arguments.
  -fp {DLIB,MEDIAPIPE}, --face-processor {DLIB,MEDIAPIPE}
                        Which face processor to use
  -om OVERLAY_MODEL, --overlay-model OVERLAY_MODEL
                        The overlay model to use
  --log-level {CRITICAL,ERROR,WARNING,INFO,DEBUG}
  --verbose, -v
  --researchers RESEARCHERS
                        A comma separated list of researchers names to replace
                        the defaults with. For example: "Name1,Name2"
```

## Datasets
This project contains three basic image loaders used in the creation of our facial recognition systems
as well as our adversarial attacks. These include:
- **Individual dataset:** this dataset supports loading images for a single class from a dataset directory.
For example, if given the class name "Name1", only images with "Name1" in their file name will be included
when reading all files in the given directory.
- **Multiclass dataset:** this dataset supports loading images for multiple classes within a dataset.
This is similar to the individual dataset albeit that we now provide a list of classes that we expect to
find and only include an image if the start of the file name matches the name of the class.
- **PubFig Dataset:** this dataset builds on the multiclass dataset to support the loading of the PubFig
dataset for different sizes. In our case we have two dataset sizes, `SMALL` and `LARGE` with 10 and 145 classes
respectively.
- **LFW Dataset:** this dataset builds on the multiclass dataset to support the loading of the LFW
(Lone Faces in the Wild) dataset. Similar to PubFig we have two sizes, `SMALL` and `LARGE` with 10 and 145 classes.

## Facial Recognition Systems
This project contains the implementation of two Facial Recognition Systems:
1. VGG16
2. OpenFace
3. CosFace ResNet
4. CosFace SwinNet
5. ArcFace ResNet
6. ArcFace SwinNet

### Loading a Facial Recognition System
Loading a facial recognition system can be achieved in two ways, using the `RecognitionArchitectures`
enum or using the `RecognitionArchitecture#contruct()` method on the corresponding network.

Loading a network using the `RecognitionArchitectures` enum can be done as follows:

```python
from advfaceutil.recognition import RecognitionArchitectures
from advfaceutil.datasets.faces import FaceDatasets, PubFigDatasetSize

vgg = RecognitionArchitectures.VGG.construct(FaceDatasets.PUBFIG, PubFigDatasetSize.SMALL)
```

Alternatively we can call `construct()` on the network directly:

```python
from advfaceutil.recognition import Vgg
from advfaceutil.datasets.faces import FaceDatasets, PubFigDatasetSize

vgg = Vgg.construct(FaceDatasets.PUBFIG, PubFigDatasetSize.SMALL)
```

Both are equivalent although the first is more commonly used. The first also allows us to enumerate or
loop through each network using `for arch in RecognitionArchitectures: ...`.
It is also possible to load in specific weights by passing a parameter to the construct method.

_Note: `RecognitionArchitectures#construct()` and `RecognitionArchitecture#construct()` have the same interface._

To then use the network you can either call the `classify(image)` function which will return the
class number with the largest softmax value and can preprocess the image for you, or use the
network directly, for example for the above network `vgg(image)`. This is possible since a
recognition architecture is a PyTorch module.

### Adding a Facial Recognition System
Adding a facial recognition system consists of a few steps.
1. Create the corresponding subpackage that is named according to the network (e.g., `vgg`).
2. In the `__init__.py` file in that package, implement a subclass of `RecognitionArchitecture` (which is
a PyTorch module). You will want to overwrite the forward pass of this module and the `construct()`
function which takes in the dataset size and network weights. It will likely be useful to refer
to the other implementations for what this function may look like.
3. Add a `training.py` file in that package that can handle the training of the network.
This should contain a class which inherits from `Component` (imported from `utils.component`) and
should contain a `run(args)` method which is where we actually perform the training loop.
The parameter we pass as args will be a `RecognitionTrainingArguments` (imported from `recognition.training`)
as this provides all the parameters that you will likely need to train your network.
If you wish to add extra arguments to the run function, see the [Components](#components) section below.
4. Now that we have implemented our network and training loop, we need to add the network to the
list of networks in the `recognition/__init__.py` file. Simply import your network (where the other networks
are imported to prevent a circular reference) and add your network using `NETWORK = (Network,)`.
This will allow us to construct the network using `RecognitionArchitectures.NETWORK.construct()` and
will also be added to the demonstration software automatically.
5. Finally, we can add our training loop to the training script by adding a value to the enum found
in `train.py`.

To train a facial recognition system, provided it has been added to the enum in `train.py`, we can run:
```
python train.py NETWORK "/path/to/dataset" "/path/to/researchers" "/path/to/base_weights.pth" "/path/to/output" "DATASET" "SIZE"
```

### Using Face Augmentation
To use face augmentation like we do in our demonstration software, we need to create the corresponding
`FaceProcessor`. This can be done in a similar manner to how we create a facial recognition network.
We can construct a face processor in one of the two following ways:

Using the `FaceProcessors` enum:

```python
from advfaceutil.recognition.processing import FaceProcessors

processor = FaceProcessors.MEDIAPIPE.construct()
```
Using the individual processor:

```python
from advfaceutil.recognition.processing import MediaPipeFaceProcessor

processor = MediaPipeFaceProcessor()
```

A face processor has a few key functions:
- `detect_faces(image)`: When given an image, find all the faces and return a list of `FaceDetectionResult`s
  (these contain bounding boxes and landmark information for each face).
- `show_bounding_boxes(image, detections, colour, thickness)`: When given an image and detections (either one
  or a list of detection results), draw a bounding box around each face.
- `show_landmarks(image, detections, colour, radius)`: When given an image and a detections (either one
  or a list of detection results), draw the landmarks that the corresponding face processor has found.
- `detect_largest_face(image)`: When given an image, find the largest face (by bounding box area).
- `augment(image, options, detections)`: When given an image, the augmentation options and optionally the detections
  to augment, apply the augmentation. The specific augmentation is defined by the options and includes the
  image to overlay, any additive blending (for dlib) or the face model to use (for mediapipe).
- `align(image, crop_size, detections)`: When given an image and the detections, align each detected face
  (or just one if only one is found) and output the aligned image which is cropped to be `crop_size`x`crop_size`.

## Components

One concept used throughout the project is the idea of "Components" where each component is essentially
a run script which can take in some arguments. By defining our own `ComponentArguments`, we can
say how we can read command line arguments and construct the corresponding argument object.
Then, using a `Component`, we can pass in the arguments and read them and do what we want with them.
We can then link multiple components together into one run script and use the `run()` function to
allow us to dynamically switch which command line arguments we need for which component. An example is as follows:

```python
from advfaceutil.utils.component import Component, ComponentArguments, ComponentEnum, run
from argparse import Namespace, ArgumentParser
from dataclasses import dataclass


@dataclass
class CustomArgs(ComponentArguments):
  value1: int

  @staticmethod
  def parse_args(args: Namespace) -> "CustomArgs":
    return CustomArgs(args.value1)

  @staticmethod
  def add_args(parser: ArgumentParser) -> None:
    parser.add_argument("value1")


class Component1(Component):

  @staticmethod
  def run(args: CustomArgs) -> None:
    print("Component 1:", args.value1)


class Component2(Component):

  @staticmethod
  def run(args: CustomArgs) -> None:
    print("Component 2:", args.value1)


class Components(ComponentEnum):
  COMP1 = (CustomArgs, Component2,)
  COMP2 = (CustomArgs, Component2,)


if __name__ == '__main__':
  run(Components)
```
Here we have defined our own command line arguments and two components that use those arguments.
By defining a `ComponentEnum` and passing that into `run()`, we are able to run the two different
components using:
```shell
python example.py COMP1 10
```
which would output "Component 1: 10".
