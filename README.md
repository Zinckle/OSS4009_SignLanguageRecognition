
# Development of a Sign Language Recognition System

This projects presents the development of a Sign Language Recognition System utilizing convolutional neural networks (CNNs) to facilitate communication for individuals with hearing impairments. Our approach involves two CNNs: one for detecting static ASL signs and another for recognizing dynamic signs, including challenging letters like J and Z. Leveraging datasets such as Sign Language MNSIT (SMNIST) and EMNIST Letters (LMNIST), coupled with finger tracking techniques, our system enables users to practice ASL through accessible means like phones or computers.
## Authors
- Ray Huda
- Mitchell Zinck - [@Zinckle](https://www.github.com/Zinckle)


## How It Works
The needed datasets can be downloaded from here:
https://www.kaggle.com/datasets/datamunge/sign-language-mnist
https://www.kaggle.com/datasets/crawford/emnist

For EMNSIT, Use only the training data from the letter version
### CNN Training Set Up
When training the convolutional neural networks using TrainLetterDetect.py or TrainSLD.py, start by ensuring that the following libraries are installed:
  - matplotlib
  - seaborn
  - keras
  - klearn
  - pandas

we then need to ensure that the variables train\_data and test\_data are set to the relevant CSV files as strings depending on which detector is being trained. The files can be located anywhere but the path to the files needs to be included in the variable. The variable for epochs can be increased or decreased depending on the training duration desired and the parameters for datagen can be tweaked to increase robustness but this will come at the cost of training time and potentially overall accuracy.The CNN can then be run and the resulting network with trained weights will be saved as either lmnist.h5 or smnist.h5 depending on which network has been trained. 



### Detection Set Up
In order to run the SignLanguageDetection.py code, a webcam and a computer capable of running tensorflow is required. Start by ensuring that the following libraries are installed:
  - cv2
  - mediapipe
  - time
  - math
  - numpy
  - tensorflow
and that the lmnist.h5 and smnist.h5 models are available to be loaded and placed in the same directory as the SignLanguageDetection.py file. When attempting to detect sign language, ensure that the background is relatively clear with balanced lighting in the scene to get best results.