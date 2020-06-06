# Behavioral Cloning for Self Driving Car

---

### My project includes the following files:


- `model.py`: containing the script to create and train the model
- `model.h5` containing a trained convolution neural network
- `drive.py`: flash frame 
- ` utils.py` : helper functions
- ` video.py` : containing the script to create video
- `writeup.md`: summarizing the results
#### How to generate deep learning model
Install imagaug package, and then run model.py in terminal

```
pip install imgaug
python model.py
```

#### How to run car simulation

1. Enable GPU model and install ffmpeg & imgaug package in terminal 

```
$ sudo apt update
$ sudo apt install ffmpeg
$ pip install imgaug
```

2. Run the simulator and select autonomous mode. and then in terminal

```
$ python drive.py model.h5 run1
```

####  How to make video

Make the video from collected simulation data, the run command below in terminal.

```
$ python video.py run1
```


### Gathering Data the following:
#####  Simulator : [Udacity Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim)

1. Download and run Simulator in training mode and click record button.

2. Use keyboard to control the throttle signal and steering angle

3. the record data includes a folder of images as well as a csv file.

4. the columns of the csv data indicates ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']

   ###### Udacity also provide sample training data for behavior cloning project.

   Click on the link below to download it.

   - [Udacity Training Data](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) 

![](https://raw.githubusercontent.com/aaron7yi/behavioral_cloning/master/camera_images.png)
Notice: it seem the default data is not sufficient for training this deep learning model, the car simulation initially shows non-smooth turning in sharp corner. Additional data are recommend for training this model.

### Preparing Data

I have used the methods below for Preparing Data.

1. limit  the amount of 0 steering angle data in the dataset

2. append left and right camera images to the data set as well as apply 0.15 offset to corresponding steering angle label.

3. split dataset with 20% validation data & 80% training data.

4. shuffle the data set

![](https://raw.githubusercontent.com/aaron7yi/behavioral_cloning/master/histogram.png)

### processing images
1. cropping up the interested area in images 
2. apply GaussianBlur filter
3. normalizing each image data 
4. randomly augment the training data set by zooming, shifting, fliting，and etc.

![](https://raw.githubusercontent.com/aaron7yi/behavioral_cloning/master/zoomed_image.png)

![](https://raw.githubusercontent.com/aaron7yi/behavioral_cloning/master/paned_image.png)

![](https://raw.githubusercontent.com/aaron7yi/behavioral_cloning/master/brightness_altered_image.png)

![](https://raw.githubusercontent.com/aaron7yi/behavioral_cloning/master/flipped_image.png)


### Model architecture
My final model  is modified base on NiVADA model, and it consisted of the following layers:

| Layer           | Description                     | Input      | Output      |
| --------------- | ------------------------------- | ---------- | ----------- |
| Convolution 5x5 | Conv2D                          | 66x200x3   | 33x 100x 24 |
| Activation      | ELU activation                  | 33x100x24  | 33x100x 24  |
| Max pooling     | 1x1 stride, 2x2 window          | 33x 100x24 | 32x99,x24   |
| Convolution 5x5 | Conv2D                          | 32x 99x 24 | 16x50x36    |
| Activation      | ELU activation                  | 16x50x 36  | 16x 50x36   |
| Max pooling     | 1x1 stride, 2x2 window          | 16x 50x36  | 15x 49x 36  |
| Convolution 5x5 | Conv2D                          | 15x 49x36  | 8x25x48     |
| Activation      | ELU activation                  | 8x25x48    | 8x 25x48    |
| Max pooling     | 1x1 stride, 2x2 window          | 8x 25x 48  | 7x24x48     |
| Convolution 3x3 | Conv2D,  ELU activation         | 7x24x 48   | 5x22x 64    |
| Convolution 3x3 | Conv2D, ELU activation          | 5x22x 64   | 3x20x 64    |
| Flatten         | dimensions -> 1 dimension       | 3x20x 64   | 3840        |
| Fully Connected | Fully Connected, ELU activation | 3840       | 1000        |
| Dropout         | RELU activation                 | 1000       | 1000        |
| Fully Connected | Fully Connected, ELU activation | 1000       | 500         |
| Fully Connected | Fully Connected, ELU activation | 500        | 100         |
| Fully Connected | Fully Connected, ELU activation | 100        | 50          |
| Fully Connected | Fully Connected, ELU activation | 50         | 10          |
| Fully Connected | Fully Connected                 | 10         | 1           |

| Hyperparameters           | type   |
| :------------------------ | :----- |
| Optimizer                 | Adam   |
| batch generation per step | 100    |
| steps per epoch           | 300    |
| learning rate             | 0.0009 |
| Epochs                    | 15     |
| Dropout                   | 0.5    |

| My final model results | Percentage |
| ---------------------- | ---------- |
| Training Results       | 0.9697     |
| Validation Accuracy    | 0.9728     |

## Discussion of  modifying  & turning model

1.reduce learning rate from default 0.001 to 0.0009 to help achieve a better optimization of loss entropy.

2.dropout increase from default 0.1 to 0.5 to prevent overfitting

3.use batch generator and fit generator to reduce memory utilization for loading large data set .

4.randomly apply image augmentation to the training data set to robust deep learning model.

5.limit the amount of 0 steering angle data to improve the steering sensitivity of the model

6.since the autonomous mode simulation shows the car didn't performance good in shape truing track, two extra fully connection layers have been add to the model and then vehicle turning  getting more smoothing.

7.three max pooling layers have been add to the model to help better obtain image features by reducing image noise.

8.replace relu activation by elu activation to resolve drying relu problem during model training.

### References:

- [Nvidia's self-driving car publication](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)