##Writeup Template


**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

- Used the `hog` function from `skimage.feature` to obtain hog. Below is the wrapper which is used to derive HOG feature, 

```
# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features
```

- Resorted to the validation score of the machine learning model on a sub-sample of the data set to predict the best set of parameters for HOG.
- Initially when RGB color space was tried on a model which was trained on 5000 images, the accuracy on validation set was not going beyond 93%.
- But with trying various color the accuracy scored the best with `YCrCb` color space. It seemed to help the model distuigh cars and non cars better.
- The same experiement was tried with other parameters of HOG too.
- After trying various combinations of features against the trained model, the prediction accuracy was its best for the following paramters, and these 
  were chosen as final set of parameters.
 ```
 colorspace = 'YCrCb' 
 orient = 8
 pix_per_cell = 8
 cell_per_block = 2
 hog_channel = 'ALL' 
 spatial_size = (16, 16)
 hist_bins = 32
 hist_range=(0, 256) 	
 ```


####2. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).
- Used a SVM classifier.
- Used GridSearchCV to obtain the optimal kernel, C and gamma combination for the SVM classifier.[Here](https://github.com/hackintoshrao/vehicle-detection-and-tracking/blob/4260c3bf66a74580ebbef71f30c425a306a260a5/vehicle-detect.py#L133) is the code snippet of how GridSearchCV was used to obtain the optimal combination of paramters.
- Pamaters `kernel='linear', C=0.001, gamma=0.001` is used to train the SVM classifier.
- Used Histogram of bins, HOG and raw pizels of scaled image was feature to train the model.
- Obtained over 99% accuracy on the validation set.
- Since the training was done on `.png` images the features were already scaled between 0 and 1.
- Standard Scaler from `sklearn.preprocessing` is used to scale the features appropriately.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?
- The search had to be performed only in the region of interest, this avoids lot of false detections.
- Used large scales to search near the bottom of the image, this is because the vehicle would appear larger near to the bottom of the image.
- Reduced scales are used to search in ranges which are close to the middle of the image, this is because the vehicle would smaller and smaller the further it moves.
- After Experimenting with various combinations of overlapping found the overlap of 0.5 to be a fair value for reasonable detection.


####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

- By trying various color channels, `YCrCb` channel seemed to be performing the best with the classifier.
- Using GridSeachCV found the optimal values of kernel, C and gamma values to be used.
- The folder `./result/first_detections/ contains multiple overlapping detections which were found after the initial round of search->classify->draw-boxes.
- The final results with heap map and labelling applied are saved in `result/final_result` folder.
- The trained model and pipeline performed reasonably well on the test images, so over optimization was not necessary.

### Video Implementation
- Here is the [Youtube link](https://youtu.be/2EZ6I_J4FQc) to the video of the detection and tracking, its also saved in `./result/video/project_video_track.mp4`.
- Series of sub clips with low level of robustness are saved in `./result/video/bad_video/`.
- The folder `./result/debug/` contains screenshots taken while debugging the code.
- The video contains zero false positives, the pipeline is very robust.
- Here are the techniques used for robust detection and tracking.
	- Create heapmap around the detections from search and then label them to obtain a single box of detection.
	- Accepting detections only if they appear for 8 consecutive frames, This totally eliminates false positivies.
	- Average values of last 10 box detections are used for smoother transition of boxes across frames.
	- If the detection goes missing for next few frames the average is used to add boxes will the confidence dies off. 

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

- The first challenge was to obtain optimal set of parameters for HOG features extraction, with trying out the machine learning prediction for various combination of color channel and other parameters finalized upon the values Ive used.
- The other big challenge was to find the optimal combination of kernel, C and gamma features for training the model. Using GridSearchCV, was able to find the optimal combination of the parameters for SVM classifier.
- The biggest challenge was to remove false postivies from the video and to make the boxes trasist smoothe across frames, to achieve this, created a called `Detection()`. Using the class allowed a box to drawn only if a detection is achieved over 8 consecutive frames, this eliminated all false positives, and by taking the average box detection values over last 10 detections, got the boxes to be moved smoothly and gracefully across frames.

- The pipelines is not effective for vehicle coming head on, need to extract more frames per second to make it better.
- The pipeline finds it hard to distinguish between vehicles when they are very close to each other, this makes the detection not so robost in traffic situations, more sophisticated training using convnets might be necessary to achieve greater efficiency in prediction.
