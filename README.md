# Uncover-your-mask
GANs for recovering the human face

Our project is to fulfill a target that Given human face with masks, then we simply let the model train to learn some features from the original face except from the mask area. Finally we can teach the machine to try their best to uncover the mask and estimate what it will look like for the original face.



![image](https://github.com/BiuYeaf/Uncover-your-mask/blob/main/whole_images_model/161_train.png)





# Dataset

vggface2 library
![image](https://github.com/BiuYeaf/Uncover-your-mask/blob/main/images/face.png)

We download the vggface2 library and use opencv to add masks on part of those pictures
![image](https://github.com/BiuYeaf/Uncover-your-mask/blob/main/images/mask01.png)
![image](https://github.com/BiuYeaf/Uncover-your-mask/blob/main/images/mask02.png)
![image](https://github.com/BiuYeaf/Uncover-your-mask/blob/main/images/mask03.png)
![image](https://github.com/BiuYeaf/Uncover-your-mask/blob/main/images/mask04.png)

We upload it on kaggle.
https://www.kaggle.com/momoxia/masked-people-dataset


# Detect model

![image](https://github.com/BiuYeaf/Uncover-your-mask/blob/main/images/detect01.png)
![image](https://github.com/BiuYeaf/Uncover-your-mask/blob/main/images/detect02.png)

Below is the result we tested on our testset. As we can see on the bottom picture I test,even though I just train it using our artificial dataset, it can still detect the mask  in reality. And the score of that one is 0.997. The machine is really sure that it is a mask.

![image](https://github.com/BiuYeaf/Uncover-your-mask/blob/main/images/detect03.png)


# Model Result
