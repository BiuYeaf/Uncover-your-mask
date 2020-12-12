# Uncover-your-mask
GANs for recovering the human face

Our project is to fulfill a target that Given human face with masks, then we simply let the model train to learn some features from the original face except from the mask area. Finally we can teach the machine to try their best to uncover the mask and estimate what it will look like for the original face.


<div align=center>
  <img width="50" height="50" src="https://github.com/BiuYeaf/Uncover-your-mask/blob/main/whole_images_model/161_train.png"/>
</div>




# Dataset

<div align=center>
  <img width="50" height="50" src="https://github.com/BiuYeaf/Uncover-your-mask/blob/main/images/face.png"/>
</div>
vggface2 library



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

In summary, our final model, the double discriminator model, is capable of recovering faces under the mask in different positions with less distortion.
Due to lacking diversity samples, our model for test selfies of ourselves, which is Asian, is not satisfactory. But it is acceptable for reconstructing western appearance. Besides, our model does not have enough evaluating process, and may have some problems when we put it into a real environment. So what we need to do are things below:

1. Train our model in a more balanced dataset. It means that we need all kinds of races in data and faces in different angles, like a side face, since there are some problems to generate side faces.
2. Find more extreme case of failing to enhance the robustness of our model.
3. Using Poisson Image Editing to make facial features more natural and symmetric.

![image](https://github.com/BiuYeaf/Uncover-your-mask/blob/main/result01.png)
![image](https://github.com/BiuYeaf/Uncover-your-mask/blob/main/result02.png)
