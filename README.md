#Food Classification App



This project had as aim to develop a deep learning model able to recognize from images, several food categories. It used a Kaggle dataset with 24k images distributed across 34 food categories, contemplating Western to Indian foods.
This was a particularly imbalanced dataset since some categories had 144 images and others 1500. Given the imbalance was adopted the F1-Score, as it is a that can capture both false positives and false negatives, making it more suitable for this type of dataset.  Taking this dataset as a starting point it was developed two models, one created by "scratch" and other by using a pre-trained model (VGG 16). The F1-Scores achieved were respectively, 55% and 77%. Both models used the standard Colab GPU for training, with a training of around 3 and 1 hour respectively. 
Finally, using the most performant VGG model, it was created a Streamlit WebApp, for easy use of the model capabilities.

From "scracth" model performance
![image](https://github.com/malasiaa/FoodClassificationCNN_Streamlit/assets/144847430/301d06ae-86b3-4b75-8847-70177c4d28e4)

VGG model performance
![image](https://github.com/malasiaa/FoodClassificationCNN_Streamlit/assets/144847430/ec7a011e-ea47-43fd-a564-70a1c2bc9e29)


WebApp: https://foodclassification-vgg.streamlit.app/
Kaggle dataset: https://www.kaggle.com/datasets/harishkumardatalab/food-image-classification-dataset/data


PretrainedModel_VGG - Transfer learning of the VGG16 model.

finalmodel_batch128 - From "scratch" CNN.

AGG_WebApp - requirements.txt and vgg_foodclass.h5 

![image](https://github.com/malasiaa/FoodClassificationCNN_Streamlit/assets/144847430/2c05aed1-3c5f-432e-884e-d1d847e11fd6)






