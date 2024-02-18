# Food Classification App
This project aimed to develop a deep learning model (CNN) capable of recognizing various food categories from images. It utilized a Kaggle dataset comprising 24,000 images spanning 34 food categories, ranging from Western to Indian cuisines.
The dataset was notably imbalanced, with some categories having as few as 144 images while others boasted 1,500. To address this imbalance, the F1-Score was employed as the evaluation metric. This metric is particularly effective in capturing both false positives and false negatives, making it an appropriate choice for this dataset.

Two models were developed using this dataset: one built from scratch and the other leveraging a pre-trained model (VGG16). The F1-Scores achieved by these models were 55% and 77%, respectively. Both models were trained on Google Colab's GPU, with the from-scratch model taking approximately 3 hours and the VGG16-based model taking around 1:30 hours of training.

Utilizing the more performant VGG16 model, a Streamlit WebApp was created to facilitate easy access to the model's capabilities.


#### WebApp: https://foodclassification-vgg.streamlit.app/

#### Kaggle dataset: https://www.kaggle.com/datasets/harishkumardatalab/food-image-classification-dataset/data

## Model performance
### From "Scratch" Model:

![image](https://github.com/malasiaa/FoodClassificationCNN_Streamlit/assets/144847430/301d06ae-86b3-4b75-8847-70177c4d28e4)

### VGG Model:

![image](https://github.com/malasiaa/FoodClassificationCNN_Streamlit/assets/144847430/ec7a011e-ea47-43fd-a564-70a1c2bc9e29)


## Resources
PretrainedModel_VGG - Transfer learning of the VGG16 model.
finalmodel_batch128 - From "scratch" CNN model.
AGG_WebApp - Script used for the development of the streamlit app, which uses requirements.txt (all the dependencies required to run the model) and vgg_foodclass.h5 (VGG model weights). 

