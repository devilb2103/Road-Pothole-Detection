import os
import numpy as np
import pandas as pd
import cv
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import joblib

from Project_Utils.utils import getGrayColorImg

segmentation_model_path = "Models/UNet_Models/unet_segmentation_pothole.keras"

def predict_and_display_from_image_path(imgarr: np.ndarray, model_path):
    def dice_coef(y_true, y_pred, smooth=1):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    def dice_coef_loss(y_true, y_pred):
        return -dice_coef(y_true, y_pred)

    # Load and preprocess the image
    img = imgarr / 255  # Read the image in grayscale and normalize
    original_img = img.copy()            # Copy for display purposes
    img = cv2.resize(img, (128, 128))    # Ensure the image is resized to the input shape
    img = np.expand_dims(img, axis=-1)   # Add channel dimension
    img = np.expand_dims(img, axis=0)    # Add batch dimension

    loaded_model = load_model(model_path, custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
   
    prediction = loaded_model.predict(img)

    return prediction

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
import pickle

def resize_image(image_array):
    if image_array is not None:
        image_array = cv2.resize(image_array, (645, 645), interpolation=cv2.INTER_AREA)
        return image_array
    else:
        print(f"Failed to read image")

def convert_image_to_grayscale(image_array):
    if(len(np.array(image_array).shape) <= 2):
        return image_array
    return cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

def extract_hog_features(image_array):
    features, _ = hog(
        image_array,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=True
    )
    return features

def pad_lbp_features(test_lbp, max_length=10):
    if isinstance(test_lbp, (list, np.ndarray)):
        if len(test_lbp) < max_length:
            pad_width = max_length - len(test_lbp)
            padded_lbp = np.pad(test_lbp, (0, pad_width), 'constant')
            return padded_lbp
        else:
            return test_lbp
    else:
        raise ValueError("Input 'test_lbp' must be a list or numpy array.")

def extract_lbp_features(image_array, radius=1, n_points=8):
    lbp_image = local_binary_pattern(image_array, n_points, radius, method='uniform')
    lbp_hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-8)
    lbp_hist = pad_lbp_features(lbp_hist)
    return lbp_hist

def pca_hog(hog_feature):
    with open("Models/PreProcessing_Models/PCA_hog_lbp.pkl", "rb") as file:
        pca = pickle.load(file)
    hog_feature = pca.transform([hog_feature])[0]
    return hog_feature

def extract_features_from_image(image_array, radius=1, n_points=8):
    image_array = resize_image(image_array)
    grayscale_image = convert_image_to_grayscale(image_array)
    hog_features = extract_hog_features(grayscale_image)
    hog_features_pca = pca_hog(hog_features)
    lbp_features = extract_lbp_features(grayscale_image, radius, n_points)
    return hog_features, lbp_features ,hog_features_pca

def predict_class(image_array):
    hog_features, lbp_features, hog_features_pca = extract_features_from_image(image_array)
    
    model_paths = [
        
        ("Models/HOG_LBP_Models/svm_combined.pkl", "SVM_COMBINED"),
        ("Models/HOG_LBP_Models/rf_combined.pkl", "RF_COMBINED"),
        ("Models/HOG_LBP_Models/knn_combined.pkl", "KNN_COMBINED"),
        ("Models/HOG_LBP_Models/svm_hog.pkl", "SVM_HOG"),
        ("Models/HOG_LBP_Models/rf_hog.pkl", "RF_HOG"),
        ("Models/HOG_LBP_Models/knn_hog.pkl", "KNN_HOG"),
        ("Models/HOG_LBP_Models/svm_lbp.pkl", "SVM_LBP"),
        ("Models/HOG_LBP_Models/rf_lbp.pkl", "RF_LBP"),
        ("Models/HOG_LBP_Models/knn_lbp.pkl", "KNN_LBP")
    ]

    results = {}

    for model_path, feature_type in model_paths:
        with open(model_path, 'rb') as file:
            Model = pickle.load(file)
        
        if "HOG" in feature_type:
            features = hog_features_pca
        elif "LBP" in feature_type:
            features = lbp_features
        elif "COMBINED" in feature_type:
            features = np.concatenate((hog_features_pca, lbp_features))
        else:
            continue 

        predicted_class = Model.predict([features])
        results[feature_type] = predicted_class[0]

    return results

def splitt(x):
    return x.split()

def resizeImage(image: np.ndarray, size: tuple) -> np.ndarray:
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

def extract_sift_features(image):
    sift = cv2.SIFT_create()
    key_points, descriptors = sift.detectAndCompute(image, None)
    return key_points, descriptors

def generateDescriptorMatrices(image) -> pd.DataFrame:
    pca = joblib.load('Models/PreProcessing_Models/pca_77.joblib') 
    KMeansModel = pickle.load(open("Models/PreProcessing_Models/kmeans_445.pickle", "rb"))
    tfidf_vect = pickle.load(open("Models/PreProcessing_Models/tfidf_445.pickle", "rb"))
    
    SVMModel = joblib.load('Models/SIFT_Models/SVM_TFIDF_445.joblib')
    descriptor_list = []
    class_list = []
    image_class = []

    # extract descriptors from image
    keypoints, descriptor = extract_sift_features(image)

    # skip image because pca cannot be done
    if(descriptor.shape[0]) < 77:
        return {"SIFT - SVM_TFIDF": "Image Does not have enough Detail"}

    # add image descriptors
    descriptor_list.append(descriptor)
    
    for i in descriptor_list:
        reduced_descriptors = pca.fit_transform(i)
        class_labels = KMeansModel.predict(reduced_descriptors)
        class_list.append(class_labels)
    
    init = np.zeros(445, dtype=int)
    for i in class_list:
        init[i] += 1

    tfidf_matrix = tfidf_vect.fit_transform([init])

    prediction = SVMModel.predict(tfidf_matrix)
    print(prediction)
    if(prediction[0] == "Pothole"):
        prediction[0] = 1
    else:
        prediction[0] = 0
    

    return {"SIFT - SVM_TFIDF": prediction[0]}

def SIFT_Predict(image):
    resized = resizeImage(image, (256, 256))
    return generateDescriptorMatrices(resized)

def predict_image_class(image_color: np.ndarray):
    color_arr, gray_arr = getGrayColorImg(image_color)
    HOG_LBP_prediction = predict_class(color_arr)
    SIFT_prediction = SIFT_Predict(gray_arr)
    segmentedImage = None
    if(HOG_LBP_prediction["SVM_HOG"] == 1):
        segmentedImage = predict_and_display_from_image_path(gray_arr, segmentation_model_path)
    return [{"HOG_LBP_Models": HOG_LBP_prediction}, {"SIFT Model": SIFT_prediction}, segmentedImage]