import streamlit as st
import cv2
import numpy as np
from PIL import Image

from Project_Utils.utils import arrToPIL, getCurrentLocation
from Project_Utils.config import initializeSessionVariables
from Project_Utils.classifiers import predict_image_class


def main():
    st.set_page_config(layout="wide")
    initializeSessionVariables(st)
    st.title("Binary Pothole Classification")

    def confirmImage(image: np.ndarray, st: st):
        st.session_state["selectedImage"] = image
    
    def getSelectedImage() -> np.ndarray:
        return st.session_state["selectedImage"]
    
    def resetPage():
        st.session_state["DataCollectionStage"] = True
        st.session_state["CameraInput"] = False
        st.session_state["PredictionProgress"] = None
        st.session_state["selectedImage"] = None

    if(st.session_state["DataCollectionStage"] == True):
        # Create a layout with three columns
        col1, col2, col3 = st.columns([8, 2, 8])

        # Column 1: Camera input
        with col1:
            st.subheader("Camera Input")
        
            if st.session_state["CameraInput"] == False:
                cam_button = st.button("Click Picture")
                if cam_button:
                    st.session_state["CameraInput"] = True
                    st.experimental_rerun()
            else:
                image_captured = st.camera_input("Camera Input")
                if image_captured is not None:
                    bytes_data = image_captured.getvalue()
                    numpy_array = np.frombuffer(bytes_data, np.uint8)
                    image_cv2 = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
                    img = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
                    # Now, 'image_cv2' is a NumPy array representing the image
                    confirmButtonCamera = col1.button("Confirm", key="CameraConfirm")
                    if(confirmButtonCamera):
                        confirmImage(img, st)
                        st.session_state["DataCollectionStage"] = False
                        st.experimental_rerun()

        # Column 2: Display "or"
        with col2:
            st.markdown("<h1 style='text-align:center;'>or</h1>", unsafe_allow_html=True)

        # Column 3: File picker for image input
        with col3:
            st.subheader("File Picker")
            file_picker = st.file_uploader("Choose a file", type=["jpg", "png", "jpeg"])
            if file_picker is not None:
                image = Image.open(file_picker)
                img_array = np.array(image)
                confirmButtonFile = col3.button("Confirm", key="FileImageConfirm")
                if(confirmButtonFile):
                    confirmImage(img_array, st)
                    st.session_state["DataCollectionStage"] = False
                    st.experimental_rerun()

    else:
        st.write("Selected Image")
        st.image(arrToPIL(getSelectedImage()))
        resetButton = st.button("Reset", key="Reset")
        predictButton = st.button("Predict", key="GeneratePredictions")

        if(resetButton):
            resetPage()
            st.experimental_rerun()

        if(predictButton):
            acc_model_id = "SVM_HOG"
            prediction = predict_image_class(getSelectedImage())
            st.title(" ")
            st.subheader("Predictions")
            pred = "Pothole Image"
            if(prediction[0]["HOG_LBP_Models"][acc_model_id] == 0): pred = "Plain Image"
            st.text(f'Image Class: {pred}')
            if(prediction[0]["HOG_LBP_Models"][acc_model_id] == 1):
                st.text("Segmented Output")
                st.image(prediction[2])
            expander = st.expander("More Info")
            with expander:
                st.write(f"Note: Model - {acc_model_id} is used for predicting image class since it has the highest accuracy of 91%")
                st.text("Predictions from various models")
                st.write(prediction[0])
                st.write(prediction[1])
            if(prediction[0]["HOG_LBP_Models"][acc_model_id] == 1):
                st.title(" ")
                st.subheader("Your Location")
                st.text("Your location with POTHOLE has been reported to the appropriate authorities")
                st.map(getCurrentLocation())
            

if __name__ == "__main__":
    main()