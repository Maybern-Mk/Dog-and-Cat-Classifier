# **AI Image Classification System**

## **Overview**  
This project develops a deep learning-based image classification system using a Convolutional Neural Network (CNN) trained on the CIFAR-10 dataset.  

The system predicts the class of an input image and provides confidence scores, enabling users to understand how the model interprets visual data. The project covers the complete deep learning lifecycle, including data preprocessing, model training, evaluation, and deployment through a Streamlit web application.

---

## **Problem Statement**  
Image classification is a fundamental problem in computer vision with applications in automation, surveillance, healthcare, and more.  

The objective of this project is to build an accurate and efficient image classification model that can identify objects in images and provide reliable predictions through an interactive interface.

---

## **Dataset**  
- Dataset: CIFAR-10  
- Classes: 10 categories of objects  

### **Target Variable**  
- Image Class (Multiclass Classification)  

### **Classes**  
- Airplane  
- Automobile  
- Bird  
- Cat  
- Deer  
- Dog  
- Frog  
- Horse  
- Ship  
- Truck  

---

## **Tools and Technologies**  
- **Python**  
- **NumPy** for numerical operations  
- **TensorFlow / Keras** for deep learning  
- **PIL (Pillow)** for image processing  
- **Streamlit** for deployment  

---

## **Exploratory Data Analysis**  
- Understanding dataset structure and class distribution  
- Visualization of sample images  
- Checking class balance  
- Identifying patterns in image data  

---

## **Data Preprocessing**  

### **Image Processing**  
- Resizing images to 32x32 pixels  
- Converting images to RGB format  
- Normalizing pixel values (0 to 1 range)  

### **Input Preparation**  
- Converting images into NumPy arrays  
- Expanding dimensions for model input  
- Handling edge cases (e.g., RGBA images)  

---

## **Deep Learning Model**  

### **Model Type**  
- Convolutional Neural Network (CNN)  

### **Architecture Highlights**  
- Convolutional layers for feature extraction  
- Pooling layers for dimensionality reduction  
- Fully connected layers for classification  
- Softmax output layer for probability distribution  

---

## **Model Evaluation**  

### **Metrics Used**  
- Accuracy  
- Loss  

### **Result**  
The trained CNN model achieves strong performance on the CIFAR-10 dataset and generalizes well to unseen images.

---

## **Final Model**  
- Model: CNN Classifier  
- Saved as: `cnn_cifar10_best_model.h5`  

### **Additional Artifacts**  
- Class label mapping for predictions  
- Preprocessing logic embedded in the application  

---

## **Prediction Pipeline**  
- Accepts image input from user  
- Applies preprocessing automatically  
- Generates:  
  - Predicted class  
  - Confidence score  
  - Top 3 predictions with probabilities  

---

## **Streamlit Application**  

An interactive web application is included for real-time predictions.

### **Features**  
- Image upload interface  
- Real-time prediction  
- Confidence score display  
- Top predictions visualization  
- Clean and responsive UI  

### **Run the App**  
```bash
streamlit run pics_app.py:
pip install numpy tensorflow pillow streamlit
