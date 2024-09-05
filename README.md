# Final-Course-Project-part-1-MLSC-Data-Science-Machine-Learning-Course
# Predicting Hazardous NEOs (Nearest Earth Objects)

This project analyzes a dataset of Near-Earth Objects (NEOs) observed by NASA from 1910 to 2024 to predict whether an object is hazardous or not. The dataset contains over 338,000 records with various features like diameter, velocity, and distance from Earth. The objective is to build a machine learning model to predict the "is_hazardous" label.

## Project Structure

### 1. Import Libraries and Dataset
The dataset is loaded, and essential libraries for data manipulation, visualization, and modeling are imported, including:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `sklearn` for machine learning

**Dataset**: The dataset used in this project is the [NASA Nearest Earth Objects Dataset (1910-2024)](https://www.kaggle.com/datasets/ivansher/nasa-nearest-earth-objects-1910-2024/data), containing 338,199 records.

Make sure to **extract the dataset** from the provided zip file before running the notebook.

### 2. Data Preprocessing
To ensure the dataset is ready for analysis and modeling, the following preprocessing steps are performed:
- **Handling Missing Values**: Using `SimpleImputer` to fill missing data.
- **Feature Scaling**: Normalizing the numerical features like diameter, velocity, and distance.
- **Dealing with Class Imbalance**: The dataset is imbalanced (more non-hazardous objects than hazardous). Various strategies can be applied to handle this, such as class weights or SMOTE (currently not applied but recommended).

### 3. Exploratory Data Analysis (EDA)
The project uses `matplotlib` and `seaborn` to explore the dataset:
- **Histograms**: For visualizing distributions of numerical variables.
- **Box Plots**: To show the relationship between numerical features and the target (`is_hazardous`).
- **Correlation Heatmaps**: To understand the relationships between variables.

### 4. Model Training and Evaluation
Several models are trained on the preprocessed data, including:
- **Logistic Regression**: A baseline model for binary classification.
- **K-Nearest Neighbors (KNN)**: A distance-based classifier.

Evaluation metrics include:
- **Accuracy**: Overall model performance.
- **Precision and Recall**: To assess performance on hazardous and non-hazardous predictions.
- **F1-Score**: To balance precision and recall.
- **Confusion Matrix**: To understand true/false positives and negatives.

### 5. Predicting New Data
The model is capable of predicting the hazardous status of new NEOs. Example code is included for making predictions on new, unseen data.

### 6. Model Performance
Initial results show good accuracy for non-hazardous objects but room for improvement in detecting hazardous ones due to class imbalance.

### 7. Additional Improvements
To further improve the model:
- **Handle Class Imbalance**: Implement techniques like SMOTE, undersampling, or adjusting class weights.
- **Consider Additional Models**: Such as Random Forest or Gradient Boosting for potentially better performance.

## Instructions for Running the Project

1. Clone this repository from GitHub.
2. Extract the dataset file from the provided zip file.
3. Install the required libraries using the command:
   ```bash
   pip install -r requirements.txt

### Notes:
- The updated README includes details about the dataset, preprocessing, modeling, and evaluation.
- A note is added to ensure users extract the dataset from the zip file before running the notebook.

Let me know if you want to adjust any other details or aspects of the notebook!
