# Final-Course-Project-part-1-MLSC-Data-Science-Machine-Learning-Course

# Predicting Hazardous NEOs (Nearest Earth Objects)

This project analyzes a dataset of Near-Earth Objects (NEOs) observed by NASA from 1910 to 2024 to predict whether an object is hazardous or not. The dataset contains over 338,000 records with various features like diameter, velocity, and distance from Earth. The objective is to build a machine learning model to predict the "is_hazardous" label.

## Project Structure

### 1. **Import Libraries and Dataset**
The dataset is loaded, and essential libraries for data manipulation, visualization, and modeling are imported, including:
- `numpy` for numerical operations.
- `pandas` for data manipulation.
- `matplotlib` and `seaborn` for data visualization.
- `sklearn` for machine learning.

The dataset used in this project is the **NASA Nearest Earth Objects Dataset (1910-2024)**, containing 338,199 records. Make sure to extract the dataset from the provided zip file before running the notebook.

### 2. **Data Preprocessing**
To ensure the dataset is ready for analysis and modeling, the following preprocessing steps are performed:
- **Handling Missing Values**: Using `SimpleImputer` to fill missing data.
- **Feature Scaling**: Normalizing numerical features like diameter, velocity, and distance.
- **Dealing with Class Imbalance**: The dataset is imbalanced, with more non-hazardous objects than hazardous ones. Strategies such as class weights or SMOTE are recommended but currently not applied.

### 3. **Exploratory Data Analysis (EDA)**
The project uses `matplotlib` and `seaborn` to explore the dataset:
- **Histograms**: To visualize the distributions of numerical variables.
- **Box Plots**: To show the relationship between numerical features and the target variable (`is_hazardous`).
- **Correlation Heatmaps**: To understand relationships between variables and identify potential feature interactions.
Questions that have been answered by using the above kind of graph:
- **What is the distribution of hazardous vs. non-hazardous objects?**
- **How do numerical features (absolute_magnitude, estimated_diameter_min, estimated_diameter_max, relative_velocity, miss_distance) correlate with each other and with is_hazardous?**
- **What are the distributions of absolute_magnitude, estimated_diameter_min, estimated_diameter_max, relative_velocity, and miss_distance for hazardous vs. non-hazardous objects?**
- **How do pairs of features (miss_distance vs. relative_velocity, etc.) relate to each other and to the hazard classification?**
- **What are the distributions of features (like absolute_magnitude, estimated_diameter_max, estimated_diameter_min, relative_velocity, miss_distance) for hazardous and non-hazardous objects?**
- **What are the distributions and variances of features like absolute_magnitude, relative_velocity, etc., when split by is_hazardous?**

### 4. **Model Training and Evaluation**
Several models are trained on the preprocessed data to predict whether an NEO is hazardous:
- **Random Forest**
- **Decision Tree Classification**
- **k-Nearest Neighbors (KNN) Classification**
- **Logistic Regression**
- **Naive Bayes**

#### Evaluation metrics include:
- **Accuracy**: Overall model performance.
- **Precision and Recall**: To assess performance on hazardous and non-hazardous predictions.
- **F1-Score**: To balance precision and recall.
- **Confusion Matrix**: To understand true/false positives and negatives.

### 5. **Predicting New Data**
The model is capable of predicting the hazardous status of new NEOs. Example code is included for making predictions on new, unseen data.

### 6. **Model Performance**
Initial results show good accuracy for non-hazardous objects but room for improvement in detecting hazardous ones due to class imbalance. Future work could focus on improving the model's sensitivity to hazardous NEOs.

### 7. **Additional Improvements**
To further improve the model:
- **Handle Class Imbalance**: Implement techniques like SMOTE, undersampling, or adjusting class weights.
- **Consider Additional Models**: Such as Random Forest or Gradient Boosting for potentially better performance.

## Instructions for Running the Project

1. **Clone this repository**:
   ```bash
   git clone https://github.com/your-username/NEO-data-analysis.git
   ```

2. **Extract the dataset**: 
   Ensure you extract the dataset file from the provided zip file before running the notebook.

3. **Install the required libraries**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Jupyter Notebook**:
   ```bash
   jupyter notebook Final-Course_Project_Part1.ipynb
   ```

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Conclusion
This project provides a starting point for analyzing Near-Earth Objects, particularly focusing on predicting whether an object is hazardous. While the initial models show promise, improvements are needed to better handle class imbalance and improve predictions for hazardous NEOs.
