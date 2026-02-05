# Student Lifestyle & Depression Prediction

## Overview
This project predicts the likelihood of **depression in students** based on lifestyle and academic features using **Machine Learning**.  
It demonstrates a full ML workflow including **data cleaning, feature scaling, model training, evaluation, threshold tuning, and feature importance analysis**.  

---

## Dataset
The dataset contains **100,000 student records** with the following features:

| Feature | Description |
|---------|-------------|
| Student_ID | Unique identifier for each student |
| Age | Age of the student |
| Gender | Male/Female |
| Department | Academic department |
| CGPA | Cumulative grade point average |
| Sleep_Duration | Hours of sleep per day |
| Study_Hours | Hours spent studying per day |
| Social_Media_Hours | Hours spent on social media per day |
| Physical_Activity | Weekly physical activity duration |
| Stress_Level | Self-reported stress level (1–10) |
| Depression | Target variable (True/False) |

---

## Features & Target
- **X (features):** Age, Gender, Department, CGPA, Sleep Duration, Study Hours, Social Media Hours, Physical Activity, Stress Level  
- **y (target):** Depression (1 = Yes, 0 = No)

---

## Methodology

1. **Data Cleaning:**  
   - Encoded categorical variables (Gender, Department)  
   - Checked for missing values and handled appropriately  

2. **Feature Scaling:**  
   - StandardScaler applied to numerical features to normalize data  

3. **Model Training:**  
   - Logistic Regression with `class_weight="balanced"` to handle class imbalance  
   - Threshold tuning applied to optimize **recall**  

4. **Evaluation Metrics:**  
   - **Accuracy** – overall correctness  
   - **Recall** – ability to detect students with depression  
   - **ROC-AUC** – threshold-independent class separation  
   - **Confusion Matrix** – visualize performance  

5. **Feature Importance:**  
   - Extracted coefficients to identify the most influential factors contributing to depression  

---

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/YourUsername/student-lifestyle-depression-prediction.git

