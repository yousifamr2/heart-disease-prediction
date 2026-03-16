# ❤️ Heart Disease Prediction System using Machine Learning

## 📌 Overview

The **Heart Disease Prediction System** is an intelligent web-based platform designed to predict the risk of cardiovascular disease using machine learning and routine laboratory test data.

The system integrates **patients, medical laboratories, and healthcare services** into a unified platform that supports **early detection and preventive healthcare**.

By analyzing clinical features such as **age, cholesterol levels, blood pressure, and chest pain type**, the system classifies patients into **Low Risk** or **High Risk** categories and provides explainable results with medical recommendations.

---

# 🚨 Problem Statement

Cardiovascular diseases remain the **leading cause of death worldwide**, particularly in developing countries.

Many patients discover heart disease **only after severe symptoms appear**, which increases mortality rates and treatment costs.

Existing heart risk calculators have several limitations:

* Require **manual data entry**
* Do **not store patient history**
* Lack **system integration with laboratories**
* Provide **limited explanation of predictions**

This project aims to build an **automated, explainable, and integrated heart disease prediction system**.

---

# 🎯 Project Objectives

### Main Objective

Develop a **web-based intelligent system** that predicts heart disease risk using machine learning and laboratory test data.

### Specific Objectives

* Provide **early prediction of heart disease risk**
* Integrate **medical laboratories through APIs**
* Deliver **explainable AI predictions**
* Support **preventive healthcare decisions**
* Recommend **nearby hospitals for high-risk patients**

---

# ⚙️ System Features

## 🧑‍💻 User Features

* User registration using **National ID**
* Secure **authentication and login**
* View laboratory test results
* Request **heart disease prediction**
* View prediction results and explanations

## 🧪 Laboratory Features

* Upload patient test results via **secure APIs**
* Link test results to patients using **National ID**

## 🤖 Machine Learning Features

* Heart disease prediction using ML models
* Risk classification:

  * **Low Risk**
  * **High Risk**
* Feature importance explanation using:

  * **SHAP**
  * **LIME**
* Optimized model accuracy: **95%**

## 🏥 Recommendation System

* High-risk patients receive **hospital recommendations**
* Integration with **Google Maps** for location guidance

---

# 🧠 Machine Learning Pipeline

1️⃣ Data Collection
Datasets used include:

* UCI Heart Disease Dataset
* Cleveland Heart Disease Dataset
* Statlog Heart Disease Dataset

2️⃣ Data Preprocessing

* Handling missing values
* Feature scaling
* Encoding categorical variables

3️⃣ Model Training
Several machine learning models were evaluated:

* Logistic Regression
* Naive Bayes
* KNN
* Support Vector Machine
* Decision Tree
* Random Forest
* Gradient Boosting
* XGBoost
* LightGBM
* CatBoost
* Neural Networks

4️⃣ Model Optimization

* Hyperparameter tuning
* Cross-validation
* Model comparison

5️⃣ Final Model
The final optimized model achieved **95% accuracy**.

---

# 🧩 System Architecture

![System Architecture](images/System_Architecture.png)

The backend communicates with the ML model through API requests to generate predictions.

---

# 🛠️ Technologies Used

## Frontend

* React.js
* Modern CSS frameworks (Bootstrap / Tailwind)

## Backend

* Node.js
* RESTful APIs

## Machine Learning

* Python
* Scikit-learn
* TensorFlow

## Database

* MongoDB / SQL Database

## Security

* JWT Authentication
* HTTPS secure communication

---

# 📁 Project Structure

```
heart-disease-prediction/
│
├── frontend/        # React user interface
├── backend/         # Node.js API
├── database/        # Database models and connection
├── ai/              # Machine learning pipeline
│
└── docs/            # Documentation and diagrams
```

---

# 📊 Expected Impact

* Early detection of cardiovascular diseases
* Reduced healthcare costs
* Data-driven medical decision support
* Increased accessibility to heart disease screening tools
* Support preventive healthcare systems

---

# 👨‍💻 Project Team

* Omar Ahmed Mohammed Bahaa EL-Din
* Youssef Amr Saeed
* George Anwar Maksadallah
* Marwan Yaser Elserafy
* Samar Hamza Mbry
* Youssef Mohamed Bassiony
* Abdelrahman Esam Sheredah
* Ranim Mohamed Elsayed

Faculty of Computers & Information Technology
Egyptian E-Learning University

---

# 🎓 Supervision

Supervisor
Dr. Mohamed Ahmed Hussein

Assistant Supervisor
Eng. Mohamed Abdelmawgoud Mostafa Khedr

---

# 🔒 License

This project is licensed under the **MIT License** and is intended for **educational and research purposes only**.

The system should not be used for real medical diagnosis without proper clinical validation.

---

# 🚀 Future Improvements

* Mobile application integration
* Integration with real hospital systems
* Real-time medical data analysis
* Larger healthcare datasets
* Advanced deep learning models
