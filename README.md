# 🍽️ Malnutrition Pattern Identification Dashboard

A **Streamlit-based Public Health Analytics Dashboard** that identifies malnutrition patterns using nutritional dish metadata, demographic indicators (Age, BMI), district-level insights, and machine learning risk prediction.

This project is designed with a **rural nutrition + public health focus**, useful for healthcare NGOs, government health departments, and awareness programs.

---

## 🚀 Project Overview

Malnutrition remains a major issue in rural and semi-urban regions.  
This dashboard helps in:

- Identifying nutrition deficiency patterns  
- Understanding calorie/protein distribution across districts  
- Predicting malnutrition risk using ML models  
- Recommending high-protein dishes based on age group needs  

---

## 🎯 Key Features

✅ Interactive Streamlit Dashboard  
✅ Rural Nutrition Analytics  
✅ BMI + Age Group Classification  
✅ District-wise Heatmap Visualization  
✅ Dish Recommendation System  
✅ Machine Learning Risk Prediction  
✅ 15+ Graphs and Insights  
✅ Logistic Regression + Random Forest Models  

---

## 📊 Dashboard Modules (Tabs)

### 📋 Dataset Viewer
- Displays full dataset with filters enabled  

---

### 📊 Analytics (15+ Visualizations)

Includes graphs like:

- Age Group Distribution  
- BMI Category Analysis  
- Calories vs Age  
- Protein vs Age  
- Calories Histogram  
- Region-wise Nutrition Comparison  
- District-wise Protein/Calories  

---

### 🗺️ District Heatmap

- Average Calories by District  
- Nutrition Status Density Heatmap  

---

### 🍽️ Dish Recommendation System

Recommends top protein-rich dishes based on:

- Age Group  
- Calorie Requirement Level  
- Protein Deficiency Threshold  

---

### 🤖 ML Risk Prediction

Predicts **High Risk Malnutrition** using:

- Calories  
- Protein  
- Fats  
- Carbohydrates  
- BMI  
- Age  

Models Used:

- Logistic Regression  
- Random Forest Classifier  

Also shows:

- Accuracy Score  
- Feature Importance (Risk Factors)  

---

## 🧠 Machine Learning Workflow

1. Data Preprocessing  
2. Feature Engineering (BMI, Age Groups)  
3. Nutrition Status Labeling  
4. Risk Classification (Poor = 1, Others = 0)  
5. Train/Test Split  
6. Model Training  
7. Prediction + Feature Importance  

---

## 📂 Dataset Used

File required: nutritionverse_dish_metadata3.csv


Dataset must contain nutritional columns like:

- `total_calories`  
- `total_protein`  
- `total_fats`  
- `total_carbohydrates`  

---

## ⚙️ Tech Stack

| Category        | Tools Used |
|---------------|------------|
| Frontend UI    | Streamlit |
| Data Handling  | Pandas, NumPy |
| Visualization  | Plotly Express |
| ML Models      | Scikit-learn |
| Deployment     | Streamlit Cloud / Local |

---

## 🛠️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Mayank10021/malnutrition-dashboard.git
cd malnutrition-dashboard
```

### 2️⃣ Install Requirements

```bash 
pip install -r requirements.txt
```

### 3️⃣ Run Streamlit App

```bash
python -m streamlit run app.py
```

### 📌 Requirements File (requirements.txt)

```bash
streamlit
pandas
numpy
plotly
scikit-learn
```

### 🔮 Future Improvements

- Real government nutrition survey integration
- Live rural health monitoring dashboard
- Deep Learning based prediction
- Multi-language support
- Mobile-friendly UI

### 🏆 Hackathon Use Case

This project was developed for:

SIC Hackathon | Public Health & Rural Nutrition Analytics

Goal: To build an intelligent system that can detect malnutrition risks early and recommend nutritious dietary solutions.

### 👨‍💻 Author

Developed by Rohit Negi
Hackathon Project – 2026

### ⭐ Support

If you like this project, don’t forget to:

⭐ Star the Repo
🍴 Fork it
📢 Share it
