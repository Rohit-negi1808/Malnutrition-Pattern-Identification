import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Malnutrition Pattern Identification",
    page_icon="🍽️",
    layout="wide"
)

# ================= GLOBAL CSS =================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: #020617;
    color: white;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
</style>
""", unsafe_allow_html=True)

# ================= TITLE =================
st.markdown("<h1 style='text-align:center;'>🍽️ Malnutrition Pattern Identification Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#94a3b8;'>Public Health • Rural Nutrition • Data Analytics</p>", unsafe_allow_html=True)

# ================= LOAD DATA =================
data = pd.read_csv("nutritionverse_dish_metadata3.csv")

# ================= REGION & DISTRICT =================
regions = ["North","South","East","West","Central"]
districts = [
    "Lucknow","Kanpur","Varanasi","Agra","Prayagraj",
    "Patna","Gaya","Bhopal","Indore","Jaipur",
    "Ahmedabad","Surat","Pune","Nagpur",
    "Kolkata","Howrah","Chennai","Madurai",
    "Bengaluru","Mysuru","Hyderabad","Warangal"
]

np.random.seed(42)
data["Region"] = np.random.choice(regions,len(data))
data["District"] = np.random.choice(districts,len(data))

# ================= AGE / BMI =================
data["Age"] = np.random.randint(5,70,len(data))
data["Height_m"] = np.random.uniform(1.45,1.8,len(data))
data["Weight_kg"] = np.random.uniform(40,95,len(data))
data["BMI"] = data["Weight_kg"] / (data["Height_m"] ** 2)

def age_group(a):
    if a <= 12: return "Child"
    elif a <= 19: return "Teen"
    elif a <= 40: return "Adult"
    elif a <= 60: return "Middle Age"
    else: return "Senior"

data["Age_Group"] = data["Age"].apply(age_group)

def bmi_cat(b):
    if b < 18.5: return "Underweight"
    elif b < 25: return "Normal"
    elif b < 30: return "Overweight"
    else: return "Obese"

data["BMI_Category"] = data["BMI"].apply(bmi_cat)

# ================= CALORIES / PROTEIN =================
def cal_level(c):
    if c < 300: return "Low"
    elif c < 600: return "Medium"
    else: return "High"

data["Calories_Level"] = data["total_calories"].apply(cal_level)

protein_need = {
    "Child":20,
    "Teen":35,
    "Adult":50,
    "Middle Age":55,
    "Senior":60
}

data["Protein_Required"] = data["Age_Group"].map(protein_need)
data["Protein_Deficient"] = data["total_protein"] < data["Protein_Required"]

def nutrition_status(row):
    if row["Protein_Deficient"] and row["Calories_Level"] == "Low":
        return "Poor"
    elif row["Calories_Level"] == "Medium":
        return "Moderate"
    else:
        return "Good"

data["Nutrition_Status"] = data.apply(nutrition_status, axis=1)

# ================= FILTER =================
st.markdown("### 🔍 Filters")
enable_filter = st.toggle("Enable Filters")

filtered = data.copy()
if enable_filter:
    cal_range = st.slider(
        "Calories Range",
        int(data["total_calories"].min()),
        int(data["total_calories"].max()),
        (200,800)
    )

    protein_min = st.slider(
        "Minimum Protein",
        0,
        int(data["total_protein"].max()),
        20
    )

    filtered = data[
        (data["total_calories"].between(cal_range[0],cal_range[1])) &
        (data["total_protein"] >= protein_min)
    ]

# ================= KPIs =================
c1,c2,c3,c4 = st.columns(4)
c1.metric("Total Records",len(filtered))
c2.metric("Avg Calories", int(filtered["total_calories"].mean()) if not filtered.empty else 0)
c3.metric("Avg Protein", round(filtered["total_protein"].mean(),1) if not filtered.empty else 0)
c4.metric("Protein Deficient", filtered["Protein_Deficient"].sum())

# ================= TABS =================
tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "📋 Dataset",
    "📊 Analytics",
    "🗺️ District Heatmap",
    "🍽️ Recommendation",
    "🤖 ML Analysis"
])

# ================= TAB 1 =================
with tab1:
    st.dataframe(filtered, use_container_width=True)

# ================= TAB 2 (15+ GRAPHS) =================
with tab2:
    graph = st.selectbox("Select Graph",[
        "Age Group Distribution",
        "BMI Category Distribution",
        "Calories Level Distribution",
        "Nutrition Status Distribution",
        "Calories vs Age",
        "Protein vs Age",
        "BMI vs Calories",
        "Calories Histogram",
        "Protein Histogram",
        "BMI Box Plot",
        "Calories by Region",
        "Protein by Region",
        "Age vs BMI",
        "Calories vs Protein",
        "Region vs Nutrition Status",
        "District vs Calories",
        "District vs Protein"
    ])

    if graph == "Age Group Distribution":
        fig = px.bar(filtered,x="Age_Group",color="Age_Group")
    elif graph == "BMI Category Distribution":
        fig = px.bar(filtered,x="BMI_Category",color="BMI_Category")
    elif graph == "Calories Level Distribution":
        fig = px.pie(filtered,names="Calories_Level")
    elif graph == "Nutrition Status Distribution":
        fig = px.bar(filtered,x="Nutrition_Status",color="Nutrition_Status")
    elif graph == "Calories vs Age":
        fig = px.scatter(filtered,x="Age",y="total_calories",color="Calories_Level")
    elif graph == "Protein vs Age":
        fig = px.scatter(filtered,x="Age",y="total_protein",color="Age_Group")
    elif graph == "BMI vs Calories":
        fig = px.scatter(filtered,x="BMI",y="total_calories",color="BMI_Category")
    elif graph == "Calories Histogram":
        fig = px.histogram(filtered,x="total_calories")
    elif graph == "Protein Histogram":
        fig = px.histogram(filtered,x="total_protein")
    elif graph == "BMI Box Plot":
        fig = px.box(filtered,y="BMI")
    elif graph == "Calories by Region":
        fig = px.bar(filtered,x="Region",y="total_calories",color="Region")
    elif graph == "Protein by Region":
        fig = px.bar(filtered,x="Region",y="total_protein",color="Region")
    elif graph == "Age vs BMI":
        fig = px.scatter(filtered,x="Age",y="BMI",color="BMI_Category")
    elif graph == "Calories vs Protein":
        fig = px.scatter(filtered,x="total_calories",y="total_protein",color="Calories_Level")
    elif graph == "District vs Calories":
        fig = px.bar(filtered,x="District",y="total_calories",color="Region")
    else:
        fig = px.bar(filtered,x="District",y="total_protein",color="Region")

    st.plotly_chart(fig,use_container_width=True)

# ================= TAB 3 (HEATMAP) =================
with tab3:
    st.subheader("🗺️ District-wise Nutrition Distribution")

    dist = filtered.groupby("District").agg({
        "total_calories":"mean",
        "total_protein":"mean"
    }).reset_index()

    fig1 = px.bar(
        dist,
        x="District",
        y="total_calories",
        color="District",
        title="Average Calories by District"
    )
    st.plotly_chart(fig1,use_container_width=True)

    fig2 = px.density_heatmap(
        filtered,
        x="District",
        y="Nutrition_Status",
        color_continuous_scale="Turbo",
        title="Nutrition Status Heatmap"
    )
    st.plotly_chart(fig2,use_container_width=True)

# ================= TAB 4 (RECOMMENDATION) =================
with tab4:
    st.subheader("🍽️ Dish Recommendation")

    age_sel = st.selectbox("Age Group",data["Age_Group"].unique())
    cal_sel = st.selectbox("Calories Level",["Low","Medium","High"])

    req = protein_need[age_sel]

    rec = data[
        (data["Age_Group"]==age_sel) &
        (data["Calories_Level"]==cal_sel) &
        (data["total_protein"]>=req)
    ].sort_values("total_protein",ascending=False).head(5)

    if rec.empty:
        st.warning("No recommendation found")
    else:
        st.dataframe(
            rec[["dish_id","total_calories","total_protein","Region"]],
            use_container_width=True
        )

# ================= TAB 5 (ML) =================
with tab5:
    st.subheader("🤖 Malnutrition Risk Prediction")

    data["Risk"] = data["Nutrition_Status"].apply(lambda x:1 if x=="Poor" else 0)

    features = [
        "total_calories","total_protein",
        "total_fats","total_carbohydrates",
        "BMI","Age"
    ]

    X = data[features]
    y = data["Risk"]

    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=0.25,random_state=42
    )

    # Logistic Regression
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train,y_train)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200,random_state=42)
    rf.fit(X_train,y_train)

    acc1 = accuracy_score(y_test,log_model.predict(X_test))
    acc2 = accuracy_score(y_test,rf.predict(X_test))

    st.success(f"Logistic Accuracy: {round(acc1*100,2)} %")
    st.success(f"Random Forest Accuracy: {round(acc2*100,2)} %")

    # Feature Importance
    imp = pd.DataFrame({
        "Feature":features,
        "Importance":rf.feature_importances_
    }).sort_values("Importance",ascending=False)

    fig_imp = px.bar(
        imp,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance (Risk Factors)"
    )
    st.plotly_chart(fig_imp,use_container_width=True)

    # USER PREDICTION
    st.markdown("### 🔮 Predict Risk for New Person")

    uc = st.number_input("Calories",200,1000,400)
    up = st.number_input("Protein",5,100,30)
    uf = st.number_input("Fats",5,100,20)
    carb = st.number_input("Carbohydrates",50,400,200)
    bmi = st.number_input("BMI",12.0,40.0,22.0)
    age = st.number_input("Age",5,80,25)

    if st.button("Predict Risk"):
        pred = rf.predict([[uc,up,uf,carb,bmi,age]])[0]
        if pred==1:
            st.error("⚠️ High Risk of Malnutrition")
        else:
            st.success("✅ Low Risk")

# ================= FOOTER =================
st.markdown("---")
st.markdown("<center><b>SIC HACKATHON PROJECT | Public Health & Nutrition (Rural Focus)</b></center>",unsafe_allow_html=True)
