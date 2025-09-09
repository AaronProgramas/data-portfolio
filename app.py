import streamlit as st
import os
import nbconvert
import nbformat
import joblib
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import io
from nbconvert import HTMLExporter
import nbformat, os, streamlit as st

# Set Page Configuration
st.set_page_config(page_title="Aaron Albrecht", layout="wide")

# Define ML project names
ml_projects = {
    "Student mental health binary prediction ML.ipynb",
    "Predicting podcast listening time with ML.ipynb",
    "Kaggle top 38% base ML model.ipynb"
}

# =======================
# Load & normalize schema
# =======================
sns.set_style("darkgrid")

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Renomeia colunas para um padrão sem espaços/maiúsculas
    col_map = {
        "Date": "DATE",
        "Time": "TIME",
        "Booking ID": "BOOKING_ID",
        "Booking Status": "BOOKING_STATUS",
        "Customer ID": "CUSTOMER_ID",
        "Vehicle Type": "VEHICLE_TYPE",
        "Pickup Location": "PICKUP_LOCATION",
        "Drop Location": "DROP_LOCATION",
        "Avg VTAT": "AVG_VTAT",
        "Avg CTAT": "AVG_CTAT",
        "Cancelled Rides by Customer": "CANCELLED_RIDES_BY_CUSTOMER",
        "Reason for cancelling by Customer": "REASON_FOR_CANCELLING_BY_CUSTOMER",
        "Cancelled Rides by Driver": "CANCELLED_RIDES_BY_DRIVER",
        "Driver Cancellation Reason": "DRIVER_CANCELLATION_REASON",
        "Incomplete Rides": "INCOMPLETE_RIDES",
        "Incomplete Rides Reason": "INCOMPLETE_RIDES_REASON",
        "Booking Value": "BOOKING_VALUE",
        "Ride Distance": "RIDE_DISTANCE",
        "Driver Ratings": "DRIVER_RATINGS",
        "Customer Rating": "CUSTOMER_RATING",
        "Payment Method": "PAYMENT_METHOD",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Tipagem
    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

    if "TIME" in df.columns:
        dt_time = pd.to_datetime(df["TIME"], format="%H:%M:%S", errors="coerce")
        df["HOUR"] = dt_time.dt.hour.astype("Int64")  # preserva NaN

    # Converte possíveis numéricos
    numeric_candidates = [
        "AVG_VTAT", "AVG_CTAT",
        "CANCELLED_RIDES_BY_CUSTOMER", "CANCELLED_RIDES_BY_DRIVER", "INCOMPLETE_RIDES",
        "BOOKING_VALUE", "RIDE_DISTANCE", "DRIVER_RATINGS", "CUSTOMER_RATING"
    ]
    for c in numeric_candidates:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # PRICE_PER_KM seguro (sem divisão por zero)
    if {"BOOKING_VALUE", "RIDE_DISTANCE"}.issubset(df.columns):
        denom = df["RIDE_DISTANCE"].replace({0: np.nan})
        df["PRICE_PER_KM"] = df["BOOKING_VALUE"] / denom
        df["PRICE_PER_KM"].replace([np.inf, -np.inf], np.nan, inplace=True)

    # Mês (para agregações)
    if "DATE" in df.columns:
        df["MONTH"] = df["DATE"].dt.to_period("M")

    return df


# ===========
# Load data
# ===========
df = load_data("ncr_ride_bookings.csv")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df_schema = pd.read_csv('uber_data_schema.csv')


st.sidebar.image("my_picture.jpg", width=150) 
st.sidebar.markdown(
"""
**Hey! I'm Aaron.**  
I specialize in Data Analytics, with proficiency in Python, Excel, SQL, and Power BI. This portfolio showcases some of my recent projects.
"""
)


# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Notebooks", "EDA", "Data Visualization", "Binary Mental Health Prediction", "Resume"])

# --- PROJECTS TAB ---
with tab1:
    st.title("Aaron Albrecht - Notebooks")

    category = st.sidebar.radio("**Select Project Category**", ["Data Analytics", "Machine Learning"])
    all_projects = sorted([f for f in os.listdir('projects') if f.endswith('.ipynb')])
    project_files = sorted([f for f in all_projects if (f in ml_projects) == (category == "Machine Learning")])

    selected_project = st.sidebar.selectbox("Choose a Project", project_files)

    if selected_project:
        st.header(f"Project: {selected_project}")

        notebook_path = os.path.join('projects', selected_project)
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        # nbconvert com template “lab” e tema escuro
        html_exporter = HTMLExporter(template_name="lab")
        html_exporter.theme = "dark"  # <- chave para fundo escuro

        body, resources = html_exporter.from_notebook_node(nb)

        # CSS extra para garantir fundo e texto
        dark_css = """
        <style>
        :root { color-scheme: dark; }

        body, .jp-Notebook, .cell, .input_area, .output_area, pre, code {
            background: #111 !important;
            color: #e6e6e6 !important;
            border-color: #333 !important;
        }
        .dataframe, table { background:#181818 !important; color:#ddd !important; }
        h1,h2,h3,h4,h5 { color:#fff !important; }

        /* strings já são vermelhas (pygments) → pegamos a cor delas */
        .s1, .s2 { color: #e06c75 !important; }

        /* keywords (import, from) ficam iguais às strings */
        .kn, .k { color: #e06c75 !important; font-weight: normal !important; }
        </style>
        """

        st.components.v1.html(dark_css + body, height=1200, scrolling=True)

# --- DEPRESSION PROB CALCULATOR TAB ---

with tab4:
    # Load dumped models & preprocessors
    
    model = joblib.load("final_depression_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoders = joblib.load("label_encoders.pkl")  # dicionário com encoders por coluna
    num_cols = joblib.load("num_cols.pkl")             # lista de colunas numéricas
    category_mapping = joblib.load("category_mapping.pkl")  # se precisar de mapping manual
    
    
    st.title("Depression Probability Calculator")
    st.write("Fill the form to predict the odds of having the big sad.")
    
    col1, col2, col3 = st.columns([1, 2, 2])
    
    # Input form

    with col1:
        st.title("About the Mental Health App")
        st.write(
            "This web app leverages a machine learning model trained on an open-source Kaggle dataset "
            "to estimate the probability of depression. ")
        st.write(
            "While the model achieves an accuracy of 87%, it is **not** a substitute for professional "
            "mental health support. Please consult a qualified professional for any medical concerns."
        )
    
        
    with col2:
    
        st.subheader("Personal Information")
    
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        profession = st.selectbox("Profession", ['Student', 'Working Professional'])
        age = st.slider("Age", 10, 60, 20)
    
        st.subheader("Mental Health Factors")
    
        suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts ?", ["Yes", "No"])
        family_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])
    
    with col3:
    
        st.subheader("Academic Information")
    
        degree = st.selectbox("Degree", ['B.Pharm', 'BSc', 'BA', 'BCA', 'M.Tech', 'MSc', 'MD', 'Class 12', 'Other'])
        cgpa = st.slider("CGPA", 0, 10, 5)
        academic_pressure = st.slider("Academic Pressure (1–5)", 1, 5, 3)
    
        st.subheader("Lifestyle Factors")
    
        sleep_duration = st.selectbox("Sleep Duration", ['Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours'])
        dietary_habits = st.selectbox("Dietary Habits", ['Healthy', 'Moderate', 'Unhealthy'])
    

    
    
    # Organize data to a DataFrame
    
    input_dict = {
        'Gender': gender,
        'Age': float(age),
        'Profession': profession,
        'Academic Pressure': float(academic_pressure),
        'Work Pressure': 0.0,
        'CGPA': float(cgpa),
        'Study Satisfaction': 3.0,
        'Job Satisfaction': 0.0,
        'Sleep Duration': sleep_duration,
        'Dietary Habits': dietary_habits,
        'Degree': degree,
        'Have you ever had suicidal thoughts ?': suicidal_thoughts,
        'Work/Study Hours': 5.0,
        'Financial Stress': 1.0,
        'Family History of Mental Illness': family_history
    }
    
    input_df = pd.DataFrame([input_dict])
    
    # Pre processing
    
    # Apply LabelEncoder to categorical cols
    
    for col, le in label_encoders.items():
        if col in input_df.columns:
            input_df[col] = le.transform(input_df[col])
    
    # Scaler to numerical cols
    
    input_df[num_cols] = scaler.transform(input_df[num_cols])
    
    # Prediction
    
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    
    st.write(f"**Predicted Class:** {'Depressed' if prediction == 1 else 'Not Depressed'}")
    
    # Convert probability to % and set bar palette
    
    prob_percent = int(prob * 100)
    
    cmap = cm.get_cmap("turbo")  
    color = cmap(prob) 
    hex_color = mcolors.rgb2hex(color)  # converts hex to string
    
    # Custom bar
    
    st.markdown(f"""
    <div style="border: 1px solid #ccc; border-radius: 10px; width: 100%; background-color: #f5f5f5; position: relative; height: 30px;">
      <div style="background-color: {hex_color}; width: {prob_percent}%; height: 100%; border-radius: 10px; text-align: center; color: black; font-weight: bold;">
        {prob_percent}%
      </div>
    </div>
    """, unsafe_allow_html=True)

# --- RESUME TAB ---
with tab5:
    st.title("About Me")
    col1, col2 = st.columns(2)
    with col1:
        # RESUME DETAILS
        st.subheader("Technical Skills")
        st.markdown("""
        - **Languages**: Python, SQL,
        - **Libraries**: Pandas, Numpy, Seaborn, Matplotlib, Scikit-learn, XGBoost  
        - **Databases**: Microsoft SQL Server, AWS  
        - **Tools**: Power BI, Git, Jupyter Notebook, Azure, AWS Athena, Airflow, Snowflake, Docker  
        - **Others**: Data Cleaning, Predictive Modeling, KPI Analysis, Machine Learning, Feature Engineering, Data Visualization, KPI Analysis
        """)
    
        st.subheader("Experience")
        st.markdown("""
        **Marketing Data Analyst – Coca Cola**  
        *Rio de Janeiro, RJ | Jul 2025 – Present*  
        - Developed SQL queries to extract and validade marketing datasets, increasing KPI accuracy and reliability by 5%.  
        - Automated the update pipeline for over 80 power bi dashboards, ensuring real-time data availability and reducing manual labor.
    
        **Database and Automation Developer – Osprey Visa Consulting**  
        *Rio de Janeiro, RJ | Mar 2025 – Aug 2025*  
        - Developed a centralized client management database (Azure SQL)  
        - Automated repetitive workflows with Python, enhancing data accuracy and reducing manual labor
    
        **Commercial Analyst – SANDECH Consultoria em Engenharia e Gestão LTDA**  
        *Rio de Janeiro, RJ | Dec 2023 – May 2024*  
        - Generated over R$ 1M in sales for engineering services  
        - Created interactive dashboards using Power BI
    
        **Sales Consultant – Conpleq Consultoria**  
        *Rio de Janeiro, RJ | Jul 2023 – Apr 2024*  
        - Built a regression model with 83% accuracy to predict client purchase behavior  
        - Boosted response rates by 20% through data-driven insights
        """)
    with col2:
        st.subheader("Projects")
        st.markdown("""
        **Kaggle Machine Learning Competition** *(May 2025)*  
        - Built a stacked regression model with XGBoost, LightGBM, and RandomForest  
        - Ranked in the top 38% for calorie expenditure prediction

        **Machine Learning-Based Depression Probability Calculator** *(Aug 2025)*  
        - Built a Logistic Regression model with 87% accuracy to predict mental health status, using an open source student dataset.  
        - Built an interactive app using Streamlit to deploy the model.
    
        **PDF/DOCX Automation Tool** *(May 2025)*  
        - Script to batch-edit PDF/DOCX using Fitz, PyPDF2, and python-docx  
        - Reduced editing time by ~30 min per document
        """)
    
        st.subheader("Education")
        st.markdown("""
        - **B.Sc. in Data Science** – Estácio de Sá *(Expected 2027)*  
        - **B.Sc. in Chemical Engineering** – UERJ *(Expected Dec 2027)*
        """)
    
        st.subheader("Languages")
        st.markdown("""
        - Portuguese (Native)  
        - English (Fluent)
        """)

with tab2:
    st.title("Uber Rides Analytics Hub")
    date_min = pd.to_datetime(df.get("DATE"), errors="coerce").min()
    date_max = pd.to_datetime(df.get("DATE"), errors="coerce").max()
    n_rows = len(df)
    n_cities = df["PICKUP_LOCATION"].nunique() if "PICKUP_LOCATION" in df.columns else None
    n_vehicle = df["VEHICLE_TYPE"].nunique() if "VEHICLE_TYPE" in df.columns else None
    
    facts = [
        f"{n_rows:,} rides",
        f"{n_vehicle} vehicle types" if n_vehicle else None,
        f"{n_cities} pickup locations" if n_cities else None,
        f"{date_min:%b %Y} — {date_max:%b %Y}" if pd.notna(date_min) and pd.notna(date_max) else None,
    ]
    facts = " · ".join([f for f in facts if f])
    
    st.caption(f"Interactive Panel with a **public Kaggle dataset**. {facts}")
    # KPIs
    n_rows, n_cols = df.shape
    total_missing = int(df.isna().sum().sum())
    dup_rows = int(df.duplicated().sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{n_rows:,}")
    c2.metric("Columns", f"{n_cols}")
    c3.metric("Total Missing", f"{total_missing:,}")
    c4.metric("Duplicate Rows", f"{dup_rows:,}")

    # Head
    st.subheader("Head of the dataframe")
    st.dataframe(df.head())

    # =======================
    # INFO  |  HEATMAP (lado a lado, compacto)
    # =======================
    st.subheader("EDA")
    col_schema, col_info, col_heat = st.columns([10,10,13.7])

    with col_schema:
        st.markdown("**DataFrame Schema**")
        st.dataframe(df_schema, height=735)

    with col_info:
        st.markdown("**DataFrame.info()**")
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.code(buffer.getvalue())

    with col_heat:
        st.markdown("**Correlation Heatmap (compact)**")

        # base numérica sem constantes
        num_base = df.select_dtypes(include=[np.number])
        num_base = num_base.loc[:, num_base.std(numeric_only=True) > 0]

        # correlação completa
        corr = num_base.corr()

        # heatmap estático
        fig, ax = plt.subplots(figsize=(16, 8))
        sns.heatmap(
            corr,
            annot=True,
            cmap="viridis",
            center=0,
            square=True,
            cbar=True,
            ax=ax
        )
        st.pyplot(fig)



    # =======================
    # Describe (ABAIXO das colunas)
    # =======================
    st.subheader("Describe (all)")
    st.dataframe(df.describe(include="all").transpose())

    # =======================
    # Extra EDA (duas colunas, abaixo)
    # =======================
    st.subheader("Extra EDA (side-by-side)")
    colA, colB = st.columns(2)

    with colA:
        # Missing values %
        st.markdown("**Missing Values (%)**")
        miss = (df.isna().mean() * 100).sort_values(ascending=False).rename("missing_%").to_frame()
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(x=miss.index[:15], y=miss["missing_%"][:15], palette="viridis", ax=ax)
        ax.set_ylabel("Missing (%)"); ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=90)
        st.pyplot(fig)

        # Correlação com alvo (top 10)
        num_base2 = df.select_dtypes(include=[np.number])
        num_base2 = num_base2.loc[:, num_base2.std(numeric_only=True) > 0]
        if not num_base2.empty:
            tgt = "BOOKING_VALUE" if "BOOKING_VALUE" in num_base2.columns else num_base2.columns[0]
            st.markdown(f"**Correlation with `{tgt}` (top 10)**")
            corr_top = num_base2.corr()[tgt].drop(tgt).sort_values(
                key=lambda s: s.abs(), ascending=False
            ).head(10)
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x=corr_top.index, y=corr_top.values, palette="viridis", ax=ax)
            ax.set_ylabel("Pearson r"); ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=90)
            st.pyplot(fig)

    with colB:
        # Cardinalidade das categóricas
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        if cat_cols:
            st.markdown("**Categorical Cardinality (top 15)**")
            card = df[cat_cols].nunique().sort_values(ascending=False).rename("unique_values").to_frame()
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.barplot(x=card.index[:15], y=card["unique_values"][:15], palette="viridis", ax=ax)
            ax.set_ylabel("Unique values"); ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=90)
            st.pyplot(fig)

        # Taxa de outliers (IQR)
        st.markdown("**Outlier Rate by Numeric Column (IQR)**")
        def outlier_rate(s: pd.Series) -> float:
            s = pd.to_numeric(s, errors="coerce").dropna()
            if s.empty:
                return 0.0
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr == 0 or pd.isna(iqr):
                return 0.0
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            return ((s < lower) | (s > upper)).mean()

        num_base3 = df.select_dtypes(include=[np.number])
        num_base3 = num_base3.loc[:, num_base3.std(numeric_only=True) > 0]
        out_rates = num_base3.apply(outlier_rate).sort_values(ascending=False).rename("outlier_rate").to_frame()
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=out_rates.index[:12], y=out_rates["outlier_rate"][:12], palette="viridis", ax=ax)
        ax.set_ylabel("Outlier rate"); ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=90)
        st.pyplot(fig)

    # Distribuições rápidas (segunda fileira)
    dist_cols = [c for c in ["DRIVER_RATINGS", "CUSTOMER_RATING"] if c in df.columns]
    if dist_cols:
        st.subheader("Ratings Distribution")
        d1, d2 = st.columns(2)
        if len(dist_cols) > 0:
            with d1:
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.histplot(df[dist_cols[0]].dropna(), bins=30, kde=True,
                             color=sns.color_palette("viridis", as_cmap=True)(0.6), ax=ax)
                ax.set_title(dist_cols[0]); st.pyplot(fig)
        if len(dist_cols) > 1:
            with d2:
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.histplot(df[dist_cols[1]].dropna(), bins=30, kde=True,
                             color=sns.color_palette("viridis", as_cmap=True)(0.6), ax=ax)
                ax.set_title(dist_cols[1]); st.pyplot(fig)

with tab3:
    st.title("Visualization")

    # Garantias (caso o CSV mude)
    if "DATE" in df.columns and "MONTH" not in df.columns:
        df["MONTH"] = pd.to_datetime(df["DATE"], errors="coerce").dt.to_period("M")
    if "TIME" in df.columns and "HOUR" not in df.columns:
        df["HOUR"] = pd.to_datetime(df["TIME"], format="%H:%M:%S", errors="coerce").dt.hour
    if {"BOOKING_VALUE", "RIDE_DISTANCE"}.issubset(df.columns) and "PRICE_PER_KM" not in df.columns:
        df["PRICE_PER_KM"] = df["BOOKING_VALUE"] / df["RIDE_DISTANCE"].replace({0: np.nan})

    col1, col2 = st.columns(2)

    with col1:
        # Monthly Rides by Vehicle Type
        if {"MONTH", "VEHICLE_TYPE", "BOOKING_ID"}.issubset(df.columns):
            st.subheader("Monthly Rides by Vehicle Type")
            rides_by_vehicle = (
                df.groupby(["MONTH", "VEHICLE_TYPE"])["BOOKING_ID"]
                  .count()
                  .reset_index()
            )
            rides_by_vehicle["MONTH"] = rides_by_vehicle["MONTH"].astype(str)
            fig, ax = plt.subplots(figsize=(14, 7))
            sns.lineplot(
                data=rides_by_vehicle,
                x="MONTH", y="BOOKING_ID", hue="VEHICLE_TYPE",
                palette="viridis", marker="o", linewidth=2.5, ax=ax
            )
            ax.set_xlabel("Month"); ax.set_ylabel("Number of Rides")
            ax.tick_params(axis="x", rotation=90)
            ax.legend(title="Vehicle Type", bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.grid(alpha=0.3)
            st.pyplot(fig)

        # Scatter: Booking Value vs Ride Distance
        if {"RIDE_DISTANCE", "BOOKING_VALUE", "VEHICLE_TYPE"}.issubset(df.columns):
            st.subheader("Booking Value vs Ride Distance")
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.scatterplot(
                data=df, x="RIDE_DISTANCE", y="BOOKING_VALUE",
                hue="VEHICLE_TYPE", palette="viridis", alpha=0.6, ax=ax
            )
            ax.set_title("Booking Value vs Ride Distance", fontsize=16, weight="bold")
            ax.set_xlabel("Ride Distance (km)")
            ax.set_ylabel("Booking Value ($)")
            ax.legend(title="Vehicle Type", bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.grid(alpha=0.3)
            st.pyplot(fig)

        # Cancellation Breakdown
        if {"CANCELLED_RIDES_BY_CUSTOMER", "CANCELLED_RIDES_BY_DRIVER"}.issubset(df.columns):
            st.subheader("Cancellation Breakdown")
            cancel_data = {
                "Cancelled by Customer": float(df["CANCELLED_RIDES_BY_CUSTOMER"].sum()),
                "Cancelled by Driver": float(df["CANCELLED_RIDES_BY_DRIVER"].sum())
            }
            fig, ax = plt.subplots(figsize=(6, 6))
            colors = sns.color_palette("viridis", n_colors=2)
            ax.pie(cancel_data.values(), labels=cancel_data.keys(),
                   autopct="%1.1f%%", startangle=90, colors=colors)
            ax.set_title("Cancellation Breakdown", fontsize=16, weight="bold")
            st.pyplot(fig)

        # Ride Demand by Hour
        if "HOUR" in df.columns:
            st.subheader("Ride Demand Distribution by Hour of the Day")
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.histplot(
                data=df.dropna(subset=["HOUR"]),
                x="HOUR", bins=24, kde=True, ax=ax,
                color=sns.color_palette("viridis", as_cmap=True)(0.6)
            )
            ax.set_title("Ride Demand Distribution by Hour of the Day", fontsize=16, weight="bold")
            ax.set_xlabel("Hour of Day"); ax.set_ylabel("Number of Rides (density overlay)")
            ax.set_xticks(range(0, 24))
            ax.grid(alpha=0.3)
            st.pyplot(fig)

    with col2:
        # Average Price per Km by Hour
        if {"HOUR", "PRICE_PER_KM"}.issubset(df.columns):
            st.subheader("Average Price per Km Throughout the Day")
            avg_price_hour = (
                df.dropna(subset=["HOUR", "PRICE_PER_KM"])
                  .groupby("HOUR")["PRICE_PER_KM"]
                  .mean()
                  .reset_index()
            )
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.lineplot(
                data=avg_price_hour, x="HOUR", y="PRICE_PER_KM",
                marker="o", linewidth=2.5, ax=ax,
                color=sns.color_palette("viridis", as_cmap=True)(0.7)
            )
            ax.set_title("Average Price per Km Throughout the Day", fontsize=16, weight="bold")
            ax.set_xlabel("Hour of Day"); ax.set_ylabel("Average Price per Km ($/km)")
            ax.set_xticks(range(0, 24))
            ax.grid(alpha=0.3)
            st.pyplot(fig)

        # Distribution of Booking Value
        if "BOOKING_VALUE" in df.columns:
            st.subheader("Distribution of Booking Value")
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.histplot(
                df["BOOKING_VALUE"].dropna(), bins=50, kde=True, ax=ax,
                color=sns.color_palette("viridis", as_cmap=True)(0.7)
            )
            ax.set_title("Distribution of Booking Value", fontsize=16, weight="bold")
            ax.set_xlabel("Booking Value ($)"); ax.set_ylabel("Frequency")
            st.pyplot(fig)

        # Total Revenue per Month
        if {"MONTH", "BOOKING_VALUE"}.issubset(df.columns):
            st.subheader("Total Revenue per Month")
            revenue_month = df.groupby("MONTH")["BOOKING_VALUE"].sum().reset_index()
            revenue_month["MONTH"] = revenue_month["MONTH"].astype(str)
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.lineplot(
                data=revenue_month, x="MONTH", y="BOOKING_VALUE",
                marker="o", linewidth=2.5, ax=ax,
                color=sns.color_palette("viridis", as_cmap=True)(0.6)
            )
            ax.set_title("Total Revenue per Month", fontsize=16, weight="bold")
            ax.set_xlabel("Month"); ax.set_ylabel("Total Revenue ($)")
            ax.tick_params(axis="x", rotation=90)
            ax.grid(alpha=0.3)
            st.pyplot(fig)

        # Violin: Price per Km by Vehicle Type
        if {"PRICE_PER_KM", "VEHICLE_TYPE"}.issubset(df.columns):
            st.subheader("Distribution of Price per Km by Vehicle Type")
            cap = df["PRICE_PER_KM"].quantile(0.80)
            df_cap = df.copy()
            df_cap.loc[df_cap["PRICE_PER_KM"] > cap, "PRICE_PER_KM"] = cap
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.violinplot(
                data=df_cap, x="VEHICLE_TYPE", y="PRICE_PER_KM",
                palette="viridis", cut=0, ax=ax
            )
            ax.set_title("Distribution of Price per Km by Vehicle Type", fontsize=16, weight="bold")
            ax.set_xlabel("Vehicle Type"); ax.set_ylabel("Price per Km ($/km)")
            ax.tick_params(axis="x", rotation=90)
            st.pyplot(fig)

st.sidebar.subheader("Portfolio Links")
st.sidebar.markdown("""
[![GitHub](https://img.shields.io/badge/GitHub-AaronProgramas-black?logo=github)](https://github.com/AaronProgramas)  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-aaron--albrecht-black?logo=linkedin)](https://www.linkedin.com/in/aaron-albrecht-32692b259/)  
[![Kaggle](https://img.shields.io/badge/Kaggle-AaronAlbrecht-black?logo=kaggle)](https://www.kaggle.com/aaronalbrecht)
""", unsafe_allow_html=True)