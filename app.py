import streamlit as st
import os
import nbconvert
import nbformat
import joblib
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Set Page Configuration
st.set_page_config(page_title="Aaron Albrecht", layout="wide")

# Define ML project names
ml_projects = {
    "Student mental health binary prediction ML.ipynb",
    "Predicting podcast listening time with ML.ipynb",
    "Kaggle top 38% base ML model.ipynb"
}

# Tabs
tab1, tab2, tab3 = st.tabs(["Projects", "Binary Mental Health Prediction", "Resume"])

# --- PROJECTS TAB ---
with tab1:
    st.title("Aaron Albrecht - Data Analytics Portfolio")

    # --- Sidebar: Category selection ---
    category = st.sidebar.radio("Select Project Category", ["Data Analytics", "Machine Learning"])

    # --- Filter projects ---
    all_projects = sorted([f for f in os.listdir('projects') if f.endswith('.ipynb')])
    if category == "Machine Learning":
        project_files = sorted([f for f in all_projects if f in ml_projects])
    else:
        project_files = sorted([f for f in all_projects if f not in ml_projects])

    # --- Project selection ---
    selected_project = st.sidebar.selectbox("Choose a Project", project_files)

    if selected_project:
        st.header(f"Project: {selected_project}")

        # Load notebook
        notebook_path = os.path.join('projects', selected_project)
        with open(notebook_path) as f:
            notebook_content = nbformat.read(f, as_version=4)

        # Convert notebook to HTML
        html_exporter = nbconvert.HTMLExporter()
        (body, resources) = html_exporter.from_notebook_node(notebook_content)

        # Display HTML
        st.components.v1.html(body, height=1200, scrolling=True)

# --- DEPRESSION PROB CALCULATOR TAB ---

with tab2:
    # Load dumped models & preprocessors
    
    model = joblib.load("final_depression_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoders = joblib.load("label_encoders.pkl")  # dicionário com encoders por coluna
    num_cols = joblib.load("num_cols.pkl")             # lista de colunas numéricas
    category_mapping = joblib.load("category_mapping.pkl")  # se precisar de mapping manual
    
    # Streamlit app
    
    st.set_page_config(page_title="Am I Depressed?", page_icon="☹️", layout="centered")
    
    st.title("Depression Probability Calculator")
    st.write("Fill the form to predict the odds of having the big sad.")
    
    with st.sidebar:
        st.title("About the App")
        st.write(
            "This web app leverages a machine learning model trained on an open-source Kaggle dataset "
            "to estimate the probability of depression. ")
        st.write(
            "While the model achieves an accuracy of 87%, it is **not** a substitute for professional "
            "mental health support. Please consult a qualified professional for any medical concerns."
        )
    
        st.title("Hire Me")
        st.markdown(
            """
            [![LinkedIn](https://img.shields.io/badge/LinkedIn-aaron--albrecht-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/aaron-albrecht-32692b259/)
            """,
            unsafe_allow_html=True
        )
    col1, col2 = st.columns([1, 1])
    
    # Input form
    
    with col1:
    
        st.subheader("Personal Information")
    
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        profession = st.selectbox("Profession", ['Student', 'Working Professional'])
        age = st.slider("Age", 10, 60, 20)
    
        st.subheader("Mental Health Factors")
    
        suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts ?", ["Yes", "No"])
        family_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])
    
    with col2:
    
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
with tab3:
    st.title("About Me")

    # Add resume download button
    with open("assets/resume.pdf", "rb") as pdf_file:
        PDFbyte = pdf_file.read()

    st.markdown("---")
    st.download_button(
        label="📄 Download Resume (PDF)",
        data=PDFbyte,
        file_name="Aaron_Albrecht_Resume.pdf",
        mime="application/pdf"
    )
    # Show your picture
    st.image("assets/my_picture.jpg", width=250)

    st.write("""
    Hi, I'm Aaron!

    I specialize in Data Analytics, with proficiency in Python, Excel, SQL, and Power BI.
    This portfolio showcases some of my recent projects.
    """)

    # RESUME DETAILS
    st.subheader("Experience")
    st.markdown("""
    **Database and Automation Developer – Osprey Visa Consulting**  
    *Rio de Janeiro, RJ | Apr 2025 – Present*  
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

    st.subheader("Projects")
    st.markdown("""
    **Kaggle Machine Learning Competition** *(May 2025)*  
    - Built a stacked regression model with XGBoost, LightGBM, and RandomForest  
    - Ranked in the top 38% for calorie expenditure prediction

    **PDF/DOCX Automation Tool** *(May 2025)*  
    - Script to batch-edit PDF/DOCX using Fitz, PyPDF2, and python-docx  
    - Reduced editing time by ~30 min per document
    """)

    st.subheader("Education")
    st.markdown("""
    - **B.Sc. in Data Science** – Estácio de Sá *(Expected 2027)*  
    - **B.Sc. in Chemical Engineering** – UERJ *(Expected Dec 2027)*
    """)

    st.subheader("Technical Skills")
    st.markdown("""
    - **Languages**: Python, SQL  
    - **Libraries**: Pandas, Numpy, Seaborn, Matplotlib, Scikit-learn, XGBoost  
    - **Databases**: Microsoft SQL Server  
    - **Tools**: Power BI, Git, Jupyter Notebook, Azure, AWS, Airflow, Snowflake, Docker  
    - **Others**: Data Cleaning, Predictive Modeling, KPI Analysis, Machune Learning, Data Visualization
    """)

    st.subheader("Languages")
    st.markdown("""
    - Portuguese (Native)  
    - English (Fluent)
    """)

    st.subheader("Portfolio Links")
    st.markdown("""
    [![GitHub](https://img.shields.io/badge/GitHub-AaronProgramas-black?logo=github)](https://github.com/AaronProgramas)  
    [![LinkedIn](https://img.shields.io/badge/LinkedIn-aaron--albrecht-black?logo=linkedin)](https://www.linkedin.com/in/aaron-albrecht-32692b259/)  
    [![Kaggle](https://img.shields.io/badge/Kaggle-AaronAlbrecht-black?logo=kaggle)](https://www.kaggle.com/aaronalbrecht)
    """, unsafe_allow_html=True)
