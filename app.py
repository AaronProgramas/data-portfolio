import streamlit as st
import os
import nbconvert
import nbformat

# Set Page Configuration
st.set_page_config(page_title="Aaron Albrecht", layout="wide")

# Define ML project names
ml_projects = {
    "Student mental health binary prediction ML.ipynb",
    "Predicting podcast listening time with ML.ipynb",
    "Kaggle top 38% base ML model.ipynb"
}

# Tabs
tab1, tab2 = st.tabs(["Projects", "Resume"])

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

# --- RESUME TAB ---
with tab2:
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
