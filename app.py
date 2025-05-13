import streamlit as st
import os
import nbconvert
import nbformat

# Set Page Configuration
st.set_page_config(page_title="Aaron Albrecht", layout="wide")

# Tabs
tab1, tab2 = st.tabs(["📊 Projects", "📄 Resume"])

# --- PROJECTS TAB ---
with tab1:
    st.title("📊 Aaron Albrecht's insane Data Analytic skills showcase")

    # List available projects
    project_files = sorted([f for f in os.listdir('projects') if f.endswith('.ipynb')])

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
    st.title("📄 About Me")

    # Show your picture
    st.image("assets/my_picture.jpg", width=250)

    st.write("""
    Hi, I'm Aaron! 👋

    I specialize in Data Analytics, with proficiency in Python, Excel, SQL, and Power BI.
    This portfolio showcases some of my recent projects.
    """)

    # RESUME DETAILS
    st.subheader("Experience")
    st.markdown("""
    **Database and Automation Developer – Osprey Visa Consulting**  
    *Rio de Janeiro, RJ | April 2025 – Present*  
    - Developed a centralized client management database (Azure SQL).  
    - Automated repetitive workflows with Python, enhancing data accuracy and reducing manual labor.

    **Commercial Analyst – SANDECH Consultoria em Engenharia e Gestão LTDA**  
    *Rio de Janeiro, RJ | Dec 2023 – May 2024*  
    - Generated over R$ 1M in sales for engineering services.  
    - Managed KPIs and created interactive dashboards using Power BI.

    **Sales Consultant – Conpleq Consultoria**  
    *Rio de Janeiro, RJ | Jul 2023 – Apr 2024*  
    - Built a Linear Regression model with 83% accuracy to predict client purchase behavior.  
    - Boosted lead response rates by 20% through targeted insights.
    """)

    st.subheader("Projects")
    st.markdown("""
    **Kaggle Machine Learning Competition** *(May 2025)*  
    - Built a stacked regression model using XGBoost, LightGBM, Gradient Boosting, and RandomForest.  
    - Ranked in the top 40% of the leaderboard for calorie expenditure prediction.

    **PDF/DOCX Automation Tool** *(May 2025)*  
    - Developed a script to mass-edit PDF/DOCX files using Fitz, PyPDF2, and python-docx.  
    - Reduced manual editing time by ~30 minutes per document.

    **Client Database System** *(April 2025)*  
    - Designed and deployed a normalized database in Azure SQL with 6 relational tables.  
    - Improved data management and client tracking.
    """)

    st.subheader("Education")
    st.markdown("""
    - **Data Science** – Universidade Estácio de Sá *(Expected 2027)*  
    - **Chemical Engineering** – Universidade do Estado do Rio de Janeiro (UERJ) *(Expected Dec 2027)*  
    """)

    st.subheader("Technical Skills")
    st.markdown("""
    - **Languages**: Python, SQL  
    - **Libraries**: Pandas, Numpy, Seaborn, Matplotlib, Scikit-learn, XGBoost  
    - **Tools**: Power BI, Jupyter Notebook, Git  
    - **Cloud Platforms**: Azure, AWS  
    - **Database**: Microsoft SQL Server  
    - **Techniques**: Data Cleaning, Predictive Modeling, Feature Engineering, KPI Analysis, Machine Learning
    """)

    st.subheader("Languages")
    st.markdown("""
    - Portuguese (Native)  
    - English (Fluent)
    """)

    st.subheader("Portfolio Links")
    st.markdown("""
    - [LinkedIn](https://www.linkedin.com/in/aaron-albrecht-32692b259/)  
    - [GitHub](https://github.com/AaronProgramas)  
    - [Kaggle](https://www.kaggle.com/aaronalbrecht)  
    """)
