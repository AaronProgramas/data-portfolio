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
    project_files = [f for f in os.listdir('projects') if f.endswith('.ipynb')]

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
        st.components.v1.html(body, height=800, scrolling=True)

# --- RESUME TAB ---
with tab2:
    st.title("📄 About Me")

    # Show your picture
    st.image("assets/my_picture.jpg", width=250)

    st.write("""
    Hi, I'm Aaron! 👋

    I consider myself to be the avatar of Data Analytics, the master of four elements (Python, Excel, SQL and Power BI).
    This portfolio showcases some of my recent projects.
    """)

    # RESUME DETAILS
    st.subheader("Experience")
    st.markdown("""
    **Data Analytics**  
    - **Universidade do Estado do Rio de Janeiro**  
      Scientific Research Project  
      Developed a colorimetric sensor for food safety analysis using ML models to assess biogenic amines in protein-rich foods. *(May 2023 – Present)*

    - **Sales Consultant – Conpleq Consultoria**  
      Implemented ML models to predict purchasing behavior based on demographic data. *(September 2023 – May 2024)*
    
    **Chemical Engineering**  
    - **Intern – SANDECH Consultoria (Dec 2023 – May 2024)**  
      Commercialization of engineering services (over R$ 1M in sales), KPI analysis using Power BI.
    
    - **Intern – Hugo Silva & Maldonado (Aug 2023 – Dec 2023)**  
      Translated 200+ technical patents; reviewed industrial process descriptions.
    """)

    st.subheader("Portfolio Links")
    st.markdown("""
    - [GitHub](https://github.com/AaronProgramas)
    - [Kaggle](https://www.kaggle.com/aaronalbrecht)
    """)

    st.subheader("Technical Skills")
    st.markdown("""
    - **Languages**: Python, SQL  
    - **Libraries**: Pandas, Numpy, Scikit-learn, Seaborn, Matplotlib  
    - **Tools**: Power BI, Jupyter Notebook, AutoCAD, Excel, Git  
    - **Cloud**: AWS (S3, QuickSight, Lambda)  
    - **Others**: Data cleaning, predictive modeling, feature engineering, KPI analysis, ML (classification, regression)
    """)

    st.subheader("Languages")
    st.markdown("""
    - Portuguese (native)  
    - English (fluent)
    """)

    st.subheader("Education")
    st.markdown("""
    - **Data Science** – Universidade Estácio de Sá *(2025–2027)*  
    - **Chemical Engineering** – Universidade do Estado do Rio de Janeiro (UERJ) *(2021–2027)*  
    - **Chemical Technician** – Escola Técnica Star Brasil
    """)