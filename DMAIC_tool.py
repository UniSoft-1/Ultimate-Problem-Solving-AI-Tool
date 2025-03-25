import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import spacy
from textblob import TextBlob
import uuid
import graphviz

# -------------------- HOME DASHBOARD -------------------- #
def run_home_ui():
    st.subheader("🏠 Welcome to the DMAIC Problem Solving Tool")
    st.write("This tool will guide you through the five phases of the DMAIC methodology: Define, Measure, Analyze, Improve, and Control.")
    st.markdown("""
    ### 🔍 What can you do here?
    - Identify and describe a manufacturing problem
    - Analyze your process data
    - Discover root causes
    - Improve your process with data-driven decisions
    - Control and sustain improvements over time
    """)
    st.success("Select a phase from the dropdown menu above to get started.")

# -------------------- DEFINE MODULE -------------------- #
def extract_key_phrases(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return list(set([chunk.text for chunk in doc.noun_chunks if len(chunk.text) > 2]))

def generate_project_charter(problem_description):
    blob = TextBlob(problem_description)
    sentiment = blob.sentiment.polarity
    mood = "Urgent" if sentiment < -0.2 else "Moderate" if sentiment < 0.2 else "Optimistic"

    key_phrases = extract_key_phrases(problem_description)

    charter = {
        "Project ID": str(uuid.uuid4())[:8],
        "Problem Statement": problem_description,
        "Project Mood": mood,
        "Key Focus Areas": key_phrases[:5],
        "Business Impact": "To be defined",
        "Goal Statement": f"Reduce impact of: {key_phrases[0] if key_phrases else 'identified issue'}",
        "Timeline": "90 days",
        "Team Members": "To be assigned"
    }
    return charter

def generate_sipoc(key_phrases):
    return {
        "Suppliers": ["Operators", "Vendors"],
        "Inputs": key_phrases[:2] or ["Raw materials"],
        "Process": ["Manufacturing Process"],
        "Outputs": key_phrases[2:4] or ["Finished Product"],
        "Customers": ["Internal QA", "End Users"]
    }

def download_df_as_excel(df_dict, filename="dmaic_define.xlsx"):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for sheet, df in df_dict.items():
            df.to_excel(writer, index=False, sheet_name=sheet)
    output.seek(0)
    return output

def run_define_ui():
    st.subheader("🟦 DMAIC - DEFINE Phase")
    st.write("Describe your manufacturing problem and let the AI help define it.")

    problem_input = st.text_area("🗣️ Describe the problem:", height=200)

    if st.button("🔍 Analyze and Generate"):
        if not problem_input.strip():
            st.warning("Please enter a problem description.")
        else:
            with st.spinner("Analyzing with AI..."):
                charter = generate_project_charter(problem_input)
                sipoc_data = generate_sipoc(charter["Key Focus Areas"])

                st.subheader("📄 Project Charter")
                st.json(charter)

                st.subheader("📊 SIPOC Diagram")
                sipoc_df = pd.DataFrame.from_dict(sipoc_data, orient='index').transpose()
                st.dataframe(sipoc_df)

                excel_data = download_df_as_excel({"Project Charter": pd.DataFrame([charter]), "SIPOC": sipoc_df})
                st.download_button("📥 Download Excel Report", data=excel_data, file_name="dmaic_define_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# -------------------- MEASURE MODULE -------------------- #
def run_measure_ui():
    st.subheader("🟩 DMAIC - MEASURE Module")
    st.write("Upload your process data to analyze variability and establish a performance baseline.")

    uploaded_file = st.file_uploader("📤 Upload your data file (.csv or .xlsx)", type=["csv", "xlsx"])

    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.subheader("🔎 Data Preview")
        st.dataframe(df.head())

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if numeric_cols:
            st.subheader("📈 Key Metric Analysis")
            selected_metric = st.selectbox("Select a metric to analyze:", numeric_cols)

            col1, col2 = st.columns(2)
            with col1:
                st.write("**Descriptive statistics:**")
                st.write(df[selected_metric].describe())

            with col2:
                st.write("**Histogram:**")
                fig, ax = plt.subplots()
                sns.histplot(df[selected_metric], kde=True, ax=ax)
                st.pyplot(fig)

            st.subheader("📊 Variability and Limits")
            mean = df[selected_metric].mean()
            std = df[selected_metric].std()
            ucl = mean + 3 * std
            lcl = mean - 3 * std

            st.metric("Mean", f"{mean:.2f}")
            st.metric("Standard Deviation", f"{std:.2f}")
            st.metric("Upper Control Limit (UCL)", f"{ucl:.2f}")
            st.metric("Lower Control Limit (LCL)", f"{lcl:.2f}")

            fig2, ax2 = plt.subplots()
            ax2.plot(df[selected_metric], label="Data")
            ax2.axhline(mean, color='green', linestyle='--', label="Mean")
            ax2.axhline(ucl, color='red', linestyle='--', label="UCL")
            ax2.axhline(lcl, color='red', linestyle='--', label="LCL")
            ax2.set_title("Control Chart")
            ax2.legend()
            st.pyplot(fig2)

            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name="Raw Data")
                pd.DataFrame({
                    'Metric': [selected_metric],
                    'Mean': [mean],
                    'STD': [std],
                    'UCL': [ucl],
                    'LCL': [lcl]
                }).to_excel(writer, index=False, sheet_name="Control Limits")
            output.seek(0)

            st.download_button("📥 Download Measure Report", data=output, file_name=f"dmaic_measure_{datetime.now().date()}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.warning("No numeric columns found for analysis.")

# -------------------- ANALYZE MODULE -------------------- #
def run_analyze_ui():
    st.subheader("🟨 DMAIC - ANALYZE Module")
    st.write("Describe the process symptoms and let the AI suggest potential root causes.")

    user_input = st.text_area("🧠 Describe the observed symptoms:", height=200)

    if st.button("🔍 Generate Root Cause Hypotheses"):
        if not user_input.strip():
            st.warning("Please describe the symptoms to proceed.")
        else:
            with st.spinner("Analyzing input and generating hypotheses..."):
                nlp = spacy.load("en_core_web_sm")
                doc = nlp(user_input)
                causes = [chunk.text for chunk in doc.noun_chunks if len(chunk.text) > 3]
                causes = list(set(causes))[:6]

                st.markdown("### 🔎 Suggested Root Causes:")
                for i, cause in enumerate(causes, 1):
                    st.markdown(f"**{i}.** {cause}")

                st.markdown("### 🧰 Ishikawa Diagram Categories")
                categories = {
                    "Machines": [],
                    "Manpower": [],
                    "Materials": [],
                    "Methods": [],
                    "Measurement": [],
                    "Environment": []
                }

                classified_causes = {}
                for cause in causes:
                    selected_category = st.selectbox(f"Which category does this cause belong to? → {cause}", list(categories.keys()), key=cause)
                    categories[selected_category].append(cause)
                    classified_causes[cause] = selected_category

                st.markdown("### 🖼️ Ishikawa Diagram (Graph)")
                dot = graphviz.Digraph()
                dot.node("Problem", "❗ Problem")
                for category in categories:
                    dot.node(category, category)
                    dot.edge("Problem", category)
                    for cause in categories[category]:
                        cause_id = f"{category}_{cause[:6]}"
                        dot.node(cause_id, cause)
                        dot.edge(category, cause_id)
                st.graphviz_chart(dot)

                if st.button("📤 Export Ishikawa to Excel"):
                    ishikawa_data = []
                    for category, causes_list in categories.items():
                        for cause in causes_list:
                            ishikawa_data.append({"Category": category, "Cause": cause})

                    df_ishikawa = pd.DataFrame(ishikawa_data)
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df_ishikawa.to_excel(writer, index=False, sheet_name="Ishikawa")
                    output.seek(0)

                    st.download_button(
                        label="📥 Download Ishikawa Diagram (Excel)",
                        data=output,
                        file_name="ishikawa_diagram.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

# -------------------- IMPROVE MODULE -------------------- #
def run_improve_ui():
    st.subheader("🟧 DMAIC - IMPROVE Phase")
    st.write("Use insights from the Analyze phase to suggest and prioritize improvements.")
    st.markdown("### 💡 AI-Suggested Improvements")
    root_causes = st.text_area("Paste your confirmed root causes here (one per line):", height=150)

    if root_causes:
        causes = [c.strip() for c in root_causes.split('\n') if c.strip()]
        suggestions = [f"Implement preventive measure for: {c}" for c in causes]

        st.markdown("### ✅ Proposed Solutions:")
        for i, s in enumerate(suggestions, 1):
            st.write(f"{i}. {s}")

        st.markdown("### 🔢 Prioritization (Effort vs. Impact)")
        priorities = []
        for s in suggestions:
            col1, col2 = st.columns(2)
            with col1:
                effort = st.slider(f"Effort for '{s}'", 1, 5, 3, key=s+"_eff")
            with col2:
                impact = st.slider(f"Impact for '{s}'", 1, 5, 3, key=s+"_imp")
            priorities.append({"Solution": s, "Effort": effort, "Impact": impact})

        df_priorities = pd.DataFrame(priorities)
        st.markdown("### 📊 Prioritization Matrix")
        st.dataframe(df_priorities)

        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_priorities.to_excel(writer, index=False, sheet_name="Improvements")
        output.seek(0)

        st.download_button(
            "📥 Download Improvement Plan",
            data=output,
            file_name="improvement_plan.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
#-------------------- CONTROL MODULE --------------------#

def run_control_ui():
    st.subheader("🟥 DMAIC - CONTROL Phase")
    st.write("Ensure sustained improvements by standardizing solutions and monitoring performance.")
    st.markdown("### 📋 Control Plan")
    control_items = []
    num_controls = st.number_input("How many control items do you want to define?", min_value=1, max_value=20, value=3)

    for i in range(num_controls):
        with st.expander(f"Control Item {i+1}"):
            process = st.text_input(f"Process Step {i+1}", key=f"step_{i}")
            metric = st.text_input(f"Monitoring Metric {i+1}", key=f"metric_{i}")
            target = st.text_input(f"Target Value {i+1}", key=f"target_{i}")
            frequency = st.selectbox(f"Monitoring Frequency {i+1}", ["Daily", "Weekly", "Monthly"], key=f"freq_{i}")
            responsible = st.text_input(f"Responsible Person {i+1}", key=f"resp_{i}")

            control_items.append({
                "Process Step": process,
                "Metric": metric,
                "Target": target,
                "Frequency": frequency,
                "Responsible": responsible
            })

    if control_items:
        df_controls = pd.DataFrame(control_items)
        st.markdown("### ✅ Control Plan Summary")
        st.dataframe(df_controls)

        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_controls.to_excel(writer, index=False, sheet_name="Control Plan")
        output.seek(0)

        st.download_button(
            label="📥 Download Control Plan (Excel)",
            data=output,
            file_name=f"control_plan_{datetime.now().date()}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    st.markdown("##### 📈 Upload Control Chart Data")
    chart_file = st.file_uploader("Upload control chart data (.csv)", type=["csv"])

    if chart_file:
        df_chart = pd.read_csv(chart_file)
        st.markdown("### 📊 Control Chart Preview")
        st.dataframe(df_chart.head())

        import matplotlib.pyplot as plt
        import seaborn as sns

        numeric_cols = df_chart.select_dtypes(include='number').columns.tolist()
        if numeric_cols:
            metric_col = st.selectbox("Select a metric to plot:", numeric_cols)

            mean = df_chart[metric_col].mean()
            std = df_chart[metric_col].std()
            ucl = mean + 3 * std
            lcl = mean - 3 * std

            fig, ax = plt.subplots()
            ax.plot(df_chart[metric_col], marker='o', linestyle='-')
            ax.axhline(mean, color='green', linestyle='--', label='Mean')
            ax.axhline(ucl, color='red', linestyle='--', label='UCL')
            ax.axhline(lcl, color='red', linestyle='--', label='LCL')
            ax.set_title(f"Control Chart - {metric_col}")
            ax.legend()
            st.pyplot(fig)

    st.markdown("### 📘 Additional Tools")
    st.markdown("- Use **Control Charts** to monitor key metrics over time.")
    st.markdown("- Establish **Standard Operating Procedures (SOPs)** for consistency.")
    st.markdown("- Conduct regular **audits** to ensure compliance and effectiveness.")


# -------------------- MAIN NAVIGATION -------------------- #
st.set_page_config(page_title="DMAIC Tool", layout="centered")
st.title("🧠 Problem Solving Tool - DMAIC")

run_home_ui()

st.markdown("### Select a phase:")
option = st.selectbox("", ["DEFINE", "MEASURE", "ANALYZE","IMPROVE","CONTROL"], index=0)

if option == "DEFINE":
    run_define_ui()
elif option == "MEASURE":
    run_measure_ui()
elif option == "ANALYZE":
    run_analyze_ui()
elif option == "IMPROVE":
    run_improve_ui()
elif option == "CONTROL":
    run_control_ui()