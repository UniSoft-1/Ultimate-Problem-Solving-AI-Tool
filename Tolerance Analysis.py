import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import io
import json
from fpdf import FPDF

def run_tolerance_analysis():
    st.title("🔩 Tolerance Stack-Up Analysis Tool")

    st.write("""
    This tool includes:
    - Worst-case and RSS Analysis
    - Monte Carlo Simulation
    - Sensitivity Analysis
    - Visual Stack-Up Builder
    - Functional Limit Validation
    - Export to Excel and PDF
    - Save/Load Stack Configurations
    """)

    # Inputs
    num_components = st.number_input("Number of components", min_value=2, max_value=20, value=3)
    dims = []
    for i in range(int(num_components)):
        name = st.text_input(f"Component {i+1} name", f"C{i+1}")
        nominal = st.number_input(f"Nominal length for {name}", key=f"nom_{i}")
        plus_tol = st.number_input(f"Upper tolerance (+) for {name}", key=f"plus_{i}")
        minus_tol = st.number_input(f"Lower tolerance (-) for {name}", key=f"minus_{i}")
        dims.append({"name": name, "nominal": nominal, "+tol": plus_tol, "-tol": minus_tol})

    # Functional limits
    st.markdown("---")
    st.subheader("✅ Functional Limit Validation")
    lower_limit = st.number_input("Min acceptable stack length", value=0.0)
    upper_limit = st.number_input("Max acceptable stack length", value=999.9)

    # Save/Load JSON templates
    st.markdown("---")
    st.subheader("💾 Save/Load Stack Configuration")
    save_name = st.text_input("Save as filename", "stack_template")
    if st.button("Save Template"):
        with open(f"{save_name}.json", "w") as f:
            json.dump(dims, f)
        st.success(f"Saved as {save_name}.json")

    uploaded_template = st.file_uploader("Load a saved stack-up JSON", type=["json"])
    if uploaded_template:
        dims = json.load(uploaded_template)
        st.info("Template loaded. Re-run analysis if needed.")

    st.markdown("---")
    if st.button("Run Analysis"):
        df = pd.DataFrame(dims)
        df['worst_case_max'] = df['nominal'] + df['+tol']
        df['worst_case_min'] = df['nominal'] - df['-tol']
        df['rss_variance'] = ((df['+tol'] + df['-tol']) / 2) ** 2

        total_nominal = df['nominal'].sum()
        worst_case_max = df['worst_case_max'].sum()
        worst_case_min = df['worst_case_min'].sum()
        rss_std = np.sqrt(df['rss_variance'].sum())

        st.subheader("🧮 Stack-Up Summary")
        st.write(df[['name', 'nominal', '+tol', '-tol']])
        st.write(f"**Total Nominal:** {total_nominal:.3f} mm")
        st.write(f"**Worst Case Range:** {worst_case_min:.3f} – {worst_case_max:.3f} mm")
        st.write(f"**RSS Range (±3σ):** {total_nominal - 3*rss_std:.3f} – {total_nominal + 3*rss_std:.3f} mm")

        if worst_case_min > upper_limit or worst_case_max < lower_limit:
            st.error("❌ Stack does NOT meet functional limits.")
        elif total_nominal - 3*rss_std > upper_limit or total_nominal + 3*rss_std < lower_limit:
            st.warning("⚠️ Stack may occasionally fall outside limits.")
        else:
            st.success("✅ Stack meets functional limits.")

        if st.button("Show Monte Carlo Simulation"):
            st.subheader("🎲 Monte Carlo Simulation")
            samples = []
            for _ in range(10000):
                value = 0
                for row in dims:
                    mean = row['nominal']
                    std = (row['+tol'] + row['-tol']) / 6
                    value += random.gauss(mean, std)
                samples.append(value)
            samples = np.array(samples)
            st.write(f"Mean: {samples.mean():.3f} mm | Std Dev: {samples.std():.3f} mm")
            fig, ax = plt.subplots()
            ax.hist(samples, bins=50, color='lightblue')
            st.pyplot(fig)

        if st.button("Run Sensitivity Analysis"):
            st.subheader("🔍 Sensitivity Analysis")
            df['contribution_%'] = 100 * df['rss_variance'] / df['rss_variance'].sum()
            st.write(df[['name', 'contribution_%']])
            top = df.loc[df['contribution_%'].idxmax()]
            st.info(f"**{top['name']}** contributes the most: {top['contribution_%']:.2f}%")

        if st.button("Show Visual Stack-Up Builder"):
            st.subheader("🧱 Visual Stack-Up")
            fig, ax = plt.subplots(figsize=(10, 2))
            pos = 0
            for _, row in df.iterrows():
                ax.barh(0, row['nominal'], left=pos, height=0.3, label=row['name'])
                pos += row['nominal']
            ax.set_xlim(0, pos * 1.1)
            ax.set_yticks([])
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            st.pyplot(fig)

        if st.button("Export to Excel"):
            export_df = df[['name', 'nominal', '+tol', '-tol', 'contribution_%']]
            export_df.loc['TOTAL'] = ['TOTAL', df['nominal'].sum(), df['+tol'].sum(), df['-tol'].sum(), '—']
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                export_df.to_excel(writer, index=False)
            st.download_button("Download Excel Report", buffer.getvalue(), file_name="tolerance_analysis.xlsx")

        if st.button("Export to PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Tolerance Stack-Up Summary", ln=1)
            pdf.cell(200, 10, txt=f"Nominal: {total_nominal:.3f} mm", ln=2)
            pdf.cell(200, 10, txt=f"Worst Case: {worst_case_min:.3f} – {worst_case_max:.3f} mm", ln=3)
            pdf.cell(200, 10, txt=f"RSS ±3σ: {total_nominal - 3*rss_std:.3f} – {total_nominal + 3*rss_std:.3f} mm", ln=4)
            pdf_buffer = io.BytesIO()
            pdf.output(pdf_buffer)
            st.download_button("Download PDF Report", data=pdf_buffer.getvalue(), file_name="tolerance_summary.pdf")

if __name__ == "__main__":
    run_tolerance_analysis()
