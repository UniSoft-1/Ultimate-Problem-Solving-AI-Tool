import streamlit as st
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import numpy as np
from statsmodels.stats.weightstats import ttest_ind
import statsmodels.formula.api as smf
from scipy.optimize import curve_fit
import io
import graphviz
import random

# Weibull Analysis Tool
def weibull_analysis_tool(df):
    st.subheader("Weibull Reliability Analysis")

    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    time_col = st.selectbox("Select Time-to-Failure Column", numeric_columns)

    if st.button("Run Weibull Analysis"):
        data = df[time_col].dropna().values
        data.sort()
        n = len(data)
        ranks = np.arange(1, n + 1)
        F = (ranks - 0.3) / (n + 0.4)  # Median rank approximation

        def weibull_cdf(t, beta, eta):
            return 1 - np.exp(-(t / eta) ** beta)

        try:
            popt, _ = curve_fit(weibull_cdf, data, F, bounds=(0, [10., 10000.]))
            beta, eta = popt

            st.write(f"Estimated shape (β): {beta:.4f}")
            st.write(f"Estimated scale (η): {eta:.4f}")

            fig, ax = plt.subplots()
            ax.plot(data, F, 'o', label='Empirical CDF')
            ax.plot(data, weibull_cdf(data, *popt), label='Weibull Fit')
            ax.set_xlabel("Time to Failure")
            ax.set_ylabel("Cumulative Probability")
            ax.set_title("Weibull Probability Plot")
            ax.legend()
            st.pyplot(fig)

            # AI Interpretation
            if beta < 1:
                st.warning("β < 1 suggests early-life failures (infant mortality).")
            elif beta == 1:
                st.info("β ≈ 1 indicates random failures (constant hazard).")
            else:
                st.success("β > 1 indicates wear-out failures — reliability decreases over time.")

        except Exception as e:
            st.error(f"Error fitting Weibull model: {e}")

# Lifetime Analysis Tool (Kaplan-Meier & Hazard Function)
def lifetime_analysis_tool(df):
    st.subheader("Lifetime Data Analysis (Kaplan-Meier & Hazard Function)")

    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    time_col = st.selectbox("Select Time-to-Failure Column (for KM analysis)", numeric_columns)

    if st.button("Run Lifetime Analysis"):
        try:
            from lifelines import KaplanMeierFitter

            kmf = KaplanMeierFitter()
            T = df[time_col].dropna().values
            E = np.ones_like(T)  # All events observed (no censoring for now)

            kmf.fit(T, event_observed=E)

            st.write(f"Median Survival Time: {kmf.median_survival_time_:.2f}")

            fig, ax = plt.subplots()
            kmf.plot_survival_function(ax=ax)
            ax.set_title("Kaplan-Meier Survival Curve")
            ax.set_xlabel("Time")
            ax.set_ylabel("Survival Probability")
            st.pyplot(fig)

            # Hazard Function Estimate (naive)
            st.write("### Estimated Hazard Function (Naive Method)")
            durations = np.sort(T)
            delta_t = np.diff(durations)
            hazard = 1 / delta_t  # Approximate hazard rate change

            fig2, ax2 = plt.subplots()
            ax2.plot(durations[1:], hazard)
            ax2.set_title("Hazard Function Estimate")
            ax2.set_xlabel("Time")
            ax2.set_ylabel("Hazard Rate")
            st.pyplot(fig2)

            # AI Insight
            st.info("If the survival curve drops quickly, early failures dominate. If it stays flat, the system is highly reliable over time.")

        except ImportError:
            st.error("Please install the 'lifelines' library to run Kaplan-Meier analysis. Use: pip install lifelines")
        except Exception as e:
            st.error(f"Error in lifetime analysis: {e}")
        
# System Reliability Calculator
def system_reliability_tool():
    st.subheader("System Reliability Calculator (Series/Parallel)")

    st.write("Enter component reliability values and select configuration type.")
    num_components = st.number_input("Number of Components", min_value=2, max_value=20, step=1)

    reliability_inputs = []
    for i in range(int(num_components)):
        r = st.number_input(f"Reliability of Component {i+1} (0 to 1)", min_value=0.0, max_value=1.0, step=0.01, key=f"comp_{i}")
        reliability_inputs.append(r)

    config = st.selectbox("Select Configuration", ["Series", "Parallel"])

    if st.button("Calculate System Reliability"):
        if config == "Series":
            system_reliability = np.prod(reliability_inputs)
            st.write(f"System Reliability (Series): {system_reliability:.4f}")
            st.info("In series systems, one failure leads to total system failure. Reliability drops quickly as more components are added.")

        elif config == "Parallel":
            unreliabilities = [1 - r for r in reliability_inputs]
            system_reliability = 1 - np.prod(unreliabilities)
            st.write(f"System Reliability (Parallel): {system_reliability:.4f}")
            st.success("Parallel systems are more robust. Redundancy increases reliability.")

        fig, ax = plt.subplots()
        ax.bar(range(1, len(reliability_inputs)+1), reliability_inputs)
        ax.set_xlabel("Component")
        ax.set_ylabel("Reliability")
        ax.set_title("Component Reliabilities")
        st.pyplot(fig)

# Accelerated Life Testing Tool
def accelerated_life_testing_tool():
    st.subheader("Accelerated Life Testing (ALT)")

    st.write("Enter stress levels, failure times, and choose model type for ALT analysis.")

    uploaded_file = st.file_uploader("Upload ALT Data (Stress, Time to Failure)", type=["csv", "xlsx"])

    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)

        st.write("### Preview of Uploaded Data")
        st.write(data.head())

        stress_col = st.selectbox("Select Stress Column", data.select_dtypes(include=['number']).columns)
        time_col = st.selectbox("Select Time-to-Failure Column", data.select_dtypes(include=['number']).columns)
        model_type = st.selectbox("Select ALT Model", ["Arrhenius", "Inverse Power Law"])

        if st.button("Run ALT Analysis"):
            try:
                x = data[stress_col].values
                y = data[time_col].values

                if model_type == "Arrhenius":
                    x_model = 1 / (x + 273.15)  # convert °C to Kelvin
                else:  # Inverse Power Law
                    x_model = np.log(x)

                log_y = np.log(y)

                coeffs = np.polyfit(x_model, log_y, 1)
                a, b = coeffs
                st.write(f"Fitted Model: ln(Time) = {a:.4f} * x + {b:.4f}")

                fig, ax = plt.subplots()
                ax.scatter(x_model, log_y, label='Data')
                ax.plot(x_model, np.polyval(coeffs, x_model), label='Fitted Line', color='red')
                ax.set_xlabel("1/T (Arrhenius) or ln(Stress)")
                ax.set_ylabel("ln(Time to Failure)")
                ax.set_title(f"ALT Model Fit ({model_type})")
                ax.legend()
                st.pyplot(fig)

                st.info("Use the fitted model to estimate life at normal-use stress conditions.")

            except Exception as e:
                st.error(f"Error in ALT analysis: {e}")

# Bathtub Curve Visualization Tool
def bathtub_curve_tool():
    st.subheader("Bathtub Curve Visualization")

    st.write("Simulate or visualize the classic bathtub curve representing product failure behavior over time.")

    time = np.linspace(0, 100, 500)
    infant_mortality = 0.05 * np.exp(-0.1 * time)
    constant_failure = np.full_like(time, 0.01)
    wear_out = 0.0005 * np.exp(0.1 * (time - 60))

    hazard_rate = infant_mortality + constant_failure + wear_out

    fig, ax = plt.subplots()
    ax.plot(time, hazard_rate, label="Total Hazard Rate", linewidth=2)
    ax.plot(time, infant_mortality, '--', label="Infant Mortality")
    ax.plot(time, constant_failure, '--', label="Useful Life")
    ax.plot(time, wear_out, '--', label="Wear-Out")
    ax.set_xlabel("Time")
    ax.set_ylabel("Hazard Rate")
    ax.set_title("Bathtub Curve")
    ax.legend()
    st.pyplot(fig)

    st.info("Interpretation:")
    st.markdown("- **Early Failures:** High hazard due to design/manufacturing issues\n- **Useful Life:** Constant hazard rate\n- **Wear-Out:** Increasing hazard due to aging")

# Goodness-of-Fit Evaluation Tool
def goodness_of_fit_tool(df):
    st.subheader("Goodness-of-Fit Evaluation")

    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    col = st.selectbox("Select Failure Time Column for Fit Test", numeric_columns)

    dist_options = ["weibull_min", "lognorm", "expon", "gamma"]
    selected_distributions = st.multiselect("Select Distributions to Fit", dist_options, default=dist_options)

    if st.button("Evaluate Fit"):
        data = df[col].dropna().values
        fig, ax = plt.subplots()
        sns.histplot(data, bins=20, kde=False, stat="density", color="lightgray", label="Data", ax=ax)

        results = []
        for dist_name in selected_distributions:
            dist = getattr(stats, dist_name)
            params = dist.fit(data)
            x = np.linspace(min(data), max(data), 100)
            pdf = dist.pdf(x, *params)
            ax.plot(x, pdf, label=dist_name)

            stat, p = stats.kstest(data, dist_name, args=params)
            results.append((dist_name, p))

        ax.set_title("Distribution Fit Comparison")
        ax.legend()
        st.pyplot(fig)

        st.write("### Goodness-of-Fit Results (Kolmogorov–Smirnov Test)")
        for name, pval in results:
            if pval < 0.05:
                st.error(f"{name}: Poor fit (p = {pval:.4f})")
            else:
                st.success(f"{name}: Good fit (p = {pval:.4f})")

# Maintenance Scheduling Tool
def maintenance_scheduling_tool():
    st.subheader("Maintenance Scheduling Tool")

    mttf = st.number_input("Enter Mean Time To Failure (MTTF)", min_value=0.1, value=100.0, step=1.0)
    threshold = st.number_input("Select Reliability Threshold (e.g., 0.90 for 90%)", min_value=0.01, max_value=0.99, value=0.90, step=0.01)

    if st.button("Calculate Maintenance Interval"):
        try:
            time_interval = -mttf * np.log(threshold)
            st.write(f"Recommended Maintenance Interval: {time_interval:.2f} time units")

            fig, ax = plt.subplots()
            t = np.linspace(0, time_interval * 2, 200)
            reliability = np.exp(-t / mttf)
            ax.plot(t, reliability, label="Reliability Curve")
            ax.axhline(y=threshold, color='r', linestyle='--', label=f"Threshold ({threshold})")
            ax.axvline(x=time_interval, color='g', linestyle='--', label=f"Maintenance Point")
            ax.set_xlabel("Time")
            ax.set_ylabel("Reliability")
            ax.set_title("Maintenance Scheduling Based on Reliability")
            ax.legend()
            st.pyplot(fig)

            st.info("Interpretation:")
            st.markdown(f"- Schedule maintenance every **{time_interval:.2f}** time units to maintain system reliability above **{threshold:.2%}**.")
        except Exception as e:
            st.error(f"Calculation error: {e}")

# Import/Export Utility Tool
def import_export_tool():
    st.subheader("Data Import & Export Tool")

    # Upload
    uploaded = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
    df = None
    if uploaded:
        try:
            if uploaded.name.endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            st.success("File loaded successfully.")
            st.write(df.head())
        except Exception as e:
            st.error(f"Error reading file: {e}")

    # Download
    if df is not None:
        filename = st.text_input("Enter filename for download (without extension)", "exported_data")
        file_format = st.selectbox("Select export format", ["CSV", "Excel"])

        if st.button("Download Processed File"):
            try:
                buffer = io.BytesIO()
                if file_format == "CSV":
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(label="Download CSV", data=csv, file_name=f"{filename}.csv", mime='text/csv')
                else:
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        df.to_excel(writer, index=False, sheet_name='Sheet1')
                    st.download_button(label="Download Excel", data=buffer.getvalue(), file_name=f"{filename}.xlsx", mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
                st.success("File ready for download.")
            except Exception as e:
                st.error(f"Export failed: {e}")

# Reliability Block Diagram (RBD) Tool
def reliability_block_diagram_tool():
    st.subheader("Reliability Block Diagram (RBD)")
    st.write("Create a simple reliability block diagram with series and parallel components.")

    with st.expander("Define System Layout"):
        diagram_type = st.selectbox("System Layout Type", ["Series", "Parallel", "Mixed"])
        num_blocks = st.slider("Number of Components", 2, 10, 4)
        block_labels = [st.text_input(f"Label for Component {i+1}", f"C{i+1}") for i in range(num_blocks)]

    if st.button("Generate RBD Diagram"):
        dot = graphviz.Digraph()
        dot.attr(rankdir='LR')

        if diagram_type == "Series":
            for label in block_labels:
                dot.node(label)
            for i in range(len(block_labels) - 1):
                dot.edge(block_labels[i], block_labels[i + 1])

        elif diagram_type == "Parallel":
            dot.node("START", shape="point")
            dot.node("END", shape="point")
            for label in block_labels:
                dot.node(label)
                dot.edge("START", label)
                dot.edge(label, "END")

        elif diagram_type == "Mixed":
            dot.node("A")
            dot.node("B")
            dot.node("C")
            dot.node("D")
            dot.edge("A", "B")
            dot.edge("A", "C")
            dot.edge("B", "D")
            dot.edge("C", "D")
            st.warning("Mixed logic is a visual demo here. Real mixed-path logic calculation not included yet.")

        st.graphviz_chart(dot)
        st.info("This is a visual-only RBD. To perform reliability calculation, use the System Reliability Calculator tool.")

# Monte Carlo Simulation for System Reliability
def monte_carlo_simulation_tool():
    st.subheader("Monte Carlo Simulation for System Reliability")
    st.write("Simulate system reliability by repeatedly sampling component states.")

    num_components = st.number_input("Number of Components", min_value=2, max_value=20, value=4, step=1)
    reliability_values = [st.slider(f"Reliability of Component {i+1}", 0.0, 1.0, 0.95, 0.01) for i in range(num_components)]
    system_type = st.selectbox("System Type", ["Series", "Parallel"])
    num_simulations = st.number_input("Number of Simulations", min_value=100, max_value=100000, value=10000, step=1000)

    if st.button("Run Monte Carlo Simulation"):
        success_count = 0
        for _ in range(int(num_simulations)):
            component_states = [random.random() < r for r in reliability_values]
            if system_type == "Series":
                system_success = all(component_states)
            else:
                system_success = any(component_states)
            if system_success:
                success_count += 1

        estimated_reliability = success_count / num_simulations
        st.write(f"Estimated System Reliability after {num_simulations} simulations: {estimated_reliability:.4f}")

        st.info("Monte Carlo methods are useful when analytical solutions are complex or infeasible.")

# Warranty Cost Analyzer
def warranty_cost_analyzer():
    st.subheader("Warranty Cost Analyzer")

    st.write("Estimate expected warranty costs based on failure rate and replacement cost.")

    num_units = st.number_input("Total Units Sold", min_value=1, step=1, value=1000)
    warranty_period = st.number_input("Warranty Period (in time units)", min_value=1.0, value=100.0)
    mttf = st.number_input("Mean Time To Failure (MTTF)", min_value=0.1, value=200.0)
    replacement_cost = st.number_input("Cost per Replacement (in $)", min_value=0.01, value=50.0)

    if st.button("Calculate Warranty Cost"):
        try:
            failure_probability = 1 - np.exp(-warranty_period / mttf)
            expected_failures = num_units * failure_probability
            total_cost = expected_failures * replacement_cost

            st.write(f"Estimated Failure Probability: {failure_probability:.4f}")
            st.write(f"Expected Number of Failures: {expected_failures:.0f}")
            st.write(f"Estimated Total Warranty Cost: ${total_cost:,.2f}")

            st.info("This assumes an exponential failure distribution with constant failure rate.")
        except Exception as e:
            st.error(f"Error calculating warranty cost: {e}")

# Unified AI Insight Generator
def ai_insight_generator(df):
    st.subheader("AI Insight Generator")

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if len(numeric_cols) < 2:
        st.warning("Please upload a dataset with at least two numeric columns.")
        return

    insights = []
    for i in range(len(numeric_cols)):
        for j in range(i+1, len(numeric_cols)):
            col1, col2 = numeric_cols[i], numeric_cols[j]
            corr = df[[col1, col2]].corr().iloc[0,1]
            if abs(corr) > 0.7:
                relationship = "positive" if corr > 0 else "negative"
                insights.append(f"Strong {relationship} correlation between **{col1}** and **{col2}** (r = {corr:.2f}).")

    if insights:
        st.write("### AI-Generated Insights:")
        for msg in insights:
            st.success(msg)
    else:
        st.info("No strong correlations found. Try using more variables or checking for patterns manually.")

# RUN THIS TO ACTIVATE THE TOOLKIT
def run_reliability_toolkit():
    st.title("🔧 Reliability Analysis Toolkit")

    uploaded_file = st.file_uploader("📤 Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])
    df = None
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.success("✅ File uploaded successfully.")
            st.write("### Preview of Uploaded Data")
            st.write(df.head())
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")
            return

    st.markdown("---")
    st.header("🛠 Reliability Tools")

    if df is not None and st.button("Weibull Analysis"):
        weibull_analysis_tool(df)

    if df is not None and st.button("Lifetime Analysis"):
        lifetime_analysis_tool(df)

    if st.button("System Reliability Calculator"):
        system_reliability_tool()

    if st.button("Accelerated Life Testing (ALT)"):
        accelerated_life_testing_tool()

    if st.button("Bathtub Curve Visualization"):
        bathtub_curve_tool()

    if df is not None and st.button("Goodness-of-Fit Evaluation"):
        goodness_of_fit_tool(df)

    if st.button("Maintenance Scheduling Tool"):
        maintenance_scheduling_tool()

    if st.button("Import/Export Tool"):
        import_export_tool()

    if st.button("Reliability Block Diagram"):
        reliability_block_diagram_tool()

    if st.button("Monte Carlo Simulation"):
        monte_carlo_simulation_tool()

    if st.button("Warranty Cost Analyzer"):
        warranty_cost_analyzer()

    if df is not None and st.button("AI Insight Generator"):
        ai_insight_generator(df)


# Ensure this runs the app
if __name__ == "__main__":
    run_reliability_toolkit()
