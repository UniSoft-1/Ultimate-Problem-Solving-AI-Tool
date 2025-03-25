import streamlit as st
import pandas as pd
import random
import itertools
from pyDOE2 import fullfact
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from scipy.optimize import linprog
from scipy.optimize import milp, LinearConstraint, Bounds


def treatment_randomizer():
    st.title("🎲 Treatment Randomizer - DOE Planner")
    st.write("""
        Generate randomized treatment combinations for experimental design.
        Ideal for full factorial DOE setups.
    """)

    num_factors = st.number_input("Number of factors", min_value=1, max_value=10, value=2)
    factors = {}

    for i in range(int(num_factors)):
        factor_name = st.text_input(f"Factor {i+1} name", value=f"Factor_{i+1}")
        levels_raw = st.text_input(f"Levels for {factor_name} (comma separated)", value="Low,High", key=f"levels_{i}")
        levels = [lvl.strip() for lvl in levels_raw.split(",") if lvl.strip() != ""]
        factors[factor_name] = levels

    num_replicates = st.number_input("Number of replicates", min_value=1, max_value=20, value=1)
    block_name = st.text_input("Optional blocking column name (leave blank if none)", value="")

    if st.button("Generate Randomized Table"):
        try:
            base_design = list(itertools.product(*factors.values()))
            full_design = base_design * int(num_replicates)
            random.shuffle(full_design)

            df = pd.DataFrame(full_design, columns=factors.keys())
            df.insert(0, "Run", range(1, len(df)+1))

            if block_name.strip():
                num_blocks = st.number_input("Number of blocks", min_value=1, max_value=len(df), value=2)
                block_cycle = [f"Block_{i+1}" for i in range(int(num_blocks))] * (len(df)//int(num_blocks)+1)
                df[block_name] = block_cycle[:len(df)]

            st.success("✅ Randomized treatment table generated!")
            st.dataframe(df)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, file_name="randomized_treatment_plan.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Error: {e}")


def full_factorial_generator():
    st.title("🧪 Full Factorial DOE Generator")
    st.write("""
        Create a full factorial design matrix and download the result.
    """)

    num_factors = st.number_input("Number of factors", min_value=1, max_value=10, value=3, key="fullfact_factors")
    levels_list = []
    for i in range(int(num_factors)):
        levels = st.number_input(f"Number of levels for Factor {i+1}", min_value=2, max_value=5, value=2, key=f"ff_level_{i}")
        levels_list.append(levels)

    if st.button("Generate Full Factorial Design"):
        try:
            design = fullfact(levels_list)
            df = pd.DataFrame(design, columns=[f"F{i+1}" for i in range(len(levels_list))])
            df = df.astype(int) + 1
            df.insert(0, "Run", range(1, len(df)+1))

            st.success("✅ Full factorial design generated!")
            st.dataframe(df)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, file_name="full_factorial_design.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Error: {e}")


def response_optimizer():
    st.title("📈 Response Surface Analyzer")
    st.write("""
        Upload experimental data and optimize a response variable using linear regression.
        Also includes response surface plots.
    """)

    uploaded_file = st.file_uploader("Upload your experiment data (CSV)", type="csv")
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.dataframe(data.head())

        factors = st.multiselect("Select factors (X variables)", options=data.columns)
        response = st.selectbox("Select response variable (Y)", options=[col for col in data.columns if col not in factors])

        if st.button("Run Optimization"):
            try:
                X = data[factors].values
                y = data[response].values
                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)

                st.write(f"R² Score: {model.score(X, y):.4f}")
                coeffs = dict(zip(factors, model.coef_))
                st.write("Coefficients:", coeffs)

                if len(factors) == 2:
                    fig = px.scatter_3d(data, x=factors[0], y=factors[1], z=response, color=response)
                    st.plotly_chart(fig)
                elif len(factors) == 1:
                    fig = px.scatter(data, x=factors[0], y=response, trendline="ols")
                    st.plotly_chart(fig)

            except Exception as e:
                st.error(f"Failed to optimize: {e}")


def linear_programming_optimizer():
    st.title("⚙️ Linear Programming Process Optimizer")
    st.write("""
        Define an objective and constraints to find the optimal resource allocation using linear or integer programming.
    """)

    st.subheader("Objective Function")
    c_raw = st.text_input("Enter coefficients of objective function (e.g., -5, -3 for max Z = 5x + 3y)", "-5,-3")
    c = [float(val) for val in c_raw.split(',') if val.strip()]

    st.subheader("Constraints")
    A = []
    b = []
    num_constraints = st.number_input("Number of constraints", min_value=1, max_value=10, value=2)
    for i in range(int(num_constraints)):
        coeffs = st.text_input(f"Coefficients for constraint {i+1} (e.g., 1,2 for x + 2y)", key=f"ac_{i}")
        rhs = st.number_input(f"RHS value for constraint {i+1}", key=f"bc_{i}")
        A.append([float(v) for v in coeffs.split(',') if v.strip()])
        b.append(rhs)

    use_integer = st.checkbox("Use Integer Programming (MILP)", value=False)

    if st.button("Solve Optimization Problem"):
        try:
            if use_integer:
                bounds = Bounds(lb=[0]*len(c), ub=[None]*len(c))
                integrality = np.ones(len(c))
                constraints = LinearConstraint(np.array(A), lb=-np.inf, ub=b)
                result = milp(c=c, integrality=integrality, constraints=constraints, bounds=bounds)
            else:
                result = linprog(c=c, A_ub=A, b_ub=b, method='highs')

            if result.success:
                st.success("✅ Optimal solution found!")
                st.write("Optimal values:", result.x)
                st.write("Optimal objective value:", -result.fun if not use_integer else result.fun)
            else:
                st.warning("No optimal solution found.")
        except Exception as e:
            st.error(f"Error: {e}")


if __name__ == "__main__":
    st.title("🧠 Randomization & Optimization Toolkit")
    tool_options = [
        "Treatment Randomizer",
        "Full Factorial Generator",
        "Response Optimizer",
        "Process Optimizer"
    ]
    selected_tool = st.selectbox("Select a tool to get started:", tool_options)

    if selected_tool == "Treatment Randomizer":
        treatment_randomizer()
    elif selected_tool == "Full Factorial Generator":
        full_factorial_generator()
    elif selected_tool == "Response Optimizer":
        response_optimizer()
    elif selected_tool == "Process Optimizer":
        linear_programming_optimizer()