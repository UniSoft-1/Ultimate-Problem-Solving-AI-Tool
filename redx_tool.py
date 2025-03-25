import streamlit as st
import pandas as pd
import json


def run_redx_tool():
    st.title("Shainin Red X® Tool")

    phase = st.radio("Select Red X Phase", [
        "Focus", "Approach", "Converge", "Test",
        "Understand", "Apply", "Leverage", "Generate Report"
    ])

    if phase == "Focus":
        st.subheader("Step 1: Focus Phase")
        business_case = st.text_area("Describe the business case")
        technical_def = st.text_area("Define the technical project")
        impact = st.text_input("Estimate the impact of solving this problem")

        if st.button("Save Focus Phase"):
            st.session_state["redx_focus"] = {
                "business_case": business_case,
                "technical_definition": technical_def,
                "impact_estimate": impact
            }
            st.success("Focus phase saved!")

    elif phase == "Approach":
        st.subheader("Step 2: Approach Phase")
        green_y = st.text_input("What is the Green Y?")
        failure_symptoms = st.text_area("Describe failure symptoms")
        strategy = st.text_area("Investigation strategy")
        verified = st.checkbox("Have you verified the measurement system?")

        uploaded_file = st.file_uploader("Upload measurement CSV file", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.session_state["redx_data"] = df.to_dict(orient="list")
            st.write(df.head())

        if st.button("Save Approach Phase"):
            st.session_state["redx_approach"] = {
                "green_y": green_y,
                "failure_symptoms": failure_symptoms,
                "investigation_strategy": strategy,
                "measurement_verified": verified
            }
            st.success("Approach phase saved!")

    elif phase == "Converge":
        st.subheader("Step 3: Converge Phase")
        red_x_candidates = st.text_input("List possible Red X candidates (comma-separated)")
        elimination_notes = st.text_area("How were irrelevant causes eliminated?")
        bob = st.text_area("Describe Best of Best (BOB) condition")
        wow = st.text_area("Describe Worst of Worst (WOW) condition")

        if st.button("Save Converge Phase"):
            st.session_state["redx_converge"] = {
                "red_x_candidates": red_x_candidates.split(","),
                "elimination_notes": elimination_notes,
                "bob": bob,
                "wow": wow
            }
            st.success("Converge phase saved!")

    elif phase == "Test":
        st.subheader("Step 4: Test Phase")
        tests = st.text_area("Describe the tests performed to confirm Red X")
        doe_needed = st.checkbox("Was a full factorial DOE required?")
        risk = st.text_area("Summarize the risk assessment")

        if st.button("Save Test Phase"):
            st.session_state["redx_test"] = {
                "tests_conducted": tests,
                "doe_needed": doe_needed,
                "risk_assessment": risk
            }
            st.success("Test phase saved!")

    elif phase == "Understand":
        st.subheader("Step 5: Understand Phase")
        relation = st.text_area("Describe the relationship between Green Y and Red X")
        limits = st.text_area("What are the customer-defined limits?")
        tolerance = st.text_input("Define tolerance limits")

        if st.button("Save Understand Phase"):
            st.session_state["redx_understand"] = {
                "green_y_red_x_relation": relation,
                "customer_limits": limits,
                "tolerance_definition": tolerance
            }
            st.success("Understand phase saved!")

    elif phase == "Apply":
        st.subheader("Step 6: Apply Phase")
        corrective_actions = st.text_area("List corrective actions proposed")
        verification = st.text_area("How was effectiveness verified?")
        procedures_updated = st.checkbox("Were procedures updated?")

        if st.button("Save Apply Phase"):
            st.session_state["redx_apply"] = {
                "corrective_actions": corrective_actions,
                "effectiveness_verification": verification,
                "procedure_update": procedures_updated
            }
            st.success("Apply phase saved!")

    elif phase == "Leverage":
        st.subheader("Step 7: Leverage Phase")
        control = st.text_area("Define control measures in place")
        feedback = st.text_area("Describe feedback loops implemented")
        savings = st.text_area("How are savings being tracked?")
        next_steps = st.text_area("What are the next steps?")

        if st.button("Save Leverage Phase"):
            st.session_state["redx_leverage"] = {
                "control_measures": control,
                "feedback_loops": feedback,
                "savings_tracking": savings,
                "next_steps": next_steps
            }
            st.success("Leverage phase saved!")

    elif phase == "Generate Report":
        st.subheader("Generate Shainin Red X® Report")
        all_data = {
            "focus": st.session_state.get("redx_focus", {}),
            "approach": st.session_state.get("redx_approach", {}),
            "converge": st.session_state.get("redx_converge", {}),
            "test": st.session_state.get("redx_test", {}),
            "understand": st.session_state.get("redx_understand", {}),
            "apply": st.session_state.get("redx_apply", {}),
            "leverage": st.session_state.get("redx_leverage", {}),
            "data": st.session_state.get("redx_data", {})
        }

        report_json = json.dumps(all_data, indent=2)
        st.download_button(
            label="Download Red X Report (JSON)",
            data=report_json,
            file_name="shainin_redx_report.json",
            mime="application/json"
        )
        st.code(report_json, language="json")
