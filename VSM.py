import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def create_value_stream_map():
    st.title("Value Stream Map Tool")

    # Step 1: Gather process details
    stages = []
    process_count = st.number_input("How many stages are there in the process?", min_value=1, step=1)

    for i in range(process_count):
        st.subheader(f"Stage {i+1}: Process Description")

        stage_name = st.text_input(f"Enter the name of Stage {i+1} (e.g., Assembly, Inspection, etc.)")
        talk_time = st.number_input(f"Enter talk time for {stage_name} (in minutes)", min_value=0.0, step=0.1)
        changeover_time = st.number_input(f"Enter changeover time for {stage_name} (in minutes)", min_value=0.0, step=0.1)
        uptime = st.number_input(f"Enter uptime percentage for {stage_name} (0 to 100)", min_value=0.0, max_value=100.0, step=0.1)
        oee = st.number_input(f"Enter OEE for {stage_name} (in percentage)", min_value=0.0, max_value=100.0, step=0.1)
        available_time = st.number_input(f"Enter available time for {stage_name} (in hours per shift)", min_value=0.0, step=0.1)
        shifts = st.number_input(f"Enter the number of shifts for {stage_name}", min_value=1, step=1)

        # Store the collected data for each stage
        stages.append({
            "name": stage_name,
            "talk_time": talk_time,
            "changeover_time": changeover_time,
            "uptime": uptime,
            "oee": oee,
            "available_time": available_time,
            "shifts": shifts
        })

    # Step 2: Create a diagram of the Value Stream Map
    # We'll use Plotly to create the VSM diagram for a more interactive experience
    fig = go.Figure()

    stage_positions = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]  # Adjust positions as necessary

    # Add process stages as rectangles
    for idx, stage in enumerate(stages):
        fig.add_shape(
            type="rect",
            x0=stage_positions[idx][0],
            y0=stage_positions[idx][1] - 0.2,
            x1=stage_positions[idx][0] + 0.2,
            y1=stage_positions[idx][1] + 0.2,
            line=dict(color="RoyalBlue", width=2),
            fillcolor="LightSkyBlue",
            opacity=0.7
        )
        
        # Add text to display stage name and metrics inside the box
        fig.add_annotation(
            x=stage_positions[idx][0] + 0.1,
            y=stage_positions[idx][1],
            text=f"{stage['name']}\nOEE: {stage['oee']}%\nUptime: {stage['uptime']}%\nAvailable Time: {stage['available_time']} hrs",
            showarrow=False,
            font=dict(size=12, color="black"),
            align="center"
        )
    
    # Add arrows to show the flow between stages
    for i in range(len(stages) - 1):
        fig.add_annotation(
            x=stage_positions[i][0] + 0.2,
            y=stage_positions[i][1],
            ax=stage_positions[i + 1][0],
            ay=stage_positions[i + 1][1],
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            opacity=0.7
        )
    
    # Adjust layout for better spacing and appearance
    fig.update_layout(
        title="Value Stream Map",
        showlegend=False,
        xaxis=dict(range=[-0.1, 1.1], zeroline=False, showgrid=False),
        yaxis=dict(range=[-0.5, len(stages) * 0.5], zeroline=False, showgrid=False),
        plot_bgcolor="white",
        height=500
    )

    # Show the VSM diagram
    st.plotly_chart(fig)

    # Step 3: Generate insights based on the VSM
    st.write("### Insights from the Value Stream Map:")

    for stage in stages:
        st.write(f"Stage: {stage['name']}")
        st.write(f"- Available Time: {stage['available_time']} hours")
        st.write(f"- OEE: {stage['oee']}%")
        st.write(f"- Talk Time: {stage['talk_time']} min")
        st.write(f"- Changeover Time: {stage['changeover_time']} min")
        st.write(f"- Uptime: {stage['uptime']}%")
        st.write("\n")

    st.write("### Conclusion:")
    st.write("Based on the data above, engineers can identify bottlenecks, areas for improvement, and optimize the process flow.")

# Running the Streamlit interface
def run_streamlit_interface():
    st.title("Process Analysis Tools")
    
    # Step 1: Run Value Stream Map Tool
    if st.button("Create Value Stream Map"):
        create_value_stream_map()

# Running the interface
if __name__ == "__main__":
    run_streamlit_interface()
