import streamlit as st
from transformers import pipeline
import plotly.graph_objects as go
import pandas as pd

# AI Conversation Class with Hugging Face Integration
class AIConversation:
    def __init__(self, categories=None):
        self.categories = categories if categories else {
            "People": [],
            "Process": [],
            "Machines": [],
            "Materials": [],
            "Environment": [],
        }
        
        # Initialize the text-generation pipeline with a different model (DialoGPT-small)
        self.conversational_pipeline = pipeline("text-generation", model="DialoGPT-small")
    
    def ai_interaction(self, category, history=""):
        """Use Hugging Face to ask the user about the specific category"""
        prompt = f"Tell me more about the {category} issues in your manufacturing process. Please provide as much detail as possible."
        
        # Generate the AI's response
        conversation_input = self.conversational_pipeline(prompt, max_length=200, num_return_sequences=1)
        ai_response = conversation_input[0]['generated_text']
        return ai_response
    
    def start_conversation(self):
        """Simulates AI asking questions for each category"""
        for category in self.categories:
            ai_response = self.ai_interaction(category)
            st.write(f"AI: {ai_response}")
            user_input = st.text_area(f"Provide details about the {category.lower()} problems:", "")
            if user_input:
                self.categories[category].append(user_input)
        
    def analyze_causes(self):
        """Analyze the causes and provide insights"""
        all_causes = []
        for category in self.categories:
            all_causes.extend(self.categories[category])
        
        cause_counts = pd.Series(all_causes).value_counts()
        top_causes = cause_counts.head(3)
        
        insights = f"Top potential root causes based on the most frequent issues:\n"
        for cause, count in top_causes.items():
            insights += f"- {cause} (mentioned {count} times)\n"
        
        st.write(insights)
    
    def generate_ishikawa_diagram(self):
        """Generate and plot the Ishikawa Diagram (Fishbone diagram) with Plotly"""
        problem = st.text_input("Enter the main problem to display at the head of the fishbone:")

        if problem:
            fig = go.Figure()

            # Define categories as bones
            categories = list(self.categories.keys())
            
            for i, category in enumerate(categories):
                # X and Y positions for each category (bones)
                fig.add_trace(go.Scatter(
                    x=[i] * len(self.categories[category]),
                    y=[j for j in range(len(self.categories[category]))],
                    mode='markers+text',
                    text=self.categories[category],
                    textposition="middle center",
                    marker=dict(size=10, color="blue"),
                    name=category,
                ))

            # Connect all categories to the problem (main node)
            for i in range(len(categories)):
                fig.add_trace(go.Scatter(
                    x=[i, -1],
                    y=[0, 0],
                    mode='lines+text',
                    line=dict(color="black", width=2),
                    text=[categories[i], problem],
                    textposition="bottom center",
                    showlegend=False,
                ))

            # Add title and layout
            fig.update_layout(
                title=f"Ishikawa Diagram for '{problem}'",
                xaxis=dict(showgrid=False, zeroline=False),
                yaxis=dict(showgrid=False, zeroline=False),
                showlegend=True,
                template="plotly_dark",
                plot_bgcolor="white"
            )
            
            st.plotly_chart(fig)

# Streamlit UI
def main():
    st.title("AI-Powered Ishikawa Diagram Generator")
    
    # Custom categories or use defaults
    user_categories = st.radio("Would you like to add custom categories?", ['No', 'Yes'])
    categories = None
    if user_categories == 'Yes':
        categories = {}
        while True:
            category_name = st.text_input("Enter category name (or leave blank to finish):")
            if not category_name:
                break
            categories[category_name] = []

    # Create AI conversation instance
    ai_tool = AIConversation(categories)
    
    # Start the conversation to collect information
    st.write("AI is asking about the manufacturing issues...")
    ai_tool.start_conversation()
    
    # Analyze causes and provide insights
    ai_tool.analyze_causes()
    
    # Generate the Ishikawa Diagram with interactivity
    ai_tool.generate_ishikawa_diagram()

if __name__ == "__main__":
    main()
