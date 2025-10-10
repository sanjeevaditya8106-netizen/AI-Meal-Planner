import streamlit as st
import requests
import json
import time
import os
import pandas as pd

# --- Configuration & Gemini API Settings ---
# NOTE: It is critical to set your Gemini API key as an environment variable
# Run your Streamlit app like this:
# GEMINI_API_KEY="YOUR_API_KEY" streamlit run streamlit_app.py
API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyB3M7eOww0MLSs-nEty4rtZxGOgF3YC6Rc")

MODEL_NAME = "gemini-2.5-flash-preview-05-20"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"

# --- JSON Schema for the structured Weekly Plan and Grocery List ---
# This schema ensures the model returns two clear outputs: the plan and the consolidated list.
PLANNER_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "weeklyPlan": {
            "type": "ARRAY",
            "description": "A list of 7 daily meal plans.",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "day": {"type": "STRING", "description": "E.g., Monday, Tuesday."},
                    "breakfast": {"type": "STRING", "description": "The name of the breakfast meal and quantity (e.g., 'Oatmeal with berries (1 serving)')."},
                    "lunch": {"type": "STRING", "description": "The name of the lunch meal."},
                    "dinner": {"type": "STRING", "description": "The name of the dinner meal."}
                },
                "required": ["day", "breakfast", "lunch", "dinner"]
            }
        },
        "optimizedGroceryList": {
            "type": "ARRAY",
            "description": "The aggregated and optimized list of all ingredients needed for the entire week.",
            "items": { "type": "STRING" }
        },
        "notes": {
            "type": "STRING",
            "description": "A brief summary of why this plan fits the user's constraints (e.g., 'This plan is low-carb and uses the chicken breast on Monday and Wednesday.')."
        }
    },
    "required": ["weeklyPlan", "optimizedGroceryList"]
}

def call_gemini_api(days, restrictions, budget, ingredients):
    """
    Handles the request to the Gemini API for generating the meal plan.
    """
    if not API_KEY:
        st.error("Gemini API Key is missing. Please set the GEMINI_API_KEY environment variable.")
        return None

    # --- Construct the System Instruction and User Prompt ---
    system_instruction = (
        "You are an expert, professional Smart Meal Planner AI. Your goal is to create a weekly meal plan "
        "and a consolidated grocery list that strictly adheres to the user's requirements. "
        "You MUST return the plan in the requested JSON schema format exactly. "
        "Ensure the plan is realistic, balanced, and uses recipes that fit the budget and restrictions."
    )

    user_prompt = (
        f"Generate a {days}-day meal plan (including breakfast, lunch, and dinner) using the following constraints:\n"
        f"- **Must-use Ingredients:** {ingredients}\n"
        f"- **Dietary Restrictions/Notes:** {restrictions}\n"
        f"- **Budget/Goal:** {budget}\n"
        f"Generate the 'weeklyPlan' as a list of 7 days, and the 'optimizedGroceryList' as a single, combined, and categorized list of all ingredients needed for all 7 days of meals."
    )

    # --- Construct the API Payload ---
    payload = {
        "contents": [{
            "parts": [{"text": user_prompt}]
        }],
        "systemInstruction": {
            "parts": [{"text": system_instruction}]
        },
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": PLANNER_SCHEMA
        }
    }

    # --- Send Request to Gemini API with Exponential Backoff ---
    max_retries = 3
    retry_delay = 1 # seconds

    for attempt in range(max_retries):
        try:
            url_with_key = f"{API_URL}?key={API_KEY}"
            response = requests.post(
                url_with_key,
                headers={'Content-Type': 'application/json'},
                data=json.dumps(payload),
                timeout=60 # Extended timeout for complex generation
            )
            response.raise_for_status()

            result = response.json()
            
            if not (result.get('candidates') and result['candidates'][0].get('content') and result['candidates'][0]['content'].get('parts')):
                st.error("Model did not return structured content. Please refine the prompt.")
                return None

            json_string = result['candidates'][0]['content']['parts'][0]['text']
            
            # Use json.loads with strict=False to handle possible non-standard JSON escaping
            plan_json = json.loads(json_string, strict=False)

            return plan_json

        except requests.exceptions.HTTPError as e:
            st.error(f"API Error (Attempt {attempt + 1}): Received status code {response.status_code}.")
            if 500 <= response.status_code < 600 and attempt < max_retries - 1:
                st.warning(f"Server error. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            st.error(f"API request failed permanently: {e.response.text}")
            return None
        except Exception as e:
            st.error(f"An unexpected error occurred (Attempt {attempt + 1}): {str(e)}")
            if attempt < max_retries - 1:
                st.warning(f"Error. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            return None
    
    return None

def display_plan(plan):
    """Formats and displays the weekly plan and grocery list."""
    
    st.subheader("Plan Summary")
    st.info(plan.get('notes', 'Plan generated successfully!'))
    
    st.markdown("---")
    
    # 1. Display Weekly Plan as a Data Table
    st.header("🗓️ Weekly Meal Schedule")
    weekly_data = plan.get('weeklyPlan', [])
    if weekly_data:
        df = pd.DataFrame(weekly_data)
        # Apply CSS for better table aesthetics
        st.dataframe(
            df,
            column_config={
                "day": st.column_config.Column("Day", width="small"),
                "breakfast": st.column_config.Column("Breakfast"),
                "lunch": st.column_config.Column("Lunch"),
                "dinner": st.column_config.Column("Dinner"),
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.warning("No weekly plan data was generated.")

    st.markdown("---")
    
    # 2. Display Optimized Grocery List
    st.header("🛒 Optimized Grocery List")
    grocery_list = plan.get('optimizedGroceryList', [])
    if grocery_list:
        # Display list in two columns for easy readability (optimized for print/mobile)
        col1, col2 = st.columns(2)
        half = len(grocery_list) // 2 + len(grocery_list) % 2
        
        with col1:
            st.markdown("\n".join([f"- {item}" for item in grocery_list[:half]]))
        
        with col2:
            st.markdown("\n".join([f"- {item}" for item in grocery_list[half:]]))

    st.balloons()


# --- Streamlit Application Layout (Frontend) ---
def main():
    st.set_page_config(page_title="Smart Meal Planner AI", layout="wide")
    
    st.title("🧠 Smart Meal Planner AI")
    st.markdown("Generate a complete, constraint-adherent **weekly meal plan** and an **optimized grocery list** based on your goals.")

    if not API_KEY:
        st.warning("⚠️ **Warning:** Gemini API Key not found. Please set the `GEMINI_API_KEY` environment variable.")
    
    # --- Input Fields for Constraints ---
    st.subheader("Your Meal Planning Constraints")
    
    col_days, col_budget = st.columns(2)
    with col_days:
        days = st.selectbox(
            "Number of Days to Plan", 
            options=[7, 5, 3],
            index=0
        )
    with col_budget:
        budget = st.text_input(
            "Budget/Cooking Style Goal", 
            value="Affordable, family-friendly meals that take < 30 min to prepare."
        )

    ingredients = st.text_input(
        "Key Ingredients to Use (e.g., 'chicken breast, potatoes, spinach')", 
        value="chicken breast, onion, rice, canned beans"
    )

    restrictions = st.text_area(
        "Dietary Restrictions & Notes (e.g., 'Gluten-free, need high protein, avoid nuts')", 
        value="Low-carb, high-protein for all dinners."
    )

    st.markdown("---")
    
    # --- Generation Button and Logic ---
    if st.button(f"Generate {days}-Day Meal Plan & Grocery List", use_container_width=True, type="primary"):
        if not API_KEY:
            st.error("Cannot generate plan: API Key is missing.")
        elif not ingredients:
            st.warning("Please enter your key ingredients to start planning.")
        else:
            # Show a spinner while the API call is running
            with st.spinner("Generating intelligent meal plan and optimizing grocery list..."):
                plan = call_gemini_api(days, restrictions, budget, ingredients)
            
            # Display the result if successful
            if plan:
                display_plan(plan)

if __name__ == "__main__":
    main()
