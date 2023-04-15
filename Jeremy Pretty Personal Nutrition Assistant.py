# Jeremy Pretty
# Final Project CSC 510
import random
import pandas as pd
from surprise import Dataset, Reader
from surprise import KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy
import os

# Path to the dataset
path_to_csv = os.path.join(os.path.dirname(__file__), 'en.openfoodfacts.org.products.tsv')

# Load the dataset
data = pd.read_csv(path_to_csv, delimiter='\t', low_memory=False)

# Filter out rows with missing values
data = data.dropna(subset=['product_name', 'main_category', 'ingredients_text'])

# Generate a sample user preference dataset
user_preferences = {
    'user_id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
    'product_id': [101, 102, 103, 101, 104, 105, 106, 107, 108],
    'rating': [5, 4, 2, 1, 5, 3, 5, 3, 2]
}

user_preferences_df = pd.DataFrame(user_preferences)

# Read the user preferences data using the Surprise library
reader = Reader(rating_scale=(1, 5))
user_data = Dataset.load_from_df(user_preferences_df[['user_id', 'product_id', 'rating']], reader)

# Split the data into training and testing sets
trainset, testset = train_test_split(user_data, test_size=0.3)

# Train a collaborative filtering model using the KNNBasic algorithm
algo = KNNBasic()
algo.fit(trainset)

# Test the model
predictions = algo.test(testset)
print("Model accuracy:", accuracy.rmse(predictions))

# Generate meal recommendations
meal_categories = {
    "vegan": "Plant-based foods and beverages",
    "vegetarian": "Vegetarian",
    "non_vegetarian": "Meat"
}

# Function to predict user ratings for products
def predict_ratings(user_id, product_ids):
    ratings = []
    for product_id in product_ids:
        rating = algo.predict(user_id, product_id)
        ratings.append((product_id, rating.est))
    return ratings

# Getting the user information, with checks to ensure proper information
def get_user_info():
    age = int(input("Please enter your age: "))
    while age < 0:
        age = int(input("Invalid input. Please enter a valid age: "))

    weight = float(input("Please enter your weight (kg): "))
    while weight < 0:
        weight = float(input("Invalid input. Please enter a valid weight (kg): "))

    activity_levels = ['sedentary', 'moderate', 'active']
    activity_level = input("Please enter your activity level (sedentary, moderate, active): ").lower()
    while activity_level not in activity_levels:
        activity_level = input("Invalid input. Please enter your activity level (sedentary, moderate, active): ").lower()

    dietary_preferences = ['vegan', 'vegetarian', 'non_vegetarian']
    dietary_preference = input("Please enter your dietary preference (vegan, vegetarian, non_vegetarian): ").lower()
    while dietary_preference not in dietary_preferences:
        dietary_preference = input("Invalid input. Please enter your dietary preference (vegan, vegetarian, non_vegetarian): ").lower()

    health_goals = ['lose weight', 'maintain weight', 'gain weight']
    health_goal = input("Please enter your health goal (lose weight, maintain weight, gain weight): ").lower()
    while health_goal not in health_goals:
        health_goal = input("Invalid input. Please enter your health goal (lose weight, maintain weight, gain weight): ").lower()

    allergies = input("Please enter any food allergies you have (separate with commas): ").lower().split(',')
    intolerances = input("Please enter any food intolerances you have (separate with commas): ").lower().split(',')
    
    macro_goals = input("Please enter your macronutrient goals in the format 'protein,carbs,fat' (in grams): ").split(',')
    while len(macro_goals) != 3 or not all([x.strip().isdigit() for x in macro_goals]):
        macro_goals = input("Invalid input. Please enter your macronutrient goals in the format 'protein,carbs,fat' (in grams): ").split(',')

    return age, weight, activity_level, dietary_preference, health_goal, allergies, intolerances, macro_goals

# Generating the meal plan
def generate_meal_plan(user_id, dietary_preference, allergies, intolerances, health_goal):
    meal_category = meal_categories[dietary_preference]
    filtered_data = data[data['main_category'].str.contains(meal_category, case=False, na=False)]

    # Filter out allergens and intolerances
    for allergen in allergies:
        filtered_data = filtered_data[~filtered_data['ingredients_text'].str.lower().str.contains(allergen.strip())]

    for intolerance in intolerances:
        filtered_data = filtered_data[~filtered_data['ingredients_text'].str.lower().str.contains(intolerance.strip())]

    # Filter based on health goal
    if health_goal == 'lose weight':
        filtered_data = filtered_data.sort_values(by='energy_100g', ascending=True)
    elif health_goal == 'gain weight':
        filtered_data = filtered_data.sort_values(by='energy_100g', ascending=False)
    # No additional filtering for 'maintain weight' as it can include a variety of food items

    # Predict user ratings for the filtered products
    product_ids = filtered_data['code'].tolist()
    predicted_ratings = predict_ratings(user_id, product_ids)

    # Sort the products by the predicted ratings
    sorted_ratings = sorted(predicted_ratings, key=lambda x: x[1], reverse=True)

    # Select the top 3 products as the meal plan
    meal_plan = [sorted_ratings[i][0] for i in range(3)]
    meal_plan = filtered_data[filtered_data['code'].isin(meal_plan)]['product_name'].tolist()
    
    return meal_plan


def main():
    age, weight, activity_level, dietary_preference, health_goal, allergies, intolerances, macro_goals = get_user_info()
    meal_plan = generate_meal_plan(age, dietary_preference, allergies, intolerances, health_goal)

    print("Your personalized meal plan:")
    for i, meal in enumerate(meal_plan):
        print(f"{i + 1}. {meal}")


# Creating a user interface for easy reading
import tkinter as tk
from tkinter import ttk
def submit_form():
    age = int(age_entry.get())
    weight = float(weight_entry.get())
    activity_level = activity_level_var.get()
    dietary_preference = dietary_preference_var.get()
    health_goal = health_goal_var.get()
    allergies = allergies_entry.get().lower().split(',')
    intolerances = intolerances_entry.get().lower().split(',')
    macro_goals = macro_goals_entry.get().split(',')

    user_id = 1
    meal_plan = generate_meal_plan(user_id, dietary_preference, allergies, intolerances, health_goal)

    meal_plan_text = "\n".join([f"{i + 1}. {meal}" for i, meal in enumerate(meal_plan)])
    result_label.config(text=f"Your personalized meal plan:\n{meal_plan_text}")


# Create the main application window
app = tk.Tk()
app.title("Personalized Nutrition Assistant")

# Create input fields
age_label = ttk.Label(app, text="Age:")
age_label.grid(column=0, row=0)
age_entry = ttk.Entry(app)
age_entry.grid(column=1, row=0)

weight_label = ttk.Label(app, text="Weight (kg):")
weight_label.grid(column=0, row=1)
weight_entry = ttk.Entry(app)
weight_entry.grid(column=1, row=1)

activity_level_label = ttk.Label(app, text="Activity Level:")
activity_level_label.grid(column=0, row=2)
activity_level_var = tk.StringVar()
activity_level_options = ['sedentary', 'moderate', 'active']
activity_level_menu = ttk.OptionMenu(app, activity_level_var, activity_level_options[0], *activity_level_options)
activity_level_menu.grid(column=1, row=2)

dietary_preference_label = ttk.Label(app, text="Dietary Preference:")
dietary_preference_label.grid(column=0, row=3)
dietary_preference_var = tk.StringVar()
dietary_preference_options = ['vegan', 'vegetarian', 'non_vegetarian']
dietary_preference_menu = ttk.OptionMenu(app, dietary_preference_var, dietary_preference_options[0], *dietary_preference_options)
dietary_preference_menu.grid(column=1, row=3)

health_goal_label = ttk.Label(app, text="Health Goal:")
health_goal_label.grid(column=0, row=4)
health_goal_var = tk.StringVar()
health_goal_options = ['lose weight', 'maintain weight', 'gain weight']
health_goal_menu = ttk.OptionMenu(app, health_goal_var, health_goal_options[0], *health_goal_options)
health_goal_menu.grid(column=1, row=4)

allergies_label = ttk.Label(app, text="Allergies:")
allergies_label.grid(column=0, row=5)
allergies_entry = ttk.Entry(app)
allergies_entry.grid(column=1, row=5)

intolerances_label = ttk.Label(app, text="Intolerances:")
intolerances_label.grid(column=0, row=6)
intolerances_entry = ttk.Entry(app)
intolerances_entry.grid(column=1, row=6)

macro_goals_label = ttk.Label(app, text="Macro Goals (Protein, Carbs, Fat):")
macro_goals_label.grid(column=0, row=7)
macro_goals_entry = ttk.Entry(app)
macro_goals_entry.grid(column=1, row=7)

# Create the submit button
submit_button = ttk.Button(app, text="Submit", command=submit_form)
submit_button.grid(column=1, row=8)

# Create a label to display the results
result_label = ttk.Label(app, text="")
result_label.grid(column=0, row=9, columnspan=2)

# Start the application's main loop
app.mainloop()
