import random
import pandas as pd
from surprise import Dataset, Reader
from surprise import KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load the dataset
data = pd.read_csv('en.openfoodfacts.org.products.tsv', delimiter='\t', low_memory=False)

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

def get_user_info():
    age = int(input("Please enter your age: "))
    weight = float(input("Please enter your weight (kg): "))
    activity_level = input("Please enter your activity level (sedentary, moderate, active): ").lower()
    dietary_preference = input("Please enter your dietary preference (vegan, vegetarian, non_vegetarian): ").lower()
    health_goal = input("Please enter your health goal (lose weight, maintain weight, gain weight): ").lower()
    allergies = input("Please enter any food allergies you have (separate with commas): ").lower().split(',')
    intolerances = input("Please enter any food intolerances you have (separate with commas): ").lower().split(',')
    macro_goals = input("Please enter your macronutrient goals in the format 'protein,carbs,fat' (in grams): ").split(',')

    return age, weight, activity_level, dietary_preference, health_goal, allergies, intolerances, macro_goals


def generate_meal_plan(user_id, dietary_preference, allergies, intolerances):
    meal_category = meal_categories[dietary_preference]
    filtered_data = data[data['main_category'].str.contains(meal_category, case=False, na=False)]

    # Filter out allergens and intolerances
    for allergen in allergies:
        filtered_data = filtered_data[~filtered_data['ingredients_text'].str.lower().str.contains(allergen.strip())]

    for intolerance in intolerances:
        filtered_data = filtered_data[~filtered_data['ingredients_text'].str.lower().str.contains(intolerance.strip())]


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
    user_id, dietary_preference, allergies, intolerances = get_user_info()
    meal_plan = generate_meal_plan(user_id, dietary_preference, allergies, intolerances)

    print("Your personalized meal plan:")
    for i, meal in enumerate(meal_plan):
        print(f"{i + 1}. {meal}")

if __name__ == "__main__":
    main()