# menu_category_analysis.py

import os
from dotenv import load_dotenv
from transformers import pipeline
from db import SessionLocal, RestaurantOrder
from collections import defaultdict

load_dotenv()

# Load zero-shot classifier from Hugging Face
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

CATEGORIES = ["Drink", "Appetizer", "Main", "Dessert"]

def categorize_menu_items_hf():
    session = SessionLocal()
    orders = session.query(RestaurantOrder).all()
    session.close()

    item_totals = defaultdict(int)
    for order in orders:
        item_totals[order.menu_item] += order.quantity

    categorized_data = defaultdict(lambda: defaultdict(int))

    for item, quantity in item_totals.items():
        prediction = classifier(item, CATEGORIES)
        best_category = prediction["labels"][0]
        categorized_data[best_category][item] += quantity

    return categorized_data
