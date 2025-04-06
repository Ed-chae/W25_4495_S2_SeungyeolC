import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from db import SessionLocal, RestaurantOrder

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def categorize_menu_items_ai():
    session = SessionLocal()
    items = session.query(RestaurantOrder.menu_item).distinct().all()
    session.close()

    unique_items = sorted(set(i[0] for i in items if i[0]))

    prompt = (
        "Categorize each of the following menu items into either 'drink', 'appetizer', or 'main'. "
        "Return the result as a JSON object with each item as a key and the category as value.\n\n"
        "Menu Items:\n" + "\n".join(unique_items)
    )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        content = response.choices[0].message.content.strip()

        # Try to extract JSON safely
        try:
            category_map = json.loads(content)
        except json.JSONDecodeError:
            # Attempt to extract JSON from inside the response
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            category_map = json.loads(content[json_start:json_end])

        category_counts = {"drink": 0, "appetizer": 0, "main": 0}
        for cat in category_map.values():
            normalized = cat.strip().lower()
            if normalized in category_counts:
                category_counts[normalized] += 1

        return {"categories": category_counts, "raw_mapping": category_map}

    except Exception as e:
        return {"categories": {}, "error": str(e)}
    
# ----------------------------------
# 📊 Get Category Counts
# ----------------------------------
def get_menu_category_distribution():
    session = SessionLocal()
    orders = session.query(RestaurantOrder).filter(RestaurantOrder.menu_item != None).all()
    session.close()

    items = list(set([o.menu_item.strip() for o in orders if o.menu_item]))

    if not items:
        return {"categories": [], "message": "No menu items found."}

    item_category_map = categorize_menu_items_with_gpt(items)
    category_counts = Counter(item_category_map.values())

    return {
        "categories": [{"label": cat, "value": count} for cat, count in category_counts.items()],
        "message": "✅ Menu item categories generated using GPT."
    }
