## Generate Dataset
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

np.random.seed(42)

food_items = [
    "veg biryani", "chicken biryani", "paneer butter masala",
    "masala dosa", "burger", "pizza", "fried rice",
    "tandoori chicken", "naan", "pasta"
]

def simulate_order():
    items = random.sample(food_items, random.randint(1, 3))
    hour = random.randint(0, 23)
    rain = np.random.choice([0, 1], p=[0.8, 0.2])
    event = np.random.choice([0, 1], p=[0.9, 0.1])
    kitchen_load = np.random.randint(1, 20)

    base_time = 10 + len(items)*5 + kitchen_load*0.5
    if hour in [12,13,20,21]:
        base_time += 5
    if rain:
        base_time += 3
    if event:
        base_time += 7

    prep_time = base_time + np.random.normal(0, 2)

    return {
        "items": ", ".join(items),
        "hour": hour,
        "rain": rain,
        "event": event,
        "kitchen_load": kitchen_load,
        "prep_time": max(5, round(prep_time,1))
    }

data = [simulate_order() for _ in range(5000)]
df = pd.DataFrame(data)
df.to_csv("food_prep_dataset.csv", index=False)

print("Dataset created.")

print(df)