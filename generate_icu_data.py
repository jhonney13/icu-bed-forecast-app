import pandas as pd
import numpy as np

# Set the date range
date_range = pd.date_range(start="2024-01-01", end="2024-06-30")

# Simulate ICU bed occupancy data
np.random.seed(42)
occupancy = np.clip(np.random.normal(loc=8, scale=2, size=len(date_range)), 4, 12).astype(int)

# Create DataFrame
icu_data = pd.DataFrame({
    'date': date_range,
    'icu_occupancy': occupancy
})

# Save to CSV
icu_data.to_csv('icu_occupancy.csv', index=False)
print("✅ Synthetic ICU data saved as 'icu_occupancy.csv'")
