"""
Generate synthetic Levels.fyi-style salary dataset.
Run once to create data/salaries.csv
"""
import numpy as np
import pandas as pd
import os

np.random.seed(42)
os.makedirs("data", exist_ok=True)

N = 5000

companies = {
    "FAANG": ["Google", "Meta", "Amazon", "Apple", "Netflix"],
    "Tier2": ["Microsoft", "Salesforce", "Uber", "Lyft", "Airbnb", "Stripe", "Databricks"],
    "Tier3": ["Oracle", "IBM", "SAP", "Intuit", "Workday", "ServiceNow"],
    "Startup": ["Series A Startup", "Series B Startup", "Early Stage Startup"],
}

roles = ["Software Engineer", "Senior Software Engineer", "Staff Engineer",
         "Principal Engineer", "Engineering Manager", "Data Scientist",
         "Senior Data Scientist", "ML Engineer", "Product Manager",
         "Senior Product Manager"]

locations = {
    "San Francisco, CA": 1.45,
    "Seattle, WA": 1.30,
    "New York, NY": 1.25,
    "Austin, TX": 1.00,
    "Remote": 1.05,
    "Boston, MA": 1.15,
    "Los Angeles, CA": 1.20,
    "Denver, CO": 0.95,
    "Chicago, IL": 0.98,
    "Atlanta, GA": 0.90,
}

education = ["Bachelor's", "Master's", "PhD", "Bootcamp", "Self-taught"]

def company_tier(company):
    for tier, companies_list in companies.items():
        if company in companies_list:
            return tier
    return "Tier3"

def tier_multiplier(tier):
    return {"FAANG": 1.5, "Tier2": 1.2, "Tier3": 1.0, "Startup": 0.9}[tier]

def role_base(role):
    base_map = {
        "Software Engineer": 120000,
        "Senior Software Engineer": 165000,
        "Staff Engineer": 220000,
        "Principal Engineer": 280000,
        "Engineering Manager": 230000,
        "Data Scientist": 130000,
        "Senior Data Scientist": 175000,
        "ML Engineer": 155000,
        "Product Manager": 140000,
        "Senior Product Manager": 185000,
    }
    return base_map.get(role, 130000)

def yoe_bonus(yoe, role):
    if "Senior" in role or "Staff" in role or "Principal" in role:
        return yoe * 4000
    return yoe * 5500

records = []
all_companies = [c for tier in companies.values() for c in tier]

for _ in range(N):
    company = np.random.choice(all_companies, p=None)
    role = np.random.choice(roles)
    location = np.random.choice(list(locations.keys()))
    edu = np.random.choice(education, p=[0.50, 0.25, 0.08, 0.10, 0.07])
    yoe = max(0, int(np.random.exponential(5)))
    yoe = min(yoe, 25)

    base = role_base(role)
    loc_mult = locations[location]
    tier_mult = tier_multiplier(company_tier(company))
    edu_bonus = {"Bachelor's": 0, "Master's": 8000, "PhD": 15000,
                 "Bootcamp": -10000, "Self-taught": -5000}[edu]

    # Base salary
    salary = (base + yoe_bonus(yoe, role) + edu_bonus) * loc_mult * tier_mult
    salary += np.random.normal(0, salary * 0.08)  # noise
    salary = max(50000, int(salary))

    # Stock (RSU)
    stock = salary * np.random.uniform(0.1, 0.6) if company_tier(company) in ["FAANG", "Tier2"] else 0
    stock = int(max(0, stock + np.random.normal(0, stock * 0.2)) if stock > 0 else 0)

    # Bonus
    bonus = salary * np.random.uniform(0.05, 0.20)
    bonus = int(max(0, bonus + np.random.normal(0, bonus * 0.15)))

    total_comp = salary + stock // 4 + bonus  # annualized stock

    records.append({
        "company": company,
        "role": role,
        "location": location,
        "education": edu,
        "years_of_experience": yoe,
        "base_salary": salary,
        "annual_bonus": bonus,
        "annual_stock": stock // 4,
        "total_compensation": total_comp,
        "company_tier": company_tier(company),
        "remote_work": np.random.choice([0, 1], p=[0.6, 0.4]),
    })

df = pd.DataFrame(records)
df.to_csv("data/salaries.csv", index=False)
print(f"✅ Generated {len(df)} records → data/salaries.csv")
print(df.describe())