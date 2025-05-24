
import pandas as pd

# Load 2024 transaction data
df_2024 = pd.read_excel("transactions_2024.xlsx")
df_2024['REQUESTED_AMOUNT_US_INTL'] = pd.to_numeric(df_2024['REQUESTED_AMOUNT_US_INTL'], errors='coerce')

# Create account-level profiles
profiles = df_2024.groupby("ACCOUNT_KEY").agg({
    "REQUESTED_AMOUNT_US_INTL": ["mean", "std"],
    "CURRENCY_CD": lambda x: x.mode().iloc[0] if not x.mode().empty else "USD"
}).reset_index()

# Rename columns
profiles.columns = ["ACCOUNT_KEY", "mean_amt", "std_amt", "typical_currency"]

# Save the profile model
profiles.to_csv("account_profiles_model.csv", index=False)
print("Profile generation complete. Saved to 'account_profiles_model.csv'")
