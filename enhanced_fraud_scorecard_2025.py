
import pandas as pd

# === Load transaction data (including engineered features) ===
df = pd.read_excel("transactions_2025_engineered.xlsx")

# === Country Risk Scores ===
country_risk = {
    "USA": 5, "UK": 5, "Argentina": 10, "Brazil": 20, "Nigeria": 30, "Russia": 30,
    "CN": 25, "IR": 30, "NG": 30, "KP": 30  # Extend as needed
}

# === Weight Dictionary for Modular KIs ===
ki_weights = {
    "KI01_amt_deviation": 20,
    "KI02_night_txn": 10,
    "KI03_new_payee_high_amt": 30,
    "KI04_burst_txns": 25,
    "KI05_new_foreign_country": 40,  # New: detailed scoring based on country
    "KI06_new_acct_high_txn": 25,
    "KI07_large_internal_transfer_no_dualauth": 30,
    "KI08_payee_risk_low_history": 15
}

# === Feature Engineering for Specific KIs ===

# KI01 - Amount Deviation
df['KI01_amt_deviation'] = ((df['REQUESTED_AMOUNT_US_INTL'] - df['account_mean_amt']).abs() > 2 * df['account_std_amt']).astype(int)

# KI02 - Night Transaction
df['KI02_night_txn'] = df['hour'].isin([0, 1, 2, 3, 4, 5]).astype(int)

# KI03 - New Payee + High Amount
df['KI03_new_payee_high_amt'] = ((df['is_new_party']) & (df['REQUESTED_AMOUNT_US_INTL'] > 10000)).astype(int)

# KI04 - Burst Transactions (txn_count_24h > 4)
df['KI04_burst_txns'] = (df['txn_count_24h'] > 4).astype(int)

# KI05 - New Foreign Country Logic (dynamic scoring)
def compute_ki05_score(row):
    prev_countries = set(str(row.get("prev_foreign_countries", "")).split(","))
    curr_countries = set(str(row.get("curr_foreign_countries", "")).split(","))
    new_countries = curr_countries - prev_countries
    scaling = 1.2 if len(prev_countries) <= 2 else 1.0 if len(prev_countries) <= 5 else 0.5
    score = sum([country_risk.get(c.strip(), 0) for c in new_countries]) * scaling
    return score

df['KI05_new_foreign_country_score'] = df.apply(compute_ki05_score, axis=1)
df['KI05_new_foreign_country'] = (df['KI05_new_foreign_country_score'] > 0).astype(int)

# KI06 - New Account + High Activity
df['KI06_new_acct_high_txn'] = ((df['acct_creation_days_ago'] < 5) & (df['txn_count_24h'] > 3)).astype(int)

# KI07 - Large Internal Transfer with No Dual Approval
df['KI07_large_internal_transfer_no_dualauth'] = ((df['REQUESTED_AMOUNT_US_INTL'] > 100000) & (~df['approval_flag'])).astype(int)

# KI08 - Low Frequency Payee + High Amount
df['KI08_payee_risk_low_history'] = ((df['payee_usage_count'] < 2) & (df['REQUESTED_AMOUNT_US_INTL'] > 7000)).astype(int)

# === Score Aggregation ===
df['raw_score'] = 0
for ki, weight in ki_weights.items():
    df['raw_score'] += df[ki] * weight

# Add dynamic score from KI05 (foreign country logic)
df['raw_score'] += df['KI05_new_foreign_country_score']

# Normalize score to 0-100
df['normalized_score'] = (df['raw_score'].clip(0, 260) / 260) * 100

# Risk Band
def band(score):
    if score >= 71:
        return "High Risk"
    elif score >= 41:
        return "Medium Risk"
    else:
        return "Low Risk"

df['risk_band'] = df['normalized_score'].apply(band)

# === Export ===
df.to_excel("scored_transactions_2025_enhanced.xlsx", index=False)
print("Enhanced scoring complete. Saved to scored_transactions_2025_enhanced.xlsx.")
