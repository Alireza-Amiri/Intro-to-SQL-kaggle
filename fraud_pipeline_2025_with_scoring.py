
import pandas as pd
import numpy as np
from datetime import timedelta

# === Load Profiles and Transactions ===
account_profiles = pd.read_csv("account_profiles_model.csv").set_index("ACCOUNT_KEY").to_dict(orient="index")
df = pd.read_excel("transactions_2025.xlsx")
df['TRANSACTION_LOCAL_DATE_TIME'] = pd.to_datetime(df['TRANSACTION_LOCAL_DATE_TIME'])
df.sort_values(by=['ACCOUNT_KEY', 'TRANSACTION_LOCAL_DATE_TIME'], inplace=True)

# === Initialize Context ===
account_context = {}
results = []

# === Weight Configuration for KIs ===
ki_weights = {
    "KI01": 2.0, "KI02": 1.5, "KI03": 2.0, "KI04": 2.0, "KI05": 1.5,
    "KI06": 2.0, "KI07": 1.5, "KI08": 1.5, "KI09": 1.5, "KI10": 1.5,
    "KI11": 2.0, "KI12": 2.0, "KI14": 2.0, "KI18": 1.0, "KI19": 2.0
}

# === Fraud Scoring Function ===
def compute_fraud_scores(df, ki_weights, normalization_method='minmax', min_score=0, max_score=20):
    df["raw_score"] = sum(df.get(ki, 0) * weight for ki, weight in ki_weights.items())
    if normalization_method == 'minmax':
        df["normalized_score"] = ((df["raw_score"] - min_score) / (max_score - min_score)).clip(0, 1) * 100
    elif normalization_method == 'logistic':
        df["normalized_score"] = (1 / (1 + np.exp(-0.1 * (df["raw_score"] - 5)))) * 100
    else:
        raise ValueError("Unsupported normalization method")

    def band(score):
        if score >= 71: return "High Risk"
        elif score >= 41: return "Medium Risk"
        else: return "Low Risk"

    df["risk_band"] = df["normalized_score"].apply(band)
    return df

# === Transaction Loop for KI Scoring ===
for _, txn in df.iterrows():
    acct = txn['ACCOUNT_KEY']
    now = txn['TRANSACTION_LOCAL_DATE_TIME']
    hour = now.hour
    party = txn['PARTY_KEY']
    amount = txn['REQUESTED_AMOUNT_US_INTL']
    channel = txn['CHANNEL']
    currency = txn['CURRENCY_CD']
    profile = account_profiles.get(acct, {'mean_amt': 0, 'std_amt': 1, 'typical_currency': currency})

    if acct not in account_context:
        account_context[acct] = {
            'last_txn_time': now - timedelta(hours=2),
            'parties_seen': set(),
            'txn_times': []
        }

    context = account_context[acct]
    kis = {}

    kis["KI01"] = int(abs(amount - profile['mean_amt']) > 2 * profile['std_amt'])
    kis["KI02"] = int(txn.get('activity_days_ago', 0) > 30)
    kis["KI03"] = int(party not in context['parties_seen'] and amount > 10000)
    kis["KI04"] = int(txn.get('txn_sum_24h', 0) > 50000)
    kis["KI05"] = int(txn.get('country_cd') in ['IR', 'RU', 'NG', 'CN'] and txn.get('is_first_foreign_txn', False))
    kis["KI06"] = int(txn.get('acct_creation_days_ago', 100) <= 3 and txn.get('txn_count_24h', 0) > 3)
    kis["KI07"] = int(txn.get('unique_payees_24h', 0) > 3)
    kis["KI08"] = int(txn.get('acct_creation_days_ago', 100) <= 7 and txn.get('txn_count_24h', 0) > 2)
    kis["KI09"] = int(channel.lower() in ['mobile', 'web'] and amount > 5000)
    kis["KI10"] = int(txn.get('payee_usage_count', 10) < 2 and amount > 5000)
    kis["KI11"] = int(txn.get('country_cd') in ['IR', 'RU', 'NG', 'CN', 'KP'])
    kis["KI12"] = int(txn.get('large_txns_2h', 0) > 2)
    kis["KI14"] = int(txn.get('is_known_name_new_account', False))
    kis["KI18"] = int(hour in [0, 1, 2, 3, 4, 5] or now.dayofweek >= 5)
    kis["KI19"] = int(amount > 100000 and not txn.get('approval_flag', True))

    result = {
        'ACCOUNT_KEY': acct,
        'PARTY_KEY': party,
        'REQUESTED_AMOUNT_US_INTL': amount,
        'TRANSACTION_LOCAL_DATE_TIME': now
    }
    result.update(kis)
    results.append(result)

    context['last_txn_time'] = now
    context['txn_times'].append(now)
    context['parties_seen'].add(party)

# === Finalize Scoring ===
results_df = pd.DataFrame(results)
results_df = compute_fraud_scores(results_df, ki_weights=ki_weights, normalization_method='minmax')

# === Export ===
results_df.to_excel("flagged_transactions_2025_scored.xlsx", index=False)
print("Fraud scoring pipeline completed. Results saved to 'flagged_transactions_2025_scored.xlsx'")
