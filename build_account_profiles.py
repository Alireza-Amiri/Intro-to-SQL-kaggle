| Test                          | What changes?                  | Purpose                                             | Expected Result                                                                       |
| ----------------------------- | ------------------------------ | --------------------------------------------------- | ------------------------------------------------------------------------------------- |
| **Base Rating**               | Nothing                        | Establish benchmark ratings                         | Matches the production model exactly                                                  |
| **One-Way Sensitivity**       | One financial factor at a time | Assess robustness to input perturbations            | Most ratings unchanged or change by at most one notch for moderate shocks             |
| **Missing Value Sensitivity** | Remove one input factor        | Verify re-weighting and missing-value penalty logic | Downgrades are generally limited and consistent with the documented one-notch penalty |
| **Weight Sensitivity**        | Modify factor weights          | Assess dependence on expert-selected weights        | Ratings remain broadly stable under reasonable alternative weight schemes             |

    
import pandas as pd
import numpy as np

# =========================
# 1. Load Excel Files
# =========================

sample_file = "NPO_sample.xlsx"
mapping_file = "mapping_table.xlsx"
weight_file = "weight_table.xlsx"

df = pd.read_excel(sample_file)
mapping = pd.read_excel(mapping_file)
weights = pd.read_excel(weight_file)

# Expected columns in df:
# Borrower_ID
# Total_Cash_Investments
# EFR_Total_Liabilities
# Free_EFR_Operations
# Gifts
# Original_Rating


# =========================
# 2. Mapping Function
# =========================

def map_factor(value, factor_name, mapping_df):
    """
    Converts financial factor value into rating 1-16.
    Assumes mapping table has:
    Factor, Threshold, Rating
    """
    factor_map = mapping_df[mapping_df["Factor"] == factor_name].copy()
    factor_map = factor_map.sort_values("Threshold")

    if pd.isna(value):
        return np.nan

    matched = factor_map[factor_map["Threshold"] <= value]

    if matched.empty:
        return factor_map.iloc[0]["Rating"]

    return matched.iloc[-1]["Rating"]


# =========================
# 3. Weight Selection
# =========================

def get_weights(row):
    """
    Assigns weights depending on missing factors.
    If all values are present, uses normal weights:
    Cash = 50%, EFR Liabilities = 20%, Free EFR Ops = 20%, Gifts = 10%
    """
    available = {
        "Total_Cash_Investments": not pd.isna(row["Total_Cash_Investments"]),
        "EFR_Total_Liabilities": not pd.isna(row["EFR_Total_Liabilities"]),
        "Free_EFR_Operations": not pd.isna(row["Free_EFR_Operations"]),
        "Gifts": not pd.isna(row["Gifts"])
    }

    base_weights = {
        "Total_Cash_Investments": 0.50,
        "EFR_Total_Liabilities": 0.20,
        "Free_EFR_Operations": 0.20,
        "Gifts": 0.10
    }

    total_available_weight = sum(
        base_weights[k] for k, v in available.items() if v
    )

    if total_available_weight == 0:
        return {k: 0 for k in base_weights}

    adjusted_weights = {
        k: base_weights[k] / total_available_weight if available[k] else 0
        for k in base_weights
    }

    return adjusted_weights


# =========================
# 4. Rating Calculation
# =========================

def calculate_rating(row, mapping_df):
    factor_scores = {
        "Total_Cash_Investments": map_factor(
            row["Total_Cash_Investments"],
            "Total_Cash_Investments",
            mapping_df
        ),
        "EFR_Total_Liabilities": map_factor(
            row["EFR_Total_Liabilities"],
            "EFR_Total_Liabilities",
            mapping_df
        ),
        "Free_EFR_Operations": map_factor(
            row["Free_EFR_Operations"],
            "Free_EFR_Operations",
            mapping_df
        ),
        "Gifts": map_factor(
            row["Gifts"],
            "Gifts",
            mapping_df
        )
    }

    missing_count = sum(pd.isna(v) for v in factor_scores.values())

    if missing_count == 4:
        return 16

    w = get_weights(row)

    weighted_score = sum(
        factor_scores[k] * w[k]
        for k in factor_scores
        if not pd.isna(factor_scores[k])
    )

    # One-notch penalty per missing factor
    weighted_score += missing_count

    # Round half up
    final_rating = int(np.floor(weighted_score + 0.5))

    return min(max(final_rating, 1), 16)


df["Base_Model_Rating"] = df.apply(
    lambda row: calculate_rating(row, mapping),
    axis=1
)


# =========================
# 5. One-Way Sensitivity
# =========================

shock_levels = [-0.50, -0.30, -0.20, -0.10, -0.05, 0.05, 0.10, 0.20, 0.30, 0.50]

factors = [
    "Total_Cash_Investments",
    "EFR_Total_Liabilities",
    "Free_EFR_Operations",
    "Gifts"
]

sensitivity_results = []

for factor in factors:
    for shock in shock_levels:
        shocked_df = df.copy()
        shocked_df[factor] = shocked_df[factor] * (1 + shock)

        shocked_df["Shocked_Rating"] = shocked_df.apply(
            lambda row: calculate_rating(row, mapping),
            axis=1
        )

        shocked_df["Rating_Change"] = (
            shocked_df["Shocked_Rating"] - shocked_df["Base_Model_Rating"]
        )

        sensitivity_results.append({
            "Factor": factor,
            "Shock": shock,
            "Average_Notch_Change": shocked_df["Rating_Change"].abs().mean(),
            "Max_Notch_Change": shocked_df["Rating_Change"].abs().max(),
            "%_Unchanged": (shocked_df["Rating_Change"] == 0).mean(),
            "%_Moved_1_Notch": (shocked_df["Rating_Change"].abs() == 1).mean(),
            "%_Moved_2_or_More": (shocked_df["Rating_Change"].abs() >= 2).mean()
        })

sensitivity_summary = pd.DataFrame(sensitivity_results)


# =========================
# 6. Missing Value Sensitivity
# =========================

missing_results = []

for factor in factors:
    temp_df = df.copy()
    temp_df[factor] = np.nan

    temp_df["Missing_Rating"] = temp_df.apply(
        lambda row: calculate_rating(row, mapping),
        axis=1
    )

    temp_df["Rating_Change"] = (
        temp_df["Missing_Rating"] - temp_df["Base_Model_Rating"]
    )

    missing_results.append({
        "Missing_Factor": factor,
        "Average_Notch_Change": temp_df["Rating_Change"].mean(),
        "Average_Absolute_Change": temp_df["Rating_Change"].abs().mean(),
        "Max_Notch_Change": temp_df["Rating_Change"].abs().max(),
        "%_Downgraded": (temp_df["Rating_Change"] > 0).mean(),
        "%_Moved_2_or_More": (temp_df["Rating_Change"].abs() >= 2).mean()
    })

missing_summary = pd.DataFrame(missing_results)


# =========================
# 7. Weight Sensitivity
# =========================

alternative_weights = {
    "Base_50_20_20_10": {
        "Total_Cash_Investments": 0.50,
        "EFR_Total_Liabilities": 0.20,
        "Free_EFR_Operations": 0.20,
        "Gifts": 0.10
    },
    "Equal_25_25_25_25": {
        "Total_Cash_Investments": 0.25,
        "EFR_Total_Liabilities": 0.25,
        "Free_EFR_Operations": 0.25,
        "Gifts": 0.25
    },
    "Lower_Cash_40_25_25_10": {
        "Total_Cash_Investments": 0.40,
        "EFR_Total_Liabilities": 0.25,
        "Free_EFR_Operations": 0.25,
        "Gifts": 0.10
    }
}


def calculate_rating_custom_weights(row, mapping_df, custom_weights):
    scores = {
        "Total_Cash_Investments": map_factor(row["Total_Cash_Investments"], "Total_Cash_Investments", mapping_df),
        "EFR_Total_Liabilities": map_factor(row["EFR_Total_Liabilities"], "EFR_Total_Liabilities", mapping_df),
        "Free_EFR_Operations": map_factor(row["Free_EFR_Operations"], "Free_EFR_Operations", mapping_df),
        "Gifts": map_factor(row["Gifts"], "Gifts", mapping_df)
    }

    weighted_score = sum(scores[k] * custom_weights[k] for k in scores)

    return int(np.floor(weighted_score + 0.5))


weight_results = []

for scenario, w in alternative_weights.items():
    df[scenario] = df.apply(
        lambda row: calculate_rating_custom_weights(row, mapping, w),
        axis=1
    )

    change = df[scenario] - df["Base_Model_Rating"]

    weight_results.append({
        "Scenario": scenario,
        "Average_Absolute_Notch_Change": change.abs().mean(),
        "Max_Notch_Change": change.abs().max(),
        "%_Unchanged": (change == 0).mean(),
        "%_Moved_2_or_More": (change.abs() >= 2).mean()
    })

weight_summary = pd.DataFrame(weight_results)


# =========================
# 8. Export Results
# =========================

with pd.ExcelWriter("NPO_Sensitivity_Analysis_Output.xlsx") as writer:
    df.to_excel(writer, sheet_name="Base Ratings", index=False)
    sensitivity_summary.to_excel(writer, sheet_name="One Way Sensitivity", index=False)
    missing_summary.to_excel(writer, sheet_name="Missing Value Test", index=False)
    weight_summary.to_excel(writer, sheet_name="Weight Sensitivity", index=False)

print("Sensitivity analysis completed.")
print("Output saved as: NPO_Sensitivity_Analysis_Output.xlsx")
