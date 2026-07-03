import pandas as pd
import numpy as np

# =========================
# File names
# =========================

sample_file = "NPO_sample.xlsx"
mapping_file = "mapping_table.xlsx"

df = pd.read_excel(sample_file)
mapping_raw = pd.read_excel(mapping_file, header=None)


# =========================
# 1. Calculate financial factors
# =========================

def safe_divide(numerator, denominator):
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return np.nan
    return numerator / denominator


def calculate_financial_factors(data):
    data = data.copy()

    data["Total_Cash_Investments"] = (
        data["Cash"]
        + data["Savings_and_Temporary_Cash"]
        + data["Investments"]
    ) / 1_000_000

    data["EFR_Total_Liabilities"] = data.apply(
        lambda row: safe_divide(
            row["Unrestricted_Net_Assets"]
            + row["Temporarily_Restricted_Net_Assets"]
            - row["PP&E"],
            row["Total_Liabilities"]
        ),
        axis=1
    )

    data["Free_EFR_Operations"] = data.apply(
        lambda row: safe_divide(
            row["Unrestricted_Net_Assets"]
            + row["Temporarily_Restricted_Net_Assets"]
            - row["PP&E"]
            - row["Total_Liabilities"],
            row["Total_Expenses"]
        ),
        axis=1
    )

    data["Gifts"] = data.apply(
        lambda row: safe_divide(
            abs(row["Gifts_Contributions_Grants"]),
            row["Total_Revenues"]
        ),
        axis=1
    )

    return data


# =========================
# 2. Build mapping tables
# =========================

def build_mapping_tables(mapping_raw):
    positive_map = mapping_raw.iloc[1:17, [0, 1, 2, 3]].copy()
    positive_map.columns = [
        "Total_Cash_Investments",
        "EFR_Total_Liabilities",
        "Free_EFR_Operations",
        "Rating"
    ]

    gift_map = mapping_raw.iloc[1:17, [5, 6]].copy()
    gift_map.columns = ["Gifts", "Rating"]

    for col in positive_map.columns:
        positive_map[col] = pd.to_numeric(positive_map[col], errors="coerce")

    for col in gift_map.columns:
        gift_map[col] = pd.to_numeric(gift_map[col], errors="coerce")

    positive_map = positive_map.dropna()
    gift_map = gift_map.dropna()

    return positive_map, gift_map


positive_map, gift_map = build_mapping_tables(mapping_raw)


# =========================
# 3. Map factors to ratings
# =========================

def map_positive_factor(value, factor_name):
    if pd.isna(value):
        return np.nan

    factor_table = positive_map[[factor_name, "Rating"]].dropna()
    factor_table = factor_table.sort_values(factor_name)

    rating = factor_table.iloc[0]["Rating"]

    for _, row in factor_table.iterrows():
        if value >= row[factor_name]:
            rating = row["Rating"]

    return int(rating)


def map_gifts(value):
    if pd.isna(value):
        return np.nan

    value_percent = value * 100

    gift_table = gift_map.sort_values("Gifts")

    for _, row in gift_table.iterrows():
        if value_percent <= row["Gifts"]:
            return int(row["Rating"])

    return int(gift_table.iloc[-1]["Rating"])


def map_factor(value, factor_name):
    if factor_name == "Gifts":
        return map_gifts(value)

    return map_positive_factor(value, factor_name)


# =========================
# 4. Missing-value weights
# =========================

base_weights = {
    "Total_Cash_Investments": 0.50,
    "EFR_Total_Liabilities": 0.20,
    "Free_EFR_Operations": 0.20,
    "Gifts": 0.10
}


def get_adjusted_weights(row):
    available = {
        factor: not pd.isna(row[factor])
        for factor in base_weights
    }

    total_available_weight = sum(
        base_weights[factor]
        for factor in base_weights
        if available[factor]
    )

    if total_available_weight == 0:
        return {factor: 0 for factor in base_weights}

    return {
        factor: base_weights[factor] / total_available_weight
        if available[factor] else 0
        for factor in base_weights
    }


# =========================
# 5. Calculate quantitative rating
# =========================

def round_half_up(x):
    return int(np.floor(x + 0.5))


def calculate_quant_rating(row):
    factor_scores = {
        "Total_Cash_Investments": map_factor(
            row["Total_Cash_Investments"],
            "Total_Cash_Investments"
        ),
        "EFR_Total_Liabilities": map_factor(
            row["EFR_Total_Liabilities"],
            "EFR_Total_Liabilities"
        ),
        "Free_EFR_Operations": map_factor(
            row["Free_EFR_Operations"],
            "Free_EFR_Operations"
        ),
        "Gifts": map_factor(
            row["Gifts"],
            "Gifts"
        )
    }

    missing_count = sum(pd.isna(score) for score in factor_scores.values())

    if missing_count == 4:
        return 16

    weights = get_adjusted_weights(row)

    weighted_score = sum(
        factor_scores[factor] * weights[factor]
        for factor in factor_scores
        if not pd.isna(factor_scores[factor])
    )

    weighted_score += missing_count

    rating = round_half_up(weighted_score)

    return min(max(rating, 1), 16)


# =========================
# 6. Base rating calculation
# =========================

df = calculate_financial_factors(df)

df["Quantitative_Rating"] = df.apply(
    lambda row: calculate_quant_rating(row),
    axis=1
)


# =========================
# 7. One-way sensitivity analysis
# =========================

shock_levels = [
    -0.50, -0.30, -0.20, -0.10, -0.05,
     0.05,  0.10,  0.20,  0.30,  0.50
]

raw_inputs_to_shock = [
    "Cash",
    "Savings_and_Temporary_Cash",
    "Investments",
    "Unrestricted_Net_Assets",
    "Temporarily_Restricted_Net_Assets",
    "PP&E",
    "Total_Liabilities",
    "Total_Expenses",
    "Gifts_Contributions_Grants",
    "Total_Revenues"
]

sensitivity_results = []

for input_col in raw_inputs_to_shock:
    for shock in shock_levels:
        shocked_df = df.copy()

        shocked_df[input_col] = shocked_df[input_col] * (1 + shock)

        shocked_df = calculate_financial_factors(shocked_df)

        shocked_df["Shocked_Rating"] = shocked_df.apply(
            lambda row: calculate_quant_rating(row),
            axis=1
        )

        shocked_df["Rating_Change"] = (
            shocked_df["Shocked_Rating"] - df["Quantitative_Rating"]
        )

        sensitivity_results.append({
            "Input_Shocked": input_col,
            "Shock": shock,
            "Average_Absolute_Notch_Change": shocked_df["Rating_Change"].abs().mean(),
            "Max_Notch_Change": shocked_df["Rating_Change"].abs().max(),
            "Percent_Unchanged": (shocked_df["Rating_Change"] == 0).mean(),
            "Percent_Moved_1_Notch": (shocked_df["Rating_Change"].abs() == 1).mean(),
            "Percent_Moved_2_or_More": (shocked_df["Rating_Change"].abs() >= 2).mean()
        })

sensitivity_summary = pd.DataFrame(sensitivity_results)


# =========================
# 8. Export results
# =========================

with pd.ExcelWriter("NPO_Sensitivity_Analysis_Output.xlsx") as writer:
    df.to_excel(writer, sheet_name="Base Rating Calculation", index=False)
    positive_map.to_excel(writer, sheet_name="Positive Mapping", index=False)
    gift_map.to_excel(writer, sheet_name="Gift Mapping", index=False)
    sensitivity_summary.to_excel(writer, sheet_name="Sensitivity Summary", index=False)

print("Done. Output saved as NPO_Sensitivity_Analysis_Output.xlsx")
