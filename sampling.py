#code for sampling the main data

import polars as pl
import random
from collections import defaultdict


# File paths
FILE_PATH = "abfss://raw@survivalproject.dfs.core.windows.net/final_loan_macro_polars.parquet"
OUTPUT_PATH = "abfss://raw@survivalproject.dfs.core.windows.net/final_loan_macro_sampled.parquet"

# Sampling parameters
random.seed(42)  # For reproducibility
TARGET_LOANS = 150000  # Target sample size
ORIGINATION_YEARS = [2017, 2018, 2019]  # Loans originated in these years

print("Starting stratified sampling for survival analysis...")
print(f"Target sample size: {TARGET_LOANS:,} loans")

# Target observation period (loans active during this time)
observation_quarters = [f"{y}Q{q}" for y in ORIGINATION_YEARS for q in range(1, 5)]
print(f"Sampling from loans active during: {observation_quarters}")

# Lazy read the full dataset
lf = pl.scan_parquet(FILE_PATH)

# Get loan characteristics for ALL loans active during target period
# This includes loans originated before 2017 that are still active
loan_chars = (lf
    .filter(pl.col("data_quarter").is_in(observation_quarters))
    .group_by("LOAN_ID")
    .agg([
        pl.col("ORIG_DATE").first().alias("actual_orig_date"),  # Actual origination date
        pl.col("data_quarter").first().alias("first_observed"), # First quarter we see them
        pl.col("CSCORE_B").first().alias("credit_score"),       # Borrower Credit Score at Origination
        pl.col("OLTV").first().alias("ltv"),                    # Original LTV
        pl.col("STATE").first().alias("state"),
        pl.col("PURPOSE").first().alias("loan_purpose"),        # Loan Purpose
        pl.col("event").max().alias("ever_defaults")            # Default event flag
    ])
    .collect())

print(f"Found {loan_chars.height:,} unique loans active during 2017-2019")

loan_chars = loan_chars.with_columns([
    # Extract actual origination year from ORIG_DATE
    pl.col("actual_orig_date").dt.year().alias("orig_year"),
    
    # Credit score tiers (using credit_score alias from CSCORE_B)
    pl.when(pl.col("credit_score") < 620).then(pl.lit("VeryLow"))
     .when(pl.col("credit_score") < 680).then(pl.lit("Low"))
     .when(pl.col("credit_score") < 740).then(pl.lit("Mid"))
     .when(pl.col("credit_score") < 800).then(pl.lit("High"))
     .otherwise(pl.lit("VeryHigh")).alias("credit_tier"),
    
    # LTV tiers (using ltv alias from OLTV)
    pl.when(pl.col("ltv") < 70).then(pl.lit("LowLTV"))
     .when(pl.col("ltv") < 85).then(pl.lit("MidLTV"))
     .otherwise(pl.lit("HighLTV")).alias("ltv_tier"),
    
    # Geographic regions (group states for adequate sample sizes)
    pl.when(pl.col("state").is_in(["CA", "NY", "FL", "TX"])).then(pl.lit("Major"))
     .when(pl.col("state").is_in(["WA", "OR", "NV", "AZ", "CO", "IL", "MI", "OH", "PA", "NJ", "VA", "NC", "GA"])).then(pl.lit("Large"))
     .otherwise(pl.lit("Other")).alias("geo_tier")
])

# Group loans by strata
strata_groups = defaultdict(list)
for row in loan_chars.iter_rows(named=True):
    # Create stratum key: (orig_year, credit_tier, ltv_tier, ever_defaults, geo_tier)
    stratum = (
        row["orig_year"], 
        row["credit_tier"], 
        row["ltv_tier"], 
        row["ever_defaults"],
        row["geo_tier"]
    )
    strata_groups[stratum].append(row["LOAN_ID"])

print(f"\nCreated {len(strata_groups)} strata")

# Sample proportionally from each stratum
sampled_loan_ids = []
total_available_loans = sum(len(loans) for loans in strata_groups.values())

print("\nSampling by stratum:")
print("Stratum (Year, Credit, LTV, Default, Geo) -> Sample Size")
print("-" * 60)

for stratum, loan_ids in strata_groups.items():
    stratum_prop = len(loan_ids) / total_available_loans
    stratum_target = max(1, int(TARGET_LOANS * stratum_prop))
    stratum_actual = min(stratum_target, len(loan_ids))
    
    stratum_sample = random.sample(loan_ids, stratum_actual)
    sampled_loan_ids.extend(stratum_sample)
    
    print(f"{stratum} -> {stratum_actual:,} loans ({stratum_prop:.1%} of population)")

print(f"\nTotal loans sampled: {len(sampled_loan_ids):,}")


print("\nExtracting complete time series for sampled loans...")

# Get ALL data for sampled loans (including full observation period through 2022)
lf_sampled = lf.filter(pl.col("LOAN_ID").is_in(sampled_loan_ids))
result = lf_sampled.collect()


print(f"\nFINAL DATASET SUMMARY:")
print(f"Rows: {result.shape[0]:,}")
print(f"Loans: {result['LOAN_ID'].n_unique():,}")
print(f"Time periods: {result['data_quarter'].unique().sort()}")

# Check default rate preservation
sample_default_rate = (result
    .group_by("LOAN_ID")
    .agg(pl.col("event").max())
    .filter(pl.col("event") == 1)
    .height / result["LOAN_ID"].n_unique())

print(f"Default rate in sample: {sample_default_rate:.3%}")

orig_dist = (result
    .group_by("LOAN_ID")
    .agg(pl.col("ORIG_DATE").first())
    .with_columns(pl.col("ORIG_DATE").dt.year().alias("orig_year"))
    .group_by("orig_year")
    .agg(pl.count().alias("count"))
    .sort("orig_year"))

print(f"\nOrigination year distribution (actual ORIG_DATE):")
for row in orig_dist.iter_rows(named=True):
    print(f"  {row['orig_year']}: {row['count']:,} loans")


result.write_parquet(OUTPUT_PATH, compression="snappy")

print(f"\nStratified sample saved to: {OUTPUT_PATH}")
print(f"Sample represents {len(sampled_loan_ids)/8_700_000:.2%} of total loan population")

# Check credit score distribution
credit_dist = (result
    .group_by("LOAN_ID")
    .agg(pl.col("CSCORE_B").first())
    .describe())

print(f"\nCredit Score Distribution in Sample:")
print(credit_dist)

# Check geographic coverage
state_count = result.select("STATE").n_unique()
print(f"\nGeographic Coverage: {state_count} unique states")

# Check time series completeness
time_coverage = result.select("data_quarter").unique().sort("data_quarter")
print(f"\nTime Coverage: {time_coverage.height} quarters")
print(f"From {time_coverage.item(0, 0)} to {time_coverage.item(-1, 0)}")
