# Processing Q1 and Q3 files from 2017-2022

import polars as pl
import os
from datetime import datetime
import warnings
import gc
warnings.filterwarnings('ignore')

# Configuration
BASE_INPUT_DIR = "/Users/sme/Downloads/project/"  # Where parquet files are stored
SURVIVAL_OUTPUT_DIR = "/sme/sundargodina/Downloads/survival_files/"

# Ensure output directory exists
os.makedirs(SURVIVAL_OUTPUT_DIR, exist_ok=True)

def safe_string_contains(col_name, pattern):
    """Safely check if string column contains pattern, handling nulls"""
    return (
        pl.col(col_name).is_not_null() & 
        pl.col(col_name).cast(pl.Utf8).str.contains(pattern)
    )

def create_survival_dataset_memory_optimized(df, quarter_name):
    """Memory-optimized survival analysis dataset creation"""
    print(f"   Creating survival dataset from {len(df):,} rows for {quarter_name}")
    
    # Check required columns
    required_cols = ["LOAN_ID", "LOAN_AGE", "DLQ_STATUS"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"  Missing required columns: {missing_cols}")
        return None
    
    # Get basic info
    unique_loans = df['LOAN_ID'].n_unique()
    print(f"   Data: {unique_loans:,} unique loans")
    
    # Select only essential columns to reduce memory
    essential_cols = [
        "LOAN_ID", "LOAN_AGE", "DLQ_STATUS", "ORIG_DATE", "CSCORE_B", 
        "DTI", "ORIG_RATE", "ORIG_UPB", "ORIG_TERM", "OLTV", "OCLTV",
        "PURPOSE", "STATE", "PROP", "OCC_STAT"
    ]
    
    # Add optional columns if they exist
    optional_cols = ["FORECLOSURE_DATE", "Zero_Bal_Code", "CURRENT_UPB", "MATR_DT"]
    for col in optional_cols:
        if col in df.columns:
            essential_cols.append(col)
    
    # Filter to only essential columns
    available_cols = [col for col in essential_cols if col in df.columns]
    df_subset = df.select(available_cols)
    
    print(f"   Using {len(available_cols)} columns to reduce memory")
    
    # Force garbage collection
    del df
    gc.collect()
    
    try:
        # Get baseline info (first observation per loan)
        baseline_cols = [col for col in available_cols if col != "LOAN_AGE" or col == "LOAN_ID"]
        
        loan_baseline = (
            df_subset
            .sort(["LOAN_ID", "LOAN_AGE"])
            .group_by("LOAN_ID")
            .first()
            .select([col for col in baseline_cols if col in df_subset.columns])
        )
        
        print(f"   Baseline: {len(loan_baseline):,} unique loans")
        
    except Exception as e:
        print(f"  Error creating baseline: {e}")
        return None
    
    try:
        # Calculate events with minimal memory usage
        print("   Calculating default events...")
        
        # Basic aggregations only
        agg_list = [
            # Severe delinquencies
            pl.when(safe_string_contains("DLQ_STATUS", "3|4|5|6|7|8|9|X"))
            .then(1).otherwise(0).max().alias("ever_default_dlq"),
            
            pl.when(safe_string_contains("DLQ_STATUS", "3|4|5|6|7|8|9|X"))
            .then(pl.col("LOAN_AGE")).min().alias("first_default_age_dlq"),
            
            # Early delinquencies
            pl.when(safe_string_contains("DLQ_STATUS", "1|2"))
            .then(pl.col("LOAN_AGE")).min().alias("first_stage2_age"),
            
            # Observation info
            pl.col("LOAN_AGE").max().alias("last_observed_age"),
            pl.col("DLQ_STATUS").last().alias("final_dlq_status")
        ]
        
        # Add foreclosure if available
        if "FORECLOSURE_DATE" in df_subset.columns:
            agg_list.extend([
                pl.when(pl.col("FORECLOSURE_DATE").is_not_null())
                .then(1).otherwise(0).max().alias("ever_foreclosure"),
                pl.when(pl.col("FORECLOSURE_DATE").is_not_null())
                .then(pl.col("LOAN_AGE")).min().alias("first_foreclosure_age")
            ])
        else:
            agg_list.extend([
                pl.lit(0).alias("ever_foreclosure"),
                pl.lit(None).cast(pl.Int32).alias("first_foreclosure_age")
            ])
        
        # Add zero balance if available
        if "Zero_Bal_Code" in df_subset.columns:
            agg_list.append(
                pl.when(pl.col("Zero_Bal_Code").is_in(["03", "06", "09"]))
                .then(1).otherwise(0).max().alias("ever_default_zb")
            )
        else:
            agg_list.append(pl.lit(0).alias("ever_default_zb"))
        
        default_events = df_subset.group_by("LOAN_ID").agg(agg_list)
        
        # Clean up
        del df_subset
        gc.collect()
        
    except Exception as e:
        print(f" Error calculating events: {e}")
        return None
    
    try:
        # Create survival variables
        print("   Creating survival variables...")
        
        survival_data = (
            default_events
            .with_columns([
                # Main default event
                pl.when(
                    (pl.col("ever_default_dlq") == 1) |
                    (pl.col("ever_foreclosure") == 1) |
                    (pl.col("ever_default_zb") == 1)
                ).then(1).otherwise(0).alias("default_event"),
                
                # Time to default
                pl.coalesce([
                    pl.col("first_default_age_dlq"),
                    pl.col("first_foreclosure_age")
                ]).alias("time_to_default"),
                
                # Stage 2 events
                pl.when(pl.col("first_stage2_age").is_not_null())
                .then(1).otherwise(0).alias("stage2_event")
            ])
            .with_columns([
                # Survival times
                pl.when(pl.col("default_event") == 1)
                .then(pl.col("time_to_default"))
                .otherwise(pl.col("last_observed_age"))
                .alias("survival_time_raw"),
                
                pl.when(pl.col("stage2_event") == 1)
                .then(pl.col("first_stage2_age"))
                .otherwise(pl.col("last_observed_age"))
                .alias("stage2_survival_time_raw")
            ])
            .with_columns([
                # Ensure positive times
                pl.when(pl.col("survival_time_raw") <= 0)
                .then(pl.lit(1))
                .otherwise(pl.col("survival_time_raw"))
                .alias("survival_time"),
                
                pl.when(pl.col("stage2_survival_time_raw") <= 0)
                .then(pl.lit(1))
                .otherwise(pl.col("stage2_survival_time_raw"))
                .alias("stage2_survival_time"),
                
                pl.col("default_event").alias("survival_event"),
                pl.col("stage2_event").alias("stage2_survival_event")
            ])
        )
        
        # Clean up
        del default_events
        gc.collect()
        
    except Exception as e:
        print(f" Error creating survival variables: {e}")
        return None
    
    try:
        # Join and finalize
        print("   Finalizing dataset...")
        
        final_dataset = (
            loan_baseline
            .join(survival_data, on="LOAN_ID", how="inner")
            .with_columns([
                # IFRS 9 staging
                pl.when(pl.col("stage2_event") == 0)
                .then(pl.lit("Stage1"))
                .when((pl.col("stage2_event") == 1) & (pl.col("default_event") == 0))
                .then(pl.lit("Stage2"))
                .when(pl.col("default_event") == 1)
                .then(pl.lit("Stage3"))
                .otherwise(pl.lit("Stage1"))
                .alias("ifrs9_stage"),
                
                # 12-month indicators
                pl.when((pl.col("survival_time") <= 12) & (pl.col("survival_event") == 1))
                .then(1).otherwise(0).alias("default_12m"),
                
                pl.when((pl.col("stage2_survival_time") <= 12) & (pl.col("stage2_survival_event") == 1))
                .then(1).otherwise(0).alias("stage2_12m"),
                
                pl.lit(quarter_name).alias("quarter")
            ])
        )
        
        # Clean up
        del loan_baseline, survival_data
        gc.collect()
        
        if len(final_dataset) == 0:
            print(f"   No data in final dataset for {quarter_name}")
            return None
        
        # Quick stats
        total_loans = len(final_dataset)
        default_count = final_dataset['default_event'].sum() or 0
        
        print(f"  Final: {total_loans:,} loans, {default_count:,} defaults ({default_count/total_loans*100:.2f}%)")
        
        return final_dataset
        
    except Exception as e:
        print(f"  Error finalizing: {e}")
        return None
for year in range(2017, 2023):
    for q in ['Q1', 'Q3']:
        quarter = f"{year}{q}"
        parquet_path = os.path.join(BASE_INPUT_DIR, f"{quarter}.parquet")

        if not os.path.exists(parquet_path):
            print(f"  File not found: {parquet_path}")
            continue

        print(f"\nProcessing {quarter}...")

        try:
            # Load data
            print("   Loading data...")
            df = pl.read_parquet(parquet_path)
            print(f"   Loaded {len(df):,} rows")

            # Create survival dataset
            survival_df = create_survival_dataset_memory_optimized(df, quarter)

            if survival_df is None:
                print(f"   ⚠  Skipping {quarter} due to invalid result")
            else:
                # Save the file
                survival_output_path = os.path.join(SURVIVAL_OUTPUT_DIR, f"survival_{quarter}.parquet")
                survival_df.write_parquet(survival_output_path, compression="snappy")
                print(f"   💾 Saved: survival_{quarter}.parquet")

                # Clean up
                del survival_df
                gc.collect()

                print(f"Completed {quarter}")

        except Exception as e:
            print(f"    Error processing {quarter}: {str(e)}")
            gc.collect()
