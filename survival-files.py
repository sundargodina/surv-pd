# Enhanced Processing with Null Handling for Cox Regression
import polars as pl
import os
from datetime import datetime
import warnings
import gc
import psutil
import numpy as np
warnings.filterwarnings('ignore')

# Configuration for MacBook Air 2020
BASE_INPUT_DIR = "/Users/sundargodina/Downloads/project/"
SURVIVAL_OUTPUT_DIR = "/Users/sundargodina/Downloads/survival_files/"
BATCH_SIZE = 50000
MAX_MEMORY_GB = 6

# Ensure output directory exists
os.makedirs(SURVIVAL_OUTPUT_DIR, exist_ok=True)

def check_memory_usage():
    """Monitor memory usage"""
    memory = psutil.virtual_memory()
    used_gb = memory.used / (1024**3)
    available_gb = memory.available / (1024**3)
    return used_gb, available_gb

def safe_string_contains(col_name, pattern):
    """Safely check if string column contains pattern, handling nulls"""
    return (
        pl.col(col_name).is_not_null() & 
        pl.col(col_name).cast(pl.Utf8).str.contains(pattern)
    )

def handle_missing_values(df_lazy, cols_to_use):
    """
    Handle missing values with appropriate strategies for Cox regression
    """
    print("   Applying null handling strategies...")
    
    # Define imputation strategies
    numeric_cols = ["CSCORE_B", "DTI", "ORIG_RATE", "ORIG_UPB", "ORIG_TERM", 
                   "OLTV", "OCLTV", "CURRENT_UPB", "LOAN_AGE"]
    categorical_cols = ["PURPOSE", "STATE", "PROP", "OCC_STAT", "DLQ_STATUS"]
    
    # Filter to only existing columns
    numeric_cols = [col for col in numeric_cols if col in cols_to_use]
    categorical_cols = [col for col in categorical_cols if col in cols_to_use]
    
    # Build imputation expressions
    imputation_exprs = []
    
    # Numeric columns - use median imputation
    for col in numeric_cols:
        if col == "LOAN_AGE":
            # LOAN_AGE should not be null for survival analysis
            imputation_exprs.append(
                pl.when(pl.col(col).is_null())
                .then(pl.lit(1))  # Default to 1 if somehow null
                .otherwise(pl.col(col))
                .alias(col)
            )
        elif col in ["CSCORE_B"]:
            # Credit scores - use median or a reasonable default
            imputation_exprs.append(
                pl.when(pl.col(col).is_null())
                .then(pl.lit(620))  # Reasonable default credit score
                .otherwise(pl.col(col))
                .alias(col)
            )
        elif col in ["DTI"]:
            # DTI - use median or reasonable default
            imputation_exprs.append(
                pl.when(pl.col(col).is_null())
                .then(pl.lit(35))  # Reasonable default DTI
                .otherwise(pl.col(col))
                .alias(col)
            )
        elif col in ["OLTV", "OCLTV"]:
            # LTV ratios - use median or reasonable default
            imputation_exprs.append(
                pl.when(pl.col(col).is_null())
                .then(pl.lit(80))  # Reasonable default LTV
                .otherwise(pl.col(col))
                .alias(col)
            )
        else:
            # For other numeric columns, use median within the dataset
            imputation_exprs.append(
                pl.col(col).fill_null(strategy="mean").alias(col)
            )
    
    # Categorical columns - use mode or 'Unknown' category
    for col in categorical_cols:
        if col == "DLQ_STATUS":
            # DLQ_STATUS is critical - use 'C' (current) as default
            imputation_exprs.append(
                pl.when(pl.col(col).is_null())
                .then(pl.lit("C"))
                .otherwise(pl.col(col))
                .alias(col)
            )
        else:
            # Other categorical - use 'Unknown' category
            imputation_exprs.append(
                pl.when(pl.col(col).is_null())
                .then(pl.lit("Unknown"))
                .otherwise(pl.col(col))
                .alias(col)
            )
    
    # Add non-imputed columns
    other_cols = [col for col in cols_to_use if col not in numeric_cols + categorical_cols]
    for col in other_cols:
        imputation_exprs.append(pl.col(col))
    
    # Apply imputation
    df_imputed = df_lazy.with_columns(imputation_exprs)
    
    return df_imputed

def create_survival_dataset_lazy(file_path, quarter_name):
    """Create survival dataset using lazy evaluation for memory efficiency"""
    print(f"\n  Processing {quarter_name} with lazy evaluation...")
    
    try:
        # Start with lazy frame
        df_lazy = pl.scan_parquet(file_path)
        
        # Check available columns
        sample_df = df_lazy.select(pl.all()).limit(1).collect()
        available_columns = sample_df.columns
        print(f"   Available columns: {len(available_columns)}")
        
        # Essential columns for survival analysis
        essential_cols = ["LOAN_ID", "LOAN_AGE", "DLQ_STATUS"]
        
        # Important columns for Cox regression features
        important_cols = [
            "ORIG_DATE", "CSCORE_B", "DTI", "ORIG_RATE", "ORIG_UPB", 
            "ORIG_TERM", "OLTV", "OCLTV", "PURPOSE", "STATE", "PROP", 
            "OCC_STAT", "FORECLOSURE_DATE", "Zero_Bal_Code", "CURRENT_UPB", 
            "MATR_DT", "NUM_BO", "FIRST_FLAG", "CHANNEL", "SELLER_NAME"
        ]
        
        # Only use columns that exist
        cols_to_use = essential_cols + [col for col in important_cols if col in available_columns]
        print(f"Using {len(cols_to_use)} columns for analysis")
        
        # Memory check
        used_gb, available_gb = check_memory_usage()
        print(f"Memory: {used_gb:.1f}GB used, {available_gb:.1f}GB available")
        
        if available_gb < 2:
            print(" Low memory warning - processing in smaller batches")
            return process_in_batches(file_path, quarter_name, cols_to_use)
        
        # Select columns and apply basic filtering
        df_filtered = (
            df_lazy
            .select(cols_to_use)
            .filter(
                pl.col("LOAN_ID").is_not_null() &
                pl.col("LOAN_AGE").is_not_null()
            )
        )
        
        # Apply null handling
        df_clean = handle_missing_values(df_filtered, cols_to_use)
        
        # Get stats
        stats = (
            df_clean
            .select([
                pl.col("LOAN_ID").n_unique().alias("unique_loans"),
                pl.len().alias("total_rows")
            ])
            .collect()
        )
        
        unique_loans = stats['unique_loans'][0]
        total_rows = stats['total_rows'][0]
        print(f"   Data after cleaning: {unique_loans:,} unique loans, {total_rows:,} total rows")
        
        # Create baseline (first observation per loan)
        baseline_lazy = (
            df_clean
            .sort(["LOAN_ID", "LOAN_AGE"])
            .group_by("LOAN_ID")
            .first()
        )
        
        # Create events
        events_lazy = create_events_lazy(df_clean)
        
        # Join and create final dataset
        final_lazy = (
            baseline_lazy
            .join(events_lazy, on="LOAN_ID", how="inner")
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
                
                pl.lit(quarter_name).alias("quarter"),
                
                # Additional Cox regression features
                pl.when(pl.col("ORIG_DATE").is_not_null())
                .then(pl.col("ORIG_DATE").str.strptime(pl.Date, "%Y%m").dt.year())
                .otherwise(pl.lit(2000))
                .alias("orig_year"),
                
                # Risk buckets for continuous variables
                pl.when(pl.col("CSCORE_B") < 620)
                .then(pl.lit("Low"))
                .when(pl.col("CSCORE_B") < 720)
                .then(pl.lit("Medium"))
                .otherwise(pl.lit("High"))
                .alias("credit_score_bucket"),
                
                pl.when(pl.col("DTI") < 25)
                .then(pl.lit("Low"))
                .when(pl.col("DTI") < 35)
                .then(pl.lit("Medium"))
                .otherwise(pl.lit("High"))
                .alias("dti_bucket"),
                
                pl.when(pl.col("OLTV") < 80)
                .then(pl.lit("Low"))
                .when(pl.col("OLTV") < 90)
                .then(pl.lit("Medium"))
                .otherwise(pl.lit("High"))
                .alias("ltv_bucket")
            ])
        )
        
        # Collect final result
        print("   Computing final dataset...")
        final_df = final_lazy.collect()
        
        # Final validation and null check
        if len(final_df) == 0:
            print(f" No data in final dataset for {quarter_name}")
            return None
        
        # Check for remaining nulls in critical columns
        critical_cols = ["LOAN_ID", "survival_time", "survival_event", "LOAN_AGE"]
        null_check = final_df.select([
            pl.col(col).is_null().sum().alias(f"{col}_nulls") 
            for col in critical_cols if col in final_df.columns
        ]).collect()
        
        print(f"   Critical column null counts: {null_check.to_dict()}")
        
        # Remove any rows with null survival times or events (critical for Cox)
        final_df = final_df.filter(
            pl.col("survival_time").is_not_null() &
            pl.col("survival_event").is_not_null() &
            (pl.col("survival_time") > 0)
        )
        
        # Stats
        total_loans = len(final_df)
        default_count = final_df['default_event'].sum() or 0
        default_rate = (default_count / total_loans * 100) if total_loans > 0 else 0
        
        print(f" Final clean dataset: {total_loans:,} loans, {default_count:,} defaults ({default_rate:.2f}%)")
        
        return final_df
        
    except Exception as e:
        print(f"Error in lazy processing: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_events_lazy(df_lazy):
    """Create event calculations using lazy evaluation with null handling"""
    
    agg_expressions = [
        # Default events - handle nulls properly
        pl.when(
            pl.col("DLQ_STATUS").is_not_null() & 
            safe_string_contains("DLQ_STATUS", "3|4|5|6|7|8|9|X")
        ).then(1).otherwise(0).max().alias("ever_default_dlq"),
        
        pl.when(
            pl.col("DLQ_STATUS").is_not_null() & 
            safe_string_contains("DLQ_STATUS", "3|4|5|6|7|8|9|X")
        ).then(pl.col("LOAN_AGE")).min().alias("first_default_age_dlq"),
        
        # Early delinquencies (Stage 2)
        pl.when(
            pl.col("DLQ_STATUS").is_not_null() & 
            safe_string_contains("DLQ_STATUS", "1|2")
        ).then(pl.col("LOAN_AGE")).min().alias("first_stage2_age"),
        
        pl.when(
            pl.col("DLQ_STATUS").is_not_null() & 
            safe_string_contains("DLQ_STATUS", "1|2")
        ).then(1).otherwise(0).max().alias("ever_stage2"),
        
        # Observation info
        pl.col("LOAN_AGE").max().alias("last_observed_age"),
        pl.col("DLQ_STATUS").last().alias("final_dlq_status")
    ]
    
    # Add optional column aggregations with null handling
    try:
        # Sample to check if column exists
        sample = df_lazy.select("FORECLOSURE_DATE").limit(1).collect()
        agg_expressions.extend([
            pl.when(pl.col("FORECLOSURE_DATE").is_not_null())
            .then(1).otherwise(0).max().alias("ever_foreclosure"),
            pl.when(pl.col("FORECLOSURE_DATE").is_not_null())
            .then(pl.col("LOAN_AGE")).min().alias("first_foreclosure_age")
        ])
    except:
        agg_expressions.extend([
            pl.lit(0).alias("ever_foreclosure"),
            pl.lit(None).cast(pl.Int32).alias("first_foreclosure_age")
        ])
    
    try:
        sample = df_lazy.select("Zero_Bal_Code").limit(1).collect()
        agg_expressions.append(
            pl.when(
                pl.col("Zero_Bal_Code").is_not_null() &
                pl.col("Zero_Bal_Code").is_in(["03", "06", "09"])
            ).then(1).otherwise(0).max().alias("ever_default_zb")
        )
    except:
        agg_expressions.append(pl.lit(0).alias("ever_default_zb"))
    
    # Create events with proper null handling
    events_lazy = (
        df_lazy
        .group_by("LOAN_ID")
        .agg(agg_expressions)
        .with_columns([
            # Main event indicators
            pl.when(
                (pl.col("ever_default_dlq") == 1) |
                (pl.col("ever_foreclosure") == 1) |
                (pl.col("ever_default_zb") == 1)
            ).then(1).otherwise(0).alias("default_event"),
            
            pl.when(pl.col("ever_stage2") == 1)
            .then(1).otherwise(0).alias("stage2_event"),
        ])
        .with_columns([
            # Time to event calculations with null handling
            pl.coalesce([
                pl.col("first_default_age_dlq"),
                pl.col("first_foreclosure_age"),
                pl.col("last_observed_age")
            ]).alias("time_to_default_raw")
        ])
        .with_columns([
            # Survival times - ensure no nulls and positive values
            pl.when(pl.col("default_event") == 1)
            .then(
                pl.when(pl.col("time_to_default_raw").is_null() | (pl.col("time_to_default_raw") <= 0))
                .then(pl.lit(1))
                .otherwise(pl.col("time_to_default_raw"))
            )
            .otherwise(
                pl.when(pl.col("last_observed_age").is_null() | (pl.col("last_observed_age") <= 0))
                .then(pl.lit(1))
                .otherwise(pl.col("last_observed_age"))
            )
            .alias("survival_time"),
            
            pl.when(pl.col("stage2_event") == 1)
            .then(
                pl.when(pl.col("first_stage2_age").is_null() | (pl.col("first_stage2_age") <= 0))
                .then(pl.lit(1))
                .otherwise(pl.col("first_stage2_age"))
            )
            .otherwise(
                pl.when(pl.col("last_observed_age").is_null() | (pl.col("last_observed_age") <= 0))
                .then(pl.lit(1))
                .otherwise(pl.col("last_observed_age"))
            )
            .alias("stage2_survival_time"),
            
            # Event indicators (no nulls allowed)
            pl.col("default_event").fill_null(0).alias("survival_event"),
            pl.col("stage2_event").fill_null(0).alias("stage2_survival_event")
        ])
    )
    
    return events_lazy

def process_in_batches(file_path, quarter_name, cols_to_use):
    """Process file in batches for very low memory situations"""
    print(f"   Processing {quarter_name} in batches due to memory constraints...")
    
    try:
        # Get total row count
        total_rows = pl.scan_parquet(file_path).select(pl.len()).collect()[0, 0]
        num_batches = (total_rows // BATCH_SIZE) + 1
        
        print(f"   Processing {total_rows:,} rows in {num_batches} batches of {BATCH_SIZE:,}")
        
        all_results = []
        
        for batch_num in range(num_batches):
            offset = batch_num * BATCH_SIZE
            print(f"   Batch {batch_num + 1}/{num_batches}...")
            
            # Process batch with null handling
            batch_lazy = (
                pl.scan_parquet(file_path)
                .select(cols_to_use)
                .slice(offset, BATCH_SIZE)
                .filter(
                    pl.col("LOAN_ID").is_not_null() &
                    pl.col("LOAN_AGE").is_not_null()
                )
            )
            
            # Apply null handling to batch
            batch_clean = handle_missing_values(batch_lazy, cols_to_use)
            batch_df = batch_clean.collect()
            
            if len(batch_df) > 0:
                all_results.append(batch_df)
            
            # Memory cleanup
            del batch_df
            gc.collect()
        
        if not all_results:
            return None
        
        # Combine all batches
        print("   Combining batches...")
        combined_df = pl.concat(all_results)
        
        # Clean up
        for df in all_results:
            del df
        del all_results
        gc.collect()
        
        # Process combined data
        print("   Creating survival dataset from combined data...")
        return process_combined_data(combined_df, quarter_name)
        
    except Exception as e:
        print(f"   Error in batch processing: {e}")
        return None

def process_combined_data(df, quarter_name):
    """Process the combined dataframe with null handling"""
    try:
        # Get baseline
        baseline = (
            df
            .sort(["LOAN_ID", "LOAN_AGE"])
            .group_by("LOAN_ID")
            .first()
        )
        
        # Calculate events with null handling
        events = (
            df
            .group_by("LOAN_ID")
            .agg([
                pl.when(
                    pl.col("DLQ_STATUS").is_not_null() &
                    safe_string_contains("DLQ_STATUS", "3|4|5|6|7|8|9|X")
                ).then(1).otherwise(0).max().alias("default_event"),
                
                pl.when(
                    pl.col("DLQ_STATUS").is_not_null() &
                    safe_string_contains("DLQ_STATUS", "3|4|5|6|7|8|9|X")
                ).then(pl.col("LOAN_AGE")).min().alias("time_to_default"),
                
                pl.col("LOAN_AGE").max().alias("last_observed_age")
            ])
            .with_columns([
                # Ensure no null survival times
                pl.when(pl.col("default_event") == 1)
                .then(
                    pl.when(pl.col("time_to_default").is_null() | (pl.col("time_to_default") <= 0))
                    .then(pl.lit(1))
                    .otherwise(pl.col("time_to_default"))
                )
                .otherwise(
                    pl.when(pl.col("last_observed_age").is_null() | (pl.col("last_observed_age") <= 0))
                    .then(pl.lit(1))
                    .otherwise(pl.col("last_observed_age"))
                )
                .alias("survival_time"),
                
                pl.col("default_event").fill_null(0).alias("survival_event")
            ])
        )
        
        # Join and finalize
        final_df = (
            baseline
            .join(events, on="LOAN_ID", how="inner")
            .with_columns([
                pl.lit(quarter_name).alias("quarter"),
                pl.when((pl.col("survival_time") <= 12) & (pl.col("survival_event") == 1))
                .then(1).otherwise(0).alias("default_12m")
            ])
            .filter(
                pl.col("survival_time").is_not_null() &
                pl.col("survival_event").is_not_null() &
                (pl.col("survival_time") > 0)
            )
        )
        
        return final_df
        
    except Exception as e:
        print(f"   Error in combined processing: {e}")
        return None

def final_data_quality_check(df):
    """Perform final data quality checks for Cox regression readiness"""
    print("\n=== Final Data Quality Check for Cox Regression ===")
    
    # Check for nulls in critical columns
    critical_cols = ["LOAN_ID", "survival_time", "survival_event"]
    null_counts = {}
    
    for col in critical_cols:
        if col in df.columns:
            null_count = df.select(pl.col(col).is_null().sum()).item()
            null_counts[col] = null_count
            print(f"{col}: {null_count} nulls")
    
    # Check survival time distribution
    survival_stats = df.select([
        pl.col("survival_time").min().alias("min_time"),
        pl.col("survival_time").max().alias("max_time"),
        pl.col("survival_time").mean().alias("mean_time"),
        pl.col("survival_time").median().alias("median_time")
    ]).collect()
    
    print(f"\nSurvival time statistics:")
    print(f"Min: {survival_stats['min_time'][0]}")
    print(f"Max: {survival_stats['max_time'][0]}")
    print(f"Mean: {survival_stats['mean_time'][0]:.2f}")
    print(f"Median: {survival_stats['median_time'][0]}")
    
    # Check event distribution
    event_stats = df.select([
        pl.col("survival_event").sum().alias("events"),
        pl.len().alias("total"),
        (pl.col("survival_event").sum() / pl.len() * 100).alias("event_rate")
    ]).collect()
    
    events = event_stats['events'][0]
    total = event_stats['total'][0]
    event_rate = event_stats['event_rate'][0]
    
    print(f"\nEvent distribution:")
    print(f"Events: {events:,}")
    print(f"Total: {total:,}")
    print(f"Event rate: {event_rate:.2f}%")
    
    # Check for any remaining data quality issues
    issues = []
    
    if any(null_counts.values()):
        issues.append("Null values in critical columns")
    
    if events < 10:
        issues.append("Very few events (< 10)")
    
    if event_rate < 0.1:
        issues.append("Very low event rate (< 0.1%)")
    
    if survival_stats['min_time'][0] <= 0:
        issues.append("Non-positive survival times")
    
    if issues:
        print(f"\n Data quality issues found:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print(f"\nData appears ready for Cox regression analysis")
    
    return len(issues) == 0

# Main processing loop
print("Starting enhanced survival analysis processing with null handling...")
print(f"Output directory: {SURVIVAL_OUTPUT_DIR}")

for year in range(2017, 2023):
    for q in ['Q1', 'Q3']:
        quarter = f"{year}{q}"
        parquet_path = os.path.join(BASE_INPUT_DIR, f"{quarter}.parquet")

        if not os.path.exists(parquet_path):
            print(f"File not found: {parquet_path}")
            continue

        print(f"\n Processing {quarter}...")
        
        # Check file size
        file_size_mb = os.path.getsize(parquet_path) / (1024 * 1024)
        print(f" File size: {file_size_mb:.1f} MB")

        try:
            used_gb, available_gb = check_memory_usage()
            print(f" Pre-processing memory: {used_gb:.1f}GB used, {available_gb:.1f}GB available")
            
            if available_gb < 1:
                print(" Very low memory - skipping this file")
                continue
                
            survival_df = create_survival_dataset_lazy(parquet_path, quarter)

            if survival_df is None:
                print(f" Skipping {quarter} - no valid data produced")
                continue

            # Perform data quality check
            is_ready = final_data_quality_check(survival_df)
            
            # Save result
            survival_output_path = os.path.join(SURVIVAL_OUTPUT_DIR, f"survival_{quarter}.parquet")
            survival_df.write_parquet(survival_output_path, compression="snappy")
            
            file_size_out = os.path.getsize(survival_output_path) / (1024 * 1024)
            print(f" Saved: survival_{quarter}.parquet ({file_size_out:.1f} MB)")

            # Cleanup
            del survival_df
            gc.collect()

            # Final memory check
            used_gb, available_gb = check_memory_usage()
            print(f"Completed {quarter} - Memory: {used_gb:.1f}GB used, {available_gb:.1f}GB available")

        except Exception as e:
            print(f"Error processing {quarter}: {str(e)}")
            import traceback
            traceback.print_exc()
            gc.collect()
            continue

print("\n Processing complete!")
