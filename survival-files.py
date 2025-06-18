# Processing Q1 and Q3 files from 2017-2022 - MacBook Air Optimized
import polars as pl
import os
from datetime import datetime
import warnings
import gc
import psutil
warnings.filterwarnings('ignore')

# Configuration for MacBook Air 2020
BASE_INPUT_DIR = "/Users/sundargodina/Downloads/project/"
SURVIVAL_OUTPUT_DIR = "/Users/sundargodina/Downloads/survival_files/"  # Fixed path
BATCH_SIZE = 50000  # Process in smaller chunks
MAX_MEMORY_GB = 6   # Conservative limit for MacBook Air

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

def create_survival_dataset_lazy(file_path, quarter_name):
    """Create survival dataset using lazy evaluation for memory efficiency"""
    print(f"\n  Processing {quarter_name} with lazy evaluation...")
    
    try:
        # Start with lazy frame - no data loaded yet
        df_lazy = pl.scan_parquet(file_path)
        
        # Check what columns we have available
        sample_df = df_lazy.select(pl.all()).limit(1).collect()
        available_columns = sample_df.columns
        print(f"   Available columns: {len(available_columns)}")
        
        # Essential columns only
        essential_cols = ["LOAN_ID", "LOAN_AGE", "DLQ_STATUS"]
        
        # Optional but useful columns
        optional_cols = [
            "ORIG_DATE", "CSCORE_B", "DTI", "ORIG_RATE", "ORIG_UPB", 
            "ORIG_TERM", "OLTV", "OCLTV", "PURPOSE", "STATE", "PROP", 
            "OCC_STAT", "FORECLOSURE_DATE", "Zero_Bal_Code", "CURRENT_UPB", "MATR_DT"
        ]
        
        # Only use columns that actually exist
        cols_to_use = essential_cols + [col for col in optional_cols if col in available_columns]
        print(f"Using {len(cols_to_use)} columns")
        
        # Memory check
        used_gb, available_gb = check_memory_usage()
        print(f"Memory: {used_gb:.1f}GB used, {available_gb:.1f}GB available")
        
        if available_gb < 2:
            print(" Low memory warning - processing in smaller batches")
            return process_in_batches(file_path, quarter_name, cols_to_use)
        
        # Select only needed columns and basic filtering
        df_filtered = (
            df_lazy
            .select(cols_to_use)
            .filter(
                pl.col("LOAN_ID").is_not_null() &
                pl.col("LOAN_AGE").is_not_null() &
                pl.col("DLQ_STATUS").is_not_null()
            )
        )
        
        # Get basic stats without collecting full data
        stats = (
            df_filtered
            .select([
                pl.col("LOAN_ID").n_unique().alias("unique_loans"),
                pl.len().alias("total_rows")
            ])
            .collect()
        )
        
        unique_loans = stats['unique_loans'][0]
        total_rows = stats['total_rows'][0]
        print(f"   Data: {unique_loans:,} unique loans, {total_rows:,} total rows")
        
        # Create baseline (first observation per loan) lazily
        baseline_lazy = (
            df_filtered
            .sort(["LOAN_ID", "LOAN_AGE"])
            .group_by("LOAN_ID")
            .first()
        )
        
        events_lazy = create_events_lazy(df_filtered)
        
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
                
                pl.lit(quarter_name).alias("quarter")
            ])
        )
        
        # Now collect the final result - this is when computation happens
        print("   Computing final dataset...")
        final_df = final_lazy.collect()
        
        # Quick validation
        if len(final_df) == 0:
            print(f" No data in final dataset for {quarter_name}")
            return None
            
        # Stats
        total_loans = len(final_df)
        default_count = final_df['default_event'].sum() or 0
        default_rate = (default_count / total_loans * 100) if total_loans > 0 else 0
        
        print(f" Final: {total_loans:,} loans, {default_count:,} defaults ({default_rate:.2f}%)")
        
        return final_df
        
    except Exception as e:
        print(f"Error in lazy processing: {e}")
        return None

def create_events_lazy(df_lazy):
    """Create event calculations using lazy evaluation"""
    
    agg_expressions = [
        # Default events
        pl.when(safe_string_contains("DLQ_STATUS", "3|4|5|6|7|8|9|X"))
        .then(1).otherwise(0).max().alias("ever_default_dlq"),
        
        pl.when(safe_string_contains("DLQ_STATUS", "3|4|5|6|7|8|9|X"))
        .then(pl.col("LOAN_AGE")).min().alias("first_default_age_dlq"),
        
        # Early delinquencies  
        pl.when(safe_string_contains("DLQ_STATUS", "1|2"))
        .then(pl.col("LOAN_AGE")).min().alias("first_stage2_age"),
        
        pl.when(safe_string_contains("DLQ_STATUS", "1|2"))
        .then(1).otherwise(0).max().alias("ever_stage2"),
        
        # Observation info
        pl.col("LOAN_AGE").max().alias("last_observed_age"),
        pl.col("DLQ_STATUS").last().alias("final_dlq_status")
    ]
    
    # Add conditional aggregations for optional columns
    try:
        # Check if FORECLOSURE_DATE exists by trying to select it
        test_fc = df_lazy.select("FORECLOSURE_DATE").limit(1)
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
        # Check if Zero_Bal_Code exists
        test_zb = df_lazy.select("Zero_Bal_Code").limit(1)
        agg_expressions.append(
            pl.when(pl.col("Zero_Bal_Code").is_in(["03", "06", "09"]))
            .then(1).otherwise(0).max().alias("ever_default_zb")
        )
    except:
        agg_expressions.append(pl.lit(0).alias("ever_default_zb"))
    
    # Create events with survival variables
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
            
            # Time calculations
            pl.coalesce([
                pl.col("first_default_age_dlq"),
                pl.col("first_foreclosure_age")
            ]).alias("time_to_default")
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
            # Ensure positive survival times
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
            
            # Process batch
            batch_lazy = (
                pl.scan_parquet(file_path)
                .select(cols_to_use)
                .slice(offset, BATCH_SIZE)
                .filter(
                    pl.col("LOAN_ID").is_not_null() &
                    pl.col("LOAN_AGE").is_not_null() &
                    pl.col("DLQ_STATUS").is_not_null()
                )
            )
            
            batch_df = batch_lazy.collect()
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
        
        # Now process normally with the combined data
        print("   Creating survival dataset from combined data...")
        return process_combined_data(combined_df, quarter_name)
        
    except Exception as e:
        print(f"   Error in batch processing: {e}")
        return None

def process_combined_data(df, quarter_name):
    """Process the combined dataframe normally"""
    # This is similar to your original logic but more memory efficient
    try:
        # Get baseline
        baseline = (
            df
            .sort(["LOAN_ID", "LOAN_AGE"])
            .group_by("LOAN_ID")
            .first()
        )
        
        # Calculate events (simplified)
        events = (
            df
            .group_by("LOAN_ID")
            .agg([
                pl.when(safe_string_contains("DLQ_STATUS", "3|4|5|6|7|8|9|X"))
                .then(1).otherwise(0).max().alias("default_event"),
                
                pl.when(safe_string_contains("DLQ_STATUS", "3|4|5|6|7|8|9|X"))
                .then(pl.col("LOAN_AGE")).min().alias("time_to_default"),
                
                pl.col("LOAN_AGE").max().alias("last_observed_age")
            ])
            .with_columns([
                pl.when(pl.col("default_event") == 1)
                .then(pl.col("time_to_default"))
                .otherwise(pl.col("last_observed_age"))
                .alias("survival_time"),
                
                pl.col("default_event").alias("survival_event")
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
        )
        
        return final_df
        
    except Exception as e:
        print(f"   Error in combined processing: {e}")
        return None

# Main processing loop
print("Starting survival analysis processing for MacBook Air...")
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
            gc.collect()
            continue

print("\n Processing complete") 
