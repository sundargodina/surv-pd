import polars as pl
import gc
import os
import psutil
from typing import Optional
import pyarrow as pa

def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process()
    return process.memory_info().rss / (1024**3)

def optimize_chunk_size(df_size: int, available_memory_gb: float = 2.0) -> int:
    """Calculate optimal chunk size based on available memory"""
    # Estimate ~8GB needed per 10M records, but be conservative for i3 MacBook Air
    records_per_gb = 1_000_000  # Very conservative estimate
    max_chunk_records = int(available_memory_gb * records_per_gb)
    
    # Ensure minimum viable chunk size
    min_chunk = 500_000
    optimal_chunk = max(min_chunk, min(max_chunk_records, df_size // 4))
    
    print(f"Dataset size: {df_size:,} records")
    print(f"Calculated chunk size: {optimal_chunk:,} records")
    print(f"Estimated chunks needed: {(df_size + optimal_chunk - 1) // optimal_chunk}")
    
    return optimal_chunk

def create_survival_dataset_chunked(df, quarter_name: str, chunk_size: Optional[int] = None):
    """Create time-varying survival dataset with chunked processing"""
    
    # Filter valid observations first
    print(f"Initial filtering for {quarter_name}...")
    df_filtered = df.filter(pl.col("LOAN_AGE") >= 0)
    total_records = len(df_filtered)
    
    print(f"Processing {quarter_name}: {total_records:,} records after filtering")
    
    # Determine chunk size if not provided
    if chunk_size is None:
        chunk_size = optimize_chunk_size(total_records)
    
    # Quick data validation on first chunk only
    sample_df = df_filtered.head(min(10000, len(df_filtered)))
    validate_dlq_status(sample_df)
    print_data_summary(sample_df, quarter_name)
    del sample_df
    gc.collect()
    
    # Process in chunks
    all_chunks = []
    num_chunks = (total_records + chunk_size - 1) // chunk_size
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, total_records)
        
        print(f"\nProcessing chunk {i+1}/{num_chunks} (rows {start_idx:,}-{end_idx:,})")
        print(f"Memory usage: {get_memory_usage():.1f}GB")
        
        # Get chunk
        chunk = df_filtered.slice(start_idx, end_idx - start_idx)
        
        # Process chunk
        processed_chunk = process_chunk(chunk, quarter_name)
        
        # Store processed chunk
        all_chunks.append(processed_chunk)
        
        # Cleanup
        del chunk, processed_chunk
        gc.collect()
        
        # Memory check
        current_memory = get_memory_usage()
        if current_memory > 6.0:  # Warning threshold for 8GB system
            print(f"Warning: High memory usage ({current_memory:.1f}GB)")
    
    # Combine all chunks
    print(f"\nCombining {len(all_chunks)} chunks...")
    survival_df = pl.concat(all_chunks)
    
    # Cleanup chunk list
    del all_chunks
    gc.collect()
    
    # Final deduplication (using renamed columns)
    print("Final deduplication...")
    original_count = len(survival_df)
    survival_df = survival_df.unique(subset=["LOAN_ID", "time"], keep="last")
    dedup_count = len(survival_df)
    
    if original_count != dedup_count:
        print(f"Removed {original_count - dedup_count:,} duplicates")
    
    # Final event summary
    print_event_summary(survival_df)
    
    return survival_df

def process_chunk(chunk_df, quarter_name: str):
    """Process a single chunk with all transformations"""
    
    return chunk_df.with_columns([
        # Event indicator: default event (90+ days delinquent or credit events)
        pl.when(
            (
                (pl.col("DLQ_STATUS").is_not_null()) &
                (pl.col("DLQ_STATUS") != "XX") &
                (pl.col("DLQ_STATUS").cast(pl.Int32, strict=False).is_not_null()) &
                (pl.col("DLQ_STATUS").cast(pl.Int32, strict=False) >= 3)
            )
            |
            pl.col("Zero_Bal_Code").is_in(["03", "09", "15", "97", "98"])
        ).then(pl.lit(1)).otherwise(pl.lit(0)).alias("event"),
        
        # Time variable
        pl.col("LOAN_AGE").alias("time"),
        
        # Create delinquency severity categories (optimized)
        pl.when(pl.col("DLQ_STATUS").is_null() | (pl.col("DLQ_STATUS") == ""))
        .then(pl.lit("Missing"))
        .when(pl.col("DLQ_STATUS") == "XX")
        .then(pl.lit("Unknown"))
        .when(pl.col("DLQ_STATUS") == "00")
        .then(pl.lit("Current"))
        .when(pl.col("DLQ_STATUS").is_in(["01", "02"]))
        .then(pl.lit("Early_DLQ"))
        .when(pl.col("DLQ_STATUS").is_in(["03", "04", "05"]))
        .then(pl.lit("Serious_DLQ"))
        .when(pl.col("DLQ_STATUS").is_in(["06", "07", "08", "09"]))
        .then(pl.lit("Severe_DLQ"))
        .when(pl.col("DLQ_STATUS") == "99")
        .then(pl.lit("Extreme_DLQ"))
        .otherwise(pl.lit("Other"))
        .alias("dlq_severity"),
        
        # Binary indicators (vectorized)
        pl.col("DLQ_STATUS").is_in(["01", "02", "03", "04", "05", "06", "07", "08", "09", "99"])
        .cast(pl.Int8).alias("any_delinquent"),
        
        pl.col("DLQ_STATUS").is_in(["03", "04", "05", "06", "07", "08", "09", "99"])
        .cast(pl.Int8).alias("serious_delinquent"),
        
        # Time-varying ratios (optimized null handling)
        (pl.col("CURRENT_UPB") / pl.col("ORIG_UPB").clip(lower_bound=1))
        .alias("balance_ratio"),
        
        (pl.col("CURR_RATE") - pl.col("ORIG_RATE")).alias("rate_change"),
        
        (pl.col("CURR_SCOREB") - pl.col("CSCORE_B")).alias("credit_score_change"),
        
        # Keep essential columns only
        pl.col("DLQ_STATUS").alias("dlq_status_original"),
        pl.lit(quarter_name).alias("data_quarter")
    ]).select([
        # Keep only essential columns to reduce memory
        "LOAN_ID", "time", "event", "dlq_severity", "any_delinquent", 
        "serious_delinquent", "balance_ratio", "rate_change", 
        "credit_score_change", "dlq_status_original", "data_quarter"
    ])

def print_data_summary(df, quarter_name: str):
    """Print concise data summary to avoid memory issues"""
    unique_dlq = df["DLQ_STATUS"].unique().sort().to_list()
    print(f"  Unique DLQ_STATUS values: {len(unique_dlq)} categories")
    
    # Count key categories only
    key_counts = df.filter(
        pl.col("DLQ_STATUS").is_in(["00", "01", "02", "03", "XX"])
    ).group_by("DLQ_STATUS").agg(pl.len().alias("count"))
    
    for row in key_counts.iter_rows():
        print(f"    {row[0]}: {row[1]:,}")

def print_event_summary(df):
    """Print event summary"""
    total_count = len(df)
    event_count = df["event"].sum()
    event_rate = (event_count / total_count * 100) if total_count > 0 else 0
    
    print("  Event distribution:")
    print(f"    No Event: {total_count - event_count:,} ({100 - event_rate:.1f}%)")
    print(f"    Default Event: {event_count:,} ({event_rate:.1f}%)")

def append_to_master_parquet_streaming(new_data, master_file="survival_master.parquet"):
    """Corrected memory-efficient append using Polars' lazy API"""
    
    print(f"Preparing to append {len(new_data):,} records...")
    
    if os.path.exists(master_file):
        print("Master file exists - will append new data")
        
        # Lazy read of the existing master file
        existing_lazy = pl.scan_parquet(master_file)
        
        # Combine the existing and new data in a lazy fashion
        combined_lazy = pl.concat([existing_lazy, new_data.lazy()])
        
        # Write the combined data back to the master file
        # The 'overwrite=True' is fine here because we combined the data first
        combined_lazy.collect().write_parquet(
            master_file, 
            compression="zstd", 
            compression_level=3,
            )
            
    else:
        print("Creating new master file...")
        new_data.write_parquet(
            master_file, 
            compression="zstd", 
            compression_level=3
        )
    
    # Get final stats using scan (memory efficient)
    final_stats = pl.scan_parquet(master_file).select([
        pl.len().alias("total_records"),
        pl.col("LOAN_ID").n_unique().alias("unique_loans"),
        pl.col("event").mean().alias("event_rate")
    ]).collect()
    
    total_records = final_stats["total_records"][0]
    unique_loans = final_stats["unique_loans"][0]
    event_rate = final_stats["event_rate"][0] * 100

    print(f"Master file updated: {total_records:,} total records")
    print(f"Summary: {unique_loans:,} unique loans, {event_rate:.2f}% event rate")
    
    return total_records, unique_loans
def validate_dlq_status(df):
    """Lightweight DLQ_STATUS validation"""
    try:
        dlq_sample = df["DLQ_STATUS"].unique().sort().to_list()[:20]  # Sample only
        
        # Check for obvious issues
        non_standard = [v for v in dlq_sample if v not in 
                       [f"{i:02d}" for i in range(100)] + ["XX", "", None]]
        
        if non_standard:
            print(f"Warning: Non-standard DLQ_STATUS values: {non_standard[:5]}...")
        else:
            print("DLQ_STATUS values appear valid")
            
    except Exception as e:
        print(f"DLQ_STATUS validation failed: {e}")

# Updated main processing function
def process_quarter_optimized(file_path: str, quarter_name: str):
    """Main function to process a quarter with memory optimization"""
    
    print(f"=== PROCESSING {quarter_name} ===")
    print(f"Initial memory usage: {get_memory_usage():.1f}GB")
    
    try:
        # First, check schema without loading full data
        schema_check = pl.scan_parquet(file_path).select(pl.all()).head(1).collect()
        print(f"Available columns: {list(schema_check.columns)}")
        
        # Read parquet file
        df = pl.read_parquet(file_path)
        print(f"Loaded {len(df):,} records")
        
        # Process with chunking
        survival_df = create_survival_dataset_chunked(df, quarter_name)
        
        # Clean up original dataframe immediately
        del df
        gc.collect()
        
        # Append to master file
        total_records, unique_loans = append_to_master_parquet_streaming(survival_df)
        
        # Final cleanup
        del survival_df
        gc.collect()
        
        print(f"Master file: {total_records:,} records, {unique_loans:,} unique loans")
        print(f"Final memory usage: {get_memory_usage():.1f}GB")
        print(f"{quarter_name} complete\n")
        
        return True
        
    except MemoryError:
        print(f"Memory error processing {quarter_name}")
        gc.collect()
        return False
    except Exception as e:
        print(f"Error processing {quarter_name}: {e}")
        gc.collect()
        return False

process_quarter_optimized("/Users/sundargodina/Downloads/project/2017Q1.parquet", "2017Q1")
process_quarter_optimized("/Users/sundargodina/Downloads/project/2017Q3.parquet", "2017Q3")
process_quarter_optimized("/Users/sundargodina/Downloads/project/2018Q1.parquet", "2018Q1")
process_quarter_optimized("/Users/sundargodina/Downloads/project/2018Q3.parquet", "2018Q3")
process_quarter_optimized("/Users/sundargodina/Downloads/project/2019Q1.parquet", "2019Q1")
process_quarter_optimized("/Users/sundargodina/Downloads/project/2019Q3.parquet", "2019Q3")
process_quarter_optimized("/Users/sundargodina/Downloads/project/2020Q1.parquet", "2020Q1")
process_quarter_optimized("/Users/sundargodina/Downloads/project/2020Q3.parquet", "2020Q3")
process_quarter_optimized("/Users/sundargodina/Downloads/project/2021Q1.parquet", "2021Q1")
process_quarter_optimized("/Users/sundargodina/Downloads/project/2021Q3.parquet", "2021Q3")
process_quarter_optimized("/Users/sundargodina/Downloads/project/2022Q1.parquet", "2022Q1")
process_quarter_optimized("/Users/sundargodina/Downloads/project/2022Q3.parquet", "2022Q3")
