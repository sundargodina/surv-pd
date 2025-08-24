import concurrent.futures
import time
import pandas as pd
import polars as pl
from fredapi import Fred
from sklearn.decomposition import PCA
import numpy as np
import os
import datetime
import gc
import math

class FREDDataFetcher:
    """
    Fetches macroeconomic time series data from the FRED API.
    Handles data cleaning and resampling to a quarterly frequency.
    """
    def __init__(self, api_key, cache_file='macro_data.parquet'):
        self.fred = Fred(api_key=api_key)
        self.cache_file = cache_file
        
        # State-level series to fetch.
        self.state_macro_series = {
            'UNRATE': 'UNR',
            'AHETPI': 'WAGE',
            'RGSP': 'RGSP',
            'HOUST': 'HOUST',
            'STHPI': 'STHPI',
            'COMPUTSA': 'COMPUTSA',
            'NFCI': 'NFCI' # Financial stress indicator
        }
        
        # National-level series to fetch.
        self.national_macro_series = {
            # Interest Rate Environment
            'DGS3MO': 'DGS3MO', 'DGS2': 'DGS2', 'DGS10': 'DGS10', 'DFF': 'DFF',
            # Credit Market Conditions
            'MORTGAGE30US': 'MORTGAGE30US', 'AAA': 'AAA', 'BAA': 'BAA', 'BAMLH0A0HYM2': 'BAMLH0A0HYM2',
            # Labor Market Dynamics
            'UNRATE': 'UNRATE', 'PAYEMS': 'PAYEMS', 'AHETPI': 'AHETPI', 'CIVPART': 'CIVPART',
            # Inflation and Price Stability
            'CPIAUCSL': 'CPIAUCSL', 'CPILFESL': 'CPILFESL', 'PCEPI': 'PCEPI',
            # Economic Activity
            'GDPC1': 'GDPC1', 'INDPRO': 'INDPRO', 'RRSFS': 'RRSFS', 'DSPIC96': 'DSPIC96', 'PSAVERT': 'PSAVERT',
            # Housing Market Fundamentals
            'CSUSHPISA': 'CSUSHPISA',
            # Financial Stress Indicators
            'NFCI': 'NFCI', 'STLFSI4': 'STLFSI4',
            # Market Sentiment
            'VIXCLS': 'VIXCLS', 'NASDAQCOM': 'NASDAQCOM', 'DEXUSEU': 'DEXUSEU'
        }

    def _fetch_series(self, series_id, start_date, end_date, retries=3, delay=5):
        """Fetches a single time series with a robust exponential backoff retry strategy."""
        attempt = 0
        while attempt < retries:
            try:
                data = self.fred.get_series(
                    series_id,
                    observation_start=start_date,
                    observation_end=end_date
                )
                if data is not None and not data.empty:
                    return data
                else:
                    print(f"✗ {series_id}: No data returned.")
                    return None
            except Exception as e:
                if "Too Many Requests" in str(e):
                    print(f"Rate limit hit for {series_id}. Retrying in {delay**attempt} seconds...")
                    time.sleep(delay**attempt)
                    attempt += 1
                elif "Bad Request" in str(e) and "series does not exist" in str(e):
                    print(f"✗ {series_id}: Failed after {attempt + 1} attempts - Bad Request. The series does not exist.")
                    return None
                else:
                    print(f"✗ {series_id}: Failed after {attempt + 1} attempts - {e}")
                    attempt += 1
        print(f"✗ {series_id}: All retry attempts failed.")
        return None
    
    def _fetch_state_data(self, state, start_date, end_date):
        """Fetches all specified macroeconomic data for a single state."""
        print(f"Fetching data for {state} from {start_date.date()} to {end_date.date()}...")
        quarterly_data = []
        for name, series_suffix in self.state_macro_series.items():
            series_id = f"{state}{series_suffix}"
            time.sleep(0.5)
            data = self._fetch_series(series_id, start_date, end_date)
            if data is not None and not data.empty:
                print(f"  ✓ {series_id}: {len(data)} observations")
                df = pl.DataFrame({
                    'date': data.index.to_numpy(),
                    series_id: data.values
                }).with_columns(
                    pl.col('date').cast(pl.Date)
                ).group_by_dynamic(
                    'date', every="3mo", closed='left'
                ).agg(
                    pl.col(series_id).mean().alias(f"{name}")
                )
                quarterly_data.append(df)
            else:
                print(f"  ✗ {series_id}: No data")
        if not quarterly_data:
            return pl.DataFrame()
        combined_df = quarterly_data[0]
        for df in quarterly_data[1:]:
            combined_df = combined_df.join(df, on='date', how='outer', coalesce=True)
        combined_df = combined_df.with_columns(pl.lit(state).alias('STATE'))
        return combined_df

    def _fetch_national_data(self, start_date, end_date):
        """Fetches all specified national macroeconomic data."""
        print(f"Fetching national data from {start_date.date()} to {end_date.date()}...")
        quarterly_data = []
        for name, series_id in self.national_macro_series.items():
            time.sleep(0.5)
            data = self._fetch_series(series_id, start_date, end_date)
            if data is not None and not data.empty:
                print(f"  ✓ {series_id}: {len(data)} observations")
                df = pl.DataFrame({
                    'date': data.index.to_numpy(),
                    series_id: data.values
                }).with_columns(
                    pl.col('date').cast(pl.Date)
                ).group_by_dynamic(
                    'date', every="3mo", closed='left'
                ).agg(
                    pl.col(series_id).mean().alias(name)
                )
                quarterly_data.append(df)
            else:
                print(f"  ✗ {series_id}: No data")
        if not quarterly_data:
            return pl.DataFrame()
        combined_df = quarterly_data[0]
        for df in quarterly_data[1:]:
            combined_df = combined_df.join(df, on='date', how='outer', coalesce=True)
        return combined_df

    def fetch_all_data(self, states, start_date, end_date):
        """Fetches and caches all macroeconomic data for a list of states."""
        if os.path.exists(self.cache_file):
            print(f"Using cached macro data from '{self.cache_file}'.")
            return pl.read_parquet(self.cache_file)

        state_results = []
        for state in states:
            state_data = self._fetch_state_data(state, start_date, end_date)
            if state_data is not None and not state_data.is_empty():
                state_results.append(state_data)
            print(f"Waiting 5 seconds before fetching data for the next state...")
            time.sleep(5)
        
        national_data = self._fetch_national_data(start_date, end_date)

        if not state_results:
            print("Failed to fetch any state-level data.")
            return pl.DataFrame()
        
        all_data = pl.concat(state_results)
        
        # Join national data to state data based on the date
        all_data = all_data.join(national_data, on='date', how='left')

        all_data.write_parquet(self.cache_file)
        print(f"Successfully fetched and cached data to '{self.cache_file}'.")
        return all_data

class MemoryEfficientProcessor:
    """
    Memory-efficient processor that avoids cross joins and processes data in chunks
    """
    def __init__(self, macro_df: pl.DataFrame, chunk_size: int = 50000):
        self.macro_df = macro_df
        self.chunk_size = chunk_size
        self.pca_domains = {
            'Interest_Rate': ['DGS3MO', 'DGS2', 'DGS10', 'DFF'],
            'Credit_Market': ['MORTGAGE30US', 'AAA', 'BAA', 'BAMLH0A0HYM2'],
            'Labor_Market': ['UNRATE', 'PAYEMS', 'AHETPI', 'CIVPART'],
            'Inflation': ['CPIAUCSL', 'CPILFESL', 'PCEPI'],
            'Economic_Activity': ['GDPC1', 'INDPRO', 'RRSFS', 'DSPIC96', 'PSAVERT'],
            'Housing_Market': ['CSUSHPISA', 'HOUST'],
            'Financial_Stress': ['NFCI', 'STLFSI4'],
            'Market_Sentiment': ['VIXCLS', 'NASDAQCOM', 'DEXUSEU']
        }

    def _impute_macro_data(self, df):
        """Comprehensively imputes missing values in the macroeconomic dataframe."""
        print("Imputing missing values in macro data...")
        
        # Check if STATE column exists, if not, sort by date only  
        if 'STATE' in df.columns:
            df = df.sort(['STATE', 'date'])
        else:
            df = df.sort('date')
        
        # Get numeric columns for imputation
        numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
        
        for col in numeric_cols:
            # Check if we have STATE column for grouped operations
            if 'STATE' in df.columns:
                # Forward fill within each state group
                df = df.with_columns(
                    pl.col(col).forward_fill().over('STATE')
                )
                # Backward fill within each state group
                df = df.with_columns(
                    pl.col(col).backward_fill().over('STATE')
                )
            else:
                # Forward and backward fill globally
                df = df.with_columns(pl.col(col).forward_fill())
                df = df.with_columns(pl.col(col).backward_fill())
            
            # If still nulls exist, use overall median
            if df[col].null_count() > 0:
                median_val = df[col].median()
                if median_val is not None:
                    df = df.with_columns(pl.col(col).fill_null(median_val))
                else:
                    # If median is also null, use 0 as last resort
                    df = df.with_columns(pl.col(col).fill_null(0))
        
        # More aggressive null handling - replace any remaining nulls
        for col in numeric_cols:
            if df[col].null_count() > 0:
                # Try mean first, then median, then 0
                mean_val = df[col].mean()
                if mean_val is not None and not np.isnan(mean_val):
                    df = df.with_columns(pl.col(col).fill_null(mean_val))
                else:
                    median_val = df[col].median()
                    if median_val is not None and not np.isnan(median_val):
                        df = df.with_columns(pl.col(col).fill_null(median_val))
                    else:
                        # Last resort - use 0
                        df = df.with_columns(pl.col(col).fill_null(0))
        
        # Convert any remaining NaN to 0 (handles numpy NaN vs Polars null)
        for col in numeric_cols:
            df = df.with_columns(
                pl.when(pl.col(col).is_nan()).then(0).otherwise(pl.col(col)).alias(col)
            )
        
        # Final verification
        null_counts = {col: df[col].null_count() for col in numeric_cols}
        nan_counts = {col: df.filter(pl.col(col).is_nan()).height for col in numeric_cols}
        
        total_nulls = sum(null_counts.values()) + sum(nan_counts.values())
        if total_nulls > 0:
            print(f"Warning: {sum(null_counts.values())} nulls and {sum(nan_counts.values())} NaNs remain")
        else:
            print("✓ All nulls and NaNs successfully handled")
            
        return df

    def _transform_macro_data(self, df):
        """Applies transformations and domain-specific PCA to the data."""
        print("Transforming macro data...")
        
        # Standardize state column name first
        if 'State' in df.columns:
            df = df.rename({'State': 'STATE'})
        
        # Ensure all columns are properly typed
        for col in df.columns:
            if col not in ['date', 'STATE'] and df[col].dtype == pl.Object:
                df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False))

        # Impute missing values
        df = self._impute_macro_data(df)
        
        # Create yield spread if both components exist
        if 'DGS10' in df.columns and 'DGS2' in df.columns:
            df = df.with_columns(
                (pl.col('DGS10') - pl.col('DGS2')).alias('YIELD_SPREAD')
            )
        
        # Get numerical columns for PCA
        numerical_cols = [c for c in df.columns 
                         if c not in ['date', 'STATE'] and 
                         df[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
        
        # Remove columns that are all null or constant
        valid_cols = []
        for col in numerical_cols:
            if df[col].null_count() == 0 and df[col].n_unique() > 1:
                valid_cols.append(col)
        
        print(f"Using {len(valid_cols)} valid columns for PCA")
        
        # Apply PCA by domain
        pca_results = []
        for domain, cols in self.pca_domains.items():
            existing_cols = [col for col in cols if col in valid_cols]
            if len(existing_cols) < 2:
                print(f"Skipping {domain} PCA - insufficient columns ({len(existing_cols)}/2)")
                continue
                
            print(f"Computing PCA for {domain} with columns: {existing_cols}")
            
            # Extract data for PCA
            pca_data = df.select(existing_cols).to_numpy()
            
            # Handle any remaining issues
            pca_data = np.nan_to_num(pca_data, nan=0, posinf=0, neginf=0)
            
            # Check for valid variance
            if np.var(pca_data, axis=0).sum() == 0:
                print(f"Skipping {domain} PCA - no variance in data")
                continue
                
            try:
                pca = PCA(n_components=1)
                pca_result = pca.fit_transform(pca_data)
                
                pca_df = pl.DataFrame({
                    f'PCA_{domain}': pca_result.flatten()
                })
                pca_results.append(pca_df)
                print(f"✓ {domain} PCA completed - explained variance: {pca.explained_variance_ratio_[0]:.3f}")
                
            except Exception as e:
                print(f"Error computing PCA for {domain}: {e}")
                continue
        
        # Combine original data with PCA results
        if pca_results:
            final_df = pl.concat([df] + pca_results, how='horizontal')
        else:
            final_df = df
            
        return final_df

    def _process_loan_chunk_memory_efficient(self, chunk_df, macro_data):
        """
        Process a chunk of loans using memory-efficient row generation
        """
        expanded_records = []
        
        for row in chunk_df.iter_rows(named=True):
            loan_id = row['LOAN_ID']
            orig_date = row['ORIG_DATE']
            state = row['STATE']
            max_age = row['LOAN_AGE']
            
            # Calculate quarterly origination date
            orig_quarter = orig_date.replace(day=1)
            orig_quarter = orig_quarter.replace(month=((orig_quarter.month - 1) // 3) * 3 + 1)
            
            # Calculate number of quarters (ensure integer)
            max_quarters = max(1, int((max_age + 2) // 3))
            
            # Generate records for each quarter
            for quarter_age in range(1, max_quarters + 1):
                # Calculate the quarter date
                quarter_offset_days = (quarter_age - 1) * 90
                quarter_date = orig_quarter + datetime.timedelta(days=quarter_offset_days)
                quarter_date = quarter_date.replace(day=1)
                quarter_date = quarter_date.replace(month=((quarter_date.month - 1) // 3) * 3 + 1)
                
                # Create record
                record = dict(row)  # Copy all original fields
                record['LOAN_AGE'] = quarter_age * 3  # Convert to monthly age
                record['Quarter_Join_Date'] = quarter_date
                
                expanded_records.append(record)
        
        # Convert to Polars DataFrame
        if not expanded_records:
            return pl.DataFrame()
        
        # Convert to DataFrame with proper type handling
        try:
            expanded_df = pl.DataFrame(expanded_records)
        except pl.ComputeError:
            # If schema inference fails, convert to pandas first then to polars
            import pandas as pd
            pandas_df = pd.DataFrame(expanded_records)
            expanded_df = pl.from_pandas(pandas_df)
        
        # Join with macro data
        result_df = expanded_df.join(
            macro_data,
            left_on=['Quarter_Join_Date', 'STATE'],
            right_on=['date', 'STATE'],
            how='left'
        )
        
        # Clean up temporary columns
        result_df = result_df.drop(['Quarter_Join_Date'], strict=False)
        
        return result_df

    def integrate_with_survival_chunked(self, survival_df, output_file="quarterly_time_varying_loan_survival_dataset.parquet"):
        """
        Memory-efficient integration processing data in chunks
        """
        print("Starting memory-efficient chunked integration...")
        
        # Validate required columns
        required_cols = ['LOAN_ID', 'ORIG_DATE', 'STATE', 'LOAN_AGE']
        missing_cols = [col for col in required_cols if col not in survival_df.columns]
        if missing_cols:
            raise ValueError(f"Survival DataFrame missing required columns: {missing_cols}")

        # Ensure ORIG_DATE is properly formatted
        if survival_df['ORIG_DATE'].dtype != pl.Date:
            try:
                survival_df = survival_df.with_columns(
                    pl.col('ORIG_DATE').str.to_date('%Y-%m-%d')
                )
            except:
                try:
                    survival_df = survival_df.with_columns(
                        pl.col('ORIG_DATE').str.to_date('%m/%d/%Y')
                    )
                except:
                    survival_df = survival_df.with_columns(
                        pl.col('ORIG_DATE').str.to_date()
                    )

        # Process macro data once
        print("Processing macro data...")
        processed_macro_df = self._transform_macro_data(self.macro_df)
        
        # Ensure STATE column exists and is standardized
        if 'State' in processed_macro_df.columns:
            processed_macro_df = processed_macro_df.rename({'State': 'STATE'})
        
        # Get all numeric columns from processed macro data
        all_macro_cols = [col for col in processed_macro_df.columns 
                         if col not in ['date', 'STATE'] and 
                         processed_macro_df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
        
        # Create national averages for missing states
        national_averages = processed_macro_df.group_by('date').agg(
            [pl.col(c).mean().alias(c) for c in all_macro_cols]
        )
        
        # Get unique states and prepare comprehensive macro dataset
        unique_states = survival_df.select('STATE').unique().to_series().to_list()
        
        macro_expanded_list = []
        for state in unique_states:
            state_macro = processed_macro_df.filter(pl.col('STATE') == state)
            if state_macro.is_empty():
                print(f"Warning: No macro data for state {state}, using national averages")
                state_macro = national_averages.with_columns(pl.lit(state).alias('STATE'))
            
            # Ensure consistent columns and fill missing values
            state_macro = state_macro.select(['date', 'STATE'] + all_macro_cols)
            for col in all_macro_cols:
                if col in state_macro.columns:
                    state_macro = state_macro.with_columns(
                        pl.col(col).forward_fill().backward_fill().fill_null(0)
                    )
                else:
                    state_macro = state_macro.with_columns(pl.lit(0.0).alias(col))
            
            macro_expanded_list.append(state_macro)
        
        final_macro_df = pl.concat(macro_expanded_list)
        
        # Clear intermediate data
        del processed_macro_df, macro_expanded_list
        gc.collect()
        
        # Process in chunks
        total_loans = len(survival_df)
        num_chunks = math.ceil(total_loans / self.chunk_size)
        print(f"Processing {total_loans:,} loans in {num_chunks} chunks of size {self.chunk_size:,}")
        
        # Initialize output file (delete if exists)
        if os.path.exists(output_file):
            os.remove(output_file)
        
        processed_count = 0
        start_time = time.time()
        
        for i in range(num_chunks):
            chunk_start = i * self.chunk_size
            chunk_end = min((i + 1) * self.chunk_size, total_loans)
            
            print(f"Processing chunk {i+1}/{num_chunks} (loans {chunk_start:,} to {chunk_end:,})...")
            
            # Get chunk
            chunk_df = survival_df.slice(chunk_start, chunk_end - chunk_start)
            
            # Process chunk
            chunk_result = self._process_loan_chunk_memory_efficient(chunk_df, final_macro_df)
            
            if not chunk_result.is_empty():
                # Fill any remaining nulls in numeric columns
                numeric_cols = [col for col in chunk_result.columns 
                               if chunk_result[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
                
                for col in numeric_cols:
                    if chunk_result[col].null_count() > 0:
                        mean_val = chunk_result[col].mean()
                        chunk_result = chunk_result.with_columns(
                            pl.col(col).fill_null(mean_val if mean_val is not None else 0)
                        )
                
                # Save chunk to file (append mode)
                if i == 0:
                    chunk_result.write_parquet(output_file)
                else:
                    # Read existing, concatenate, and write back
                    existing_df = pl.read_parquet(output_file)
                    combined_df = pl.concat([existing_df, chunk_result])
                    combined_df.write_parquet(output_file)
                
                processed_count += len(chunk_result)
                
                elapsed_time = time.time() - start_time
                rate = processed_count / elapsed_time if elapsed_time > 0 else 0
                print(f"  ✓ Chunk {i+1} completed: {len(chunk_result):,} records, "
                      f"Total: {processed_count:,}, Rate: {rate:.0f} records/sec")
            
            # Clean up
            del chunk_df, chunk_result
            gc.collect()
        
        # Load final result
        final_df = pl.read_parquet(output_file)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\nProcessing completed in {total_time:.2f} seconds!")
        print(f"Final dataset saved to '{output_file}' with shape {final_df.shape}")
        print(f"Total records: {len(final_df):,}")
        print(f"Unique loans: {final_df['LOAN_ID'].n_unique():,}")
        print(f"Processing rate: {len(final_df)/total_time:.0f} records/second")
        print("\nSample of the final dataset:")
        print(final_df.head(10))
        
        return final_df

def analyze_loan_distribution(survival_df):
    """Analyze loan age distribution to estimate processing time"""
    print("\n" + "="*60)
    print("LOAN AGE DISTRIBUTION ANALYSIS")
    print("="*60)
    
    print(f"Total loans: {len(survival_df):,}")
    print(f"Mean age: {survival_df['LOAN_AGE'].mean():.1f} months")
    print(f"Median age: {survival_df['LOAN_AGE'].median():.1f} months")
    print(f"Max age: {survival_df['LOAN_AGE'].max()} months")
    
    print("\nAge Distribution:")
    for p in [50, 75, 90, 95, 99]:
        age = survival_df['LOAN_AGE'].quantile(p/100)
        print(f"  {p:2d}th percentile: {age:6.1f} months")
    
    # Estimate quarterly records
    quarterly_records = survival_df.with_columns(
        ((pl.col('LOAN_AGE') + 2) // 3).alias('quarters')
    )['quarters'].sum()
    
    print(f"\nEstimated quarterly records: {quarterly_records:,}")
    
    # Time estimation
    records_per_hour = 500000  # Conservative estimate
    estimated_hours = quarterly_records / records_per_hour
    
    print(f"Estimated processing time: {estimated_hours:.1f} hours")
    
    if estimated_hours > 24:
        print("⚠️  WARNING: Processing will take more than 24 hours")
        print("   Consider running in smaller batches or overnight")
    elif estimated_hours > 8:
        print("ℹ️  INFO: Processing will take most of a workday")
    else:
        print("✅ GOOD: Processing should complete within a few hours")
    
    return quarterly_records, estimated_hours

def run_test_sample(survival_df, macro_df, sample_size=25000):
    """Run a test with sample data to get accurate timing"""
    print(f"\n{'='*60}")
    print(f"RUNNING TEST SAMPLE ({sample_size:,} loans)")
    print(f"{'='*60}")
    
    test_sample = survival_df.sample(n=sample_size, seed=42)
    
    processor = MemoryEfficientProcessor(macro_df, chunk_size=10000)
    
    start_time = time.time()
    test_result = processor.integrate_with_survival_chunked(test_sample, f"test_{sample_size}.parquet")
    test_time = time.time() - start_time
    
    # Calculate scaling factor
    total_loans = len(survival_df)
    scaling_factor = total_loans / sample_size
    estimated_full_time = test_time * scaling_factor / 3600  # Convert to hours
    
    print(f"\nTest Results:")
    print(f"Sample size: {sample_size:,} loans")
    print(f"Test time: {test_time:.1f} seconds")
    print(f"Output records: {len(test_result):,}")
    print(f"Processing rate: {len(test_result)/test_time:.0f} records/second")
    print(f"Estimated full processing time: {estimated_full_time:.1f} hours")
    
    return estimated_full_time

def main():
    """Main function with analysis and testing options"""
    # Use your FRED API key
    fred_fetcher = FREDDataFetcher(api_key='da6d2782ed02d7ebbdbfa87d9c4ece28')
    
    # Load survival data
    print("Loading survival data...")
    survival_df = df_main
    
    # Analyze the dataset first
    quarterly_records, estimated_hours = analyze_loan_distribution(survival_df)
    
    # Get user decision on how to proceed
    print(f"\nProcessing Options:")
    print(f"1. Run test sample first (recommended)")
    print(f"2. Process full dataset ({len(survival_df):,} loans)")
    print(f"3. Process subset (specify number of loans)")
    
    choice = input("Enter choice (1-3) or press Enter for test sample: ").strip()
    
    if choice == "2":
        # Process full dataset
        process_full = True
        test_first = False
    elif choice == "3":
        # Process subset
        try:
            subset_size = int(input("Enter number of loans to process: "))
            survival_df = survival_df.sample(n=subset_size, seed=42)
            process_full = True
            test_first = False
        except:
            print("Invalid input, defaulting to test sample")
            process_full = False
            test_first = True
    else:
        # Default: test sample first
        process_full = False
        test_first = True
    
    # Get unique states and date range
    states = survival_df['STATE'].unique().to_list()
    start_date = survival_df['ORIG_DATE'].min() - datetime.timedelta(days=90)
    end_date = survival_df['ORIG_DATE'].max() + datetime.timedelta(days=365*3)

    # Fetch or load macroeconomic data
    macro_df = fred_fetcher.fetch_all_data(states, start_date, end_date)

    if macro_df.is_empty():
        print("Failed to fetch macroeconomic data. Exiting.")
        return
    
    if test_first:
        # Run test sample
        estimated_time = run_test_sample(survival_df, macro_df, sample_size=25000)
        
        proceed = input(f"\nProceed with full dataset? (estimated {estimated_time:.1f} hours) [y/N]: ")
        if proceed.lower() != 'y':
            print("Stopping after test. Test results saved.")
            return
        process_full = True
    
    if process_full:
        # Process full dataset
        print(f"\nStarting full dataset processing...")
        processor = MemoryEfficientProcessor(macro_df, chunk_size=25000)
        
        output_file = "quarterly_time_varying_loan_survival_dataset.parquet"
        
        start_time = time.time()
        final_dataset = processor.integrate_with_survival_chunked(survival_df, output_file)
        total_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print("PROCESSING COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Final dataset: {len(final_dataset):,} records")
        print(f"Unique loans: {final_dataset['LOAN_ID'].n_unique():,}")
        print(f"Processing rate: {len(final_dataset)/total_time:.0f} records/second")
        
        # Show original loan information preservation
        original_cols = [c for c in survival_df.columns if c in final_dataset.columns]
        print(f"Preserved {len(original_cols)}/{len(survival_df.columns)} original columns")

if __name__ == "__main__":
    main()
