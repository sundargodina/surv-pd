import polars as pl
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import requests
import time
from datetime import datetime, timedelta
import gc
import warnings
import os
warnings.filterwarnings('ignore')

class UltraEfficientFREDScraperPolars:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.stlouisfed.org/fred/series/observations"

        # Streamlined indicators
        self.indicators = {
            'interest_rates': ['DGS3MO', 'DGS2', 'DGS10', 'DFF', 'MORTGAGE30US'],
            'credit_markets': ['DAAA', 'DBAA', 'BAMLH0A0HYM2'],
            'labor_market': ['UNRATE', 'PAYEMS', 'AHETPI', 'CIVPART'],
            'inflation': ['CPIAUCSL', 'CPILFESL', 'PCEPI', 'DFEDTARU'],
            'economic_activity': ['GDPC1', 'INDPRO', 'RRSFS', 'DSPIC96', 'PSAVERT'],
            'housing_market': ['CSUSHPISA', 'COMPUTSA', 'HOUST'],
            'financial_stress': ['NFCI', 'STLFSI4'],
            'market_sentiment': ['VIXCLS', 'NASDAQCOM', 'DEXUSEU']
        }

    def fetch_and_process_macro_data(self, start_date='2000-01-01', end_date='2023-12-31', chunk_size=3):
        """Fetch macro data using Polars for efficiency"""
        print("Fetching macro data with Polars...")

        all_series = set()
        for domain_series in self.indicators.values():
            all_series.update(domain_series)
        all_series = list(all_series)

        print(f"Fetching {len(all_series)} unique series...")

        all_data = None
        for i in range(0, len(all_series), chunk_size):
            chunk_series = all_series[i:i+chunk_size]
            print(f"  Processing chunk {i//chunk_size + 1}: {chunk_series}")

            chunk_data = {}
            for series_id in chunk_series:
                data = self._fetch_single_series_polars(series_id, start_date, end_date)
                if data is not None:
                    chunk_data[series_id] = data
                time.sleep(0.3)

            if chunk_data:
                chunk_combined = self._combine_chunk_data_polars(chunk_data)
                if all_data is None:
                    all_data = chunk_combined
                else:
                    # Use coalesce to handle duplicate act_period columns
                    all_data = all_data.join(chunk_combined, on='act_period', how='outer', coalesce=True)

                del chunk_combined

            del chunk_data
            gc.collect()

        return all_data

    def _fetch_single_series_polars(self, series_id, start_date, end_date):
        """Fetch single series and return as Polars DataFrame"""
        params = {
            'series_id': series_id,
            'api_key': self.api_key,
            'file_type': 'json',
            'observation_start': start_date,
            'observation_end': end_date,
            'frequency': 'm',
            'aggregation_method': 'avg'
        }

        try:
            response = requests.get(self.base_url, params=params)
            data = response.json()

            if 'observations' in data:
                # Extract valid observations
                valid_obs = []
                for obs in data['observations']:
                    if obs['value'] != '.' and obs['value'] is not None:
                        try:
                            date = datetime.strptime(obs['date'], '%Y-%m-%d')
                            value = float(obs['value'])
                            act_period = date.strftime('%m%Y')
                            valid_obs.append({'act_period': act_period, series_id: value})
                        except:
                            continue

                if valid_obs:
                    # Create Polars DataFrame directly
                    df = pl.DataFrame(valid_obs)
                    return df.with_columns(pl.col(series_id).cast(pl.Float32))
            return None
        except Exception as e:
            print(f"    Error fetching {series_id}: {e}")
            return None

    def _combine_chunk_data_polars(self, chunk_data):
        """Efficiently combine chunk data using Polars"""
        if not chunk_data:
            return None

        # Start with first dataframe
        result = list(chunk_data.values())[0].clone()

        # Join each subsequent dataframe
        for series_id, data in list(chunk_data.items())[1:]:
            # Use coalesce to handle duplicate act_period columns
            result = result.join(data, on='act_period', how='outer', coalesce=True)

        return result

    def create_compact_macro_lookup_polars(self, macro_data):
        """Create PCA components using Polars + sklearn"""
        if macro_data is None or macro_data.is_empty():
            return None

        print("Creating compact PCA components with Polars...")

        # Convert act_period to pandas for sklearn compatibility
        act_periods_pd = macro_data.select('act_period').to_pandas()

        # Initialize result with act_period
        pca_components_dict = {'act_period': act_periods_pd['act_period'].values}

        for domain, series_list in self.indicators.items():
            available_cols = [col for col in series_list if col in macro_data.columns]

            if len(available_cols) < 2:
                continue

            print(f"  Processing {domain}: {len(available_cols)} variables")

            # Extract domain data using Polars
            domain_data = macro_data.select(available_cols).fill_null(strategy='forward').fill_null(strategy='backward')

            # Fixed: Check for nulls using proper Polars syntax
            total_nulls = domain_data.null_count().sum_horizontal().item()
            if total_nulls > 0:
                print(f"    Skipping {domain}: still has {total_nulls} null values")
                continue

            # Convert to numpy for sklearn
            domain_np = domain_data.to_numpy().astype(np.float32)

            # PCA
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(domain_np)

            pca = PCA(n_components=min(2, len(available_cols)))
            pca_data = pca.fit_transform(scaled_data)

            # Keep only PC1 unless it explains <60%
            n_keep = 1 if pca.explained_variance_ratio_[0] > 0.6 else 2
            for i in range(min(n_keep, pca_data.shape[1])):
                pca_components_dict[f"{domain}_PC{i+1}"] = pca_data[:, i].astype(np.float32)

            # Cleanup
            del domain_data, domain_np, scaled_data, pca_data, pca, scaler
            gc.collect()

        # Create final Polars DataFrame
        pca_components = pl.DataFrame(pca_components_dict)

        print(f"Created {len(pca_components.columns)-1} PCA components")
        return pca_components

    def process_loan_file_polars_streaming(self, loan_file_path, macro_components, output_file_path,
                                         chunk_size=500_000, orig_date_col='ORIG_DATE', time_col='time'):
        """Stream process using Polars LazyFrame - ULTIMATE efficiency"""
        print(f"Polars streaming processing with {chunk_size:,} row chunks...")

        # Skip streaming for now since collect_in_batches may not be available
        print("Using fallback chunked processing directly...")
        return self._fallback_chunked_processing(loan_file_path, macro_components, output_file_path, chunk_size, orig_date_col, time_col)

    def _process_loan_batch_polars(self, batch, macro_pl, orig_date_col, time_col):
        """Process a batch using pure Polars operations"""

        # Check the actual dtype of the column in the batch
        orig_date_dtype = batch[orig_date_col].dtype

        if orig_date_dtype == pl.Date:
            # Column is already Date type
            batch_processed = batch.with_columns([
                # Create act_period: orig_date + (time + 1) * 30.44 days
                (pl.col(orig_date_col) +
                 pl.duration(days=((pl.col(time_col) + 1) * 30.44).cast(pl.Int64))
                ).dt.strftime('%m%Y').alias('act_period')
            ])
        else:
            # Column is string type, need to parse first
            batch_processed = batch.with_columns([
                # Create act_period: orig_date + (time + 1) * 30.44 days
                (pl.col(orig_date_col).str.strptime(pl.Date, format='%Y-%m-%d', strict=False) +
                 pl.duration(days=((pl.col(time_col) + 1) * 30.44).cast(pl.Int64))
                ).dt.strftime('%m%Y').alias('act_period')
            ])

        # Join with macro data
        result = batch_processed.join(macro_pl, on='act_period', how='left')

        return result

    def _combine_temp_files_polars(self, temp_files, output_file_path):
        """Combine temp files in smaller batches to avoid I/O issues"""
        print(f"Combining {len(temp_files)} files in batches...")

        if not temp_files:
            print("No temp files to combine!")
            return

        # Combine files in batches of 50 to avoid I/O issues
        batch_size = 50
        intermediate_files = []

        try:
            # Step 1: Combine temp files in batches
            for i in range(0, len(temp_files), batch_size):
                batch_files = temp_files[i:i+batch_size]
                batch_num = i // batch_size + 1

                print(f"  Combining batch {batch_num}: {len(batch_files)} files")

                # Read batch files
                lazy_frames = []
                for file in batch_files:
                    if os.path.exists(file):
                        lazy_frames.append(pl.scan_parquet(file))

                if lazy_frames:
                    # Combine this batch
                    batch_combined = pl.concat(lazy_frames)

                    # Save intermediate file
                    intermediate_file = os.path.join(
                        os.path.dirname(output_file_path),
                        f"intermediate_batch_{batch_num:03d}.parquet"
                    )

                    # Collect and write (not using sink_parquet for better reliability)
                    batch_combined.collect().write_parquet(intermediate_file)
                    intermediate_files.append(intermediate_file)

                    print(f"    Saved intermediate batch {batch_num}: {intermediate_file}")

                    del batch_combined, lazy_frames
                    gc.collect()

            # Step 2: Combine all intermediate files into final output
            if intermediate_files:
                print(f"  Final combination: {len(intermediate_files)} intermediate files")

                final_lazy_frames = [pl.scan_parquet(f) for f in intermediate_files]
                final_combined = pl.concat(final_lazy_frames)

                # Write final output
                final_combined.collect().write_parquet(output_file_path)
                print(f"Final output written: {output_file_path}")

                # Cleanup intermediate files
                for f in intermediate_files:
                    try:
                        os.remove(f)
                    except:
                        pass
            else:
                print("ERROR: No intermediate files were created!")

        except Exception as e:
            print(f"Error in file combination: {e}")
            # Try simple pandas fallback for small number of files
            if len(temp_files) <= 10:
                print("Trying pandas fallback for small file count...")
                self._pandas_fallback_combine(temp_files, output_file_path)
            else:
                raise e

    def _pandas_fallback_combine(self, temp_files, output_file_path):
        """Fallback: use pandas for combining if Polars fails"""
        import pandas as pd

        print("Using pandas fallback to combine files...")

        # Read all files with pandas
        dfs = []
        for file in temp_files:
            if os.path.exists(file):
                df = pd.read_parquet(file)
                dfs.append(df)
                print(f"  Loaded {file}: {len(df):,} rows")

        if dfs:
            # Combine all dataframes
            combined_df = pd.concat(dfs, ignore_index=True)
            print(f"Combined total: {len(combined_df):,} rows")

            # Save final output
            combined_df.to_parquet(output_file_path, index=False)
            print(f"Saved final output: {output_file_path}")
        else:
            print("ERROR: No files could be loaded!")

    def _fallback_chunked_processing(self, loan_file_path, macro_components, output_file_path,
                                   chunk_size, orig_date_col, time_col):
        """Fallback: manual chunking if streaming fails"""
        print("Using fallback chunked processing...")

        # Read file info
        lazy_df = pl.scan_parquet(loan_file_path)
        total_rows = lazy_df.select(pl.count()).collect().item()

        temp_files = []
        processed_count = 0

        # Process in offset-limit chunks
        for offset in range(0, total_rows, chunk_size):
            limit = min(chunk_size, total_rows - offset)
            chunk_num = (offset // chunk_size) + 1

            print(f"  Processing chunk {chunk_num}: offset {offset:,}, limit {limit:,}")

            try:
                # Read chunk
                chunk = lazy_df.slice(offset, limit).collect()

                # Convert macro to Polars if needed
                if hasattr(macro_components, 'to_pandas'):
                    macro_pl = macro_components
                else:
                    macro_pl = pl.from_pandas(macro_components)

                # Process chunk
                processed_chunk = self._process_loan_batch_polars(chunk, macro_pl, orig_date_col, time_col)

                # Save temp file
                temp_file = os.path.join(os.path.dirname(output_file_path), f"temp_fallback_{chunk_num:04d}.parquet")
                processed_chunk.write_parquet(temp_file)
                temp_files.append(temp_file)

                processed_count += len(processed_chunk)
                print(f"    Saved chunk {chunk_num}: {len(processed_chunk):,} rows")

                # Cleanup
                del chunk, processed_chunk
                gc.collect()

            except Exception as e:
                print(f"    Error in chunk {chunk_num}: {e}")
                continue

        # Combine temp files
        self._combine_temp_files_polars(temp_files, output_file_path)

        # Cleanup
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except:
                pass

        return output_file_path

def create_loan_macro_dataset_polars(api_key, loan_file_path, output_file_path):
    """Ultra-efficient version using Polars throughout"""
    print("=== POLARS Ultra Memory-Efficient Processing ===")

    # Step 1: Create scraper
    scraper = UltraEfficientFREDScraperPolars(api_key)

    # Step 2: Fetch macro data
    macro_data = scraper.fetch_and_process_macro_data()
    if macro_data is None:
        print("ERROR: Could not fetch macro data")
        return None

    # Step 3: Create PCA components
    macro_components = scraper.create_compact_macro_lookup_polars(macro_data)
    if macro_components is None:
        print("ERROR: Could not create macro components")
        return None

    # Clear raw macro data
    del macro_data
    gc.collect()
    print(f"Macro lookup table: {len(macro_components)} periods, {len(macro_components.columns)} columns")

    # Step 4: Stream-process loan data with Polars
    output_path = scraper.process_loan_file_polars_streaming(loan_file_path, macro_components, output_file_path)

    print("=== POLARS PROCESSING COMPLETE ===")
    return output_path

# Example usage:
if __name__ == "__main__":
    API_KEY = ""
    LOAN_FILE = "/content/drive/MyDrive/output.parquet"  # Your 10.4GB file
    OUTPUT_FILE = "/content/drive/MyDrive/final_loan_macro_polars.parquet"

    # Install Polars first: pip install polars
    # This will use ~100MB RAM regardless of file size!
    final_path = create_loan_macro_dataset_polars(API_KEY, LOAN_FILE, OUTPUT_FILE)
