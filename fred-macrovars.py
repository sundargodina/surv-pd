import pandas as pd
import numpy as np
from fredapi import Fred
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings('ignore')

FRED_API_KEY = 'e6afe32a2806088e5d190997847b1665'
RAW_MACRO_FILE = '/Users/sundargodina/Downloads/fred/raw_macro_data_by_state.csv'

fred = Fred(api_key=FRED_API_KEY)

# State-level macro variables (FRED state-level series)
state_variables = {
    'employment': {
        'unemployment_rate': '{state}UR',  # e.g., CAUR for California
        'nonfarm_payrolls': '{state}NA',   # e.g., CANA for California
        'labor_force': '{state}LF',        # e.g., CALF for California
    },
    'housing': {
        'house_price_index': '{state}STHPI',  # e.g., CASTHPI for California
        'housing_permits': '{state}BPPRIVSA', # e.g., CABPPRIVSA for California
    },
    'income': {
        'per_capita_income': '{state}PCPI',   # e.g., CAPCPI for California
        'total_wages': '{state}WAGES',        # e.g., CAWAGES for California
    },
    'activity': {
        'coincident_index': '{state}PHCI',    # e.g., CAPHCI for California
        'gdp': '{state}RGSP',                 # e.g., CARGSP for California
    }
}

# National variables that apply to all states (for context)
national_variables = {
    'rates': ['FEDFUNDS', 'GS10', 'GS2', 'TB3MS'],
    'credit': ['DBAA', 'DAAA', 'BAMLH0A0HYM2'],
    'inflation': ['CPIAUCSL', 'CPILFESL', 'PCEPI'],
    'stress': ['NFCI', 'VIXCLS', 'TEDRATE'],
    'sentiment': ['UMCSENT', 'SP500']
}

# State abbreviations for FRED codes
STATE_ABBREVIATIONS = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
    'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
    'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
    'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO',
    'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ',
    'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH',
    'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
    'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY',
    'District of Columbia': 'DC'
}

def fetch_series_with_retry(code, start_date, end_date, max_retries=3):
    """Fetch FRED series with retry logic, using dynamic start and end dates"""
    for attempt in range(max_retries):
        try:
            data = fred.get_series(code, observation_start=start_date, observation_end=end_date)
            time.sleep(0.1)  # Rate limiting
            return data
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"✗ {code}: Failed after {max_retries} attempts - {str(e)}")
                return pd.Series(dtype=float)
            time.sleep(1)  # Wait before retry
    return pd.Series(dtype=float)

def fetch_state_data(state_abbrev, start_date, end_date):
    """Fetch all data for a single state"""
    print(f"Fetching data for {state_abbrev} from {start_date} to {end_date}...")
    state_data = {}
    
    # Fetch state-specific variables
    for domain, variables in state_variables.items():
        for var_name, code_template in variables.items():
            code = code_template.format(state=state_abbrev)
            data = fetch_series_with_retry(code, start_date, end_date)
            if not data.empty:
                state_data[f"{state_abbrev}_{var_name}"] = data
                print(f"  ✓ {code}: {len(data)} observations")
            else:
                print(f"  ✗ {code}: No data")
    
    return state_abbrev, state_data

def fetch_macro_data_by_state(start_date, end_date):
    """Fetch macro data for all states with parallel processing"""
    print(f"Fetching state-level macro data from {start_date} to {end_date}...")
    
    # Get unique states from abbreviations
    states_to_fetch = list(STATE_ABBREVIATIONS.values())
    
    all_state_data = {}
    
    # Fetch state data in parallel (but with rate limiting)
    with ThreadPoolExecutor(max_workers=3) as executor:  # Limited workers for rate limiting
        future_to_state = {executor.submit(fetch_state_data, state, start_date, end_date): state for state in states_to_fetch}
        
        for future in as_completed(future_to_state):
            state = future_to_state[future]
            try:
                state_abbrev, state_data = future.result()
                all_state_data.update(state_data)
            except Exception as e:
                print(f"Error fetching data for {state}: {e}")
    
    # Fetch national variables
    print("Fetching national variables...")
    national_data = {}
    for domain, codes in national_variables.items():
        for code in codes:
            data = fetch_series_with_retry(code, start_date, end_date)
            if not data.empty:
                national_data[code] = data
                print(f"✓ {code}: {len(data)} observations")
            else:
                print(f"✗ {code}: No data")
    
    # Combine all data
    all_data = {**all_state_data, **national_data}
    
    if not all_data:
        raise ValueError("No macro data could be fetched!")
    
    # Convert to DataFrame
    combined_df = pd.DataFrame(all_data)
    
    # Convert to quarterly (standard quarterly end dates)
    # Ensure index is datetime before resample
    combined_df.index = pd.to_datetime(combined_df.index)
    quarterly_data = combined_df.resample('Q').last()
    
    # Save raw data
    quarterly_data.to_csv(RAW_MACRO_FILE)
    print(f"Saved {quarterly_data.shape[0]} quarterly observations with {quarterly_data.shape[1]} variables")
    
    return quarterly_data

class StateMacroSurvivalProcessor:
    def __init__(self, raw_file):
        self.raw_file = raw_file
        self.macro_data = None
        self.scalers = {}
        self.pca_models = {}
        self.state_data = {}  # Store processed data by state
        
    def load_and_process(self):
        """Load and process state-level macro data"""
        print("Loading and processing state-level macro data...")
        
        # Load raw data
        raw_data = pd.read_csv(self.raw_file, index_col=0, parse_dates=True)
        print(f"Raw data shape: {raw_data.shape}")
        
        # Step 1: Clean data - remove columns with >50% missing (more lenient for state data)
        missing_pct = raw_data.isnull().sum() / len(raw_data)
        good_cols = missing_pct[missing_pct <= 0.50].index.tolist()
        clean_data = raw_data[good_cols].copy()
        print(f"After removing high-missing columns: {clean_data.shape}")
        
        # Step 2: Separate state and national data
        state_cols = [col for col in clean_data.columns if any(col.startswith(state + '_') for state in STATE_ABBREVIATIONS.values())]
        national_cols = [col for col in clean_data.columns if col not in state_cols]
        
        print(f"State columns: {len(state_cols)}")
        print(f"National columns: {len(national_cols)}")
        
        # Step 3: Process national data
        national_processed = self._process_national_data(clean_data[national_cols])
        
        # Step 4: Process state data
        state_processed = self._process_state_data(clean_data[state_cols])
        
        # Step 5: Combine processed data
        final_data = pd.concat([national_processed, state_processed], axis=1)
        
        # Step 6: Final cleanup
        final_data = self._final_cleanup(final_data)
        
        self.macro_data = final_data
        print(f"Final macro data shape: {final_data.shape}")
        print(f"Null values: {final_data.isnull().sum().sum()}")
        
        return final_data
    
    def _process_national_data(self, df):
        """Process national-level variables"""
        print("Processing national variables...")
        
        # Fill missing values
        filled_df = self._fill_missing_values(df)
        
        # Create transformations
        trans_data = pd.DataFrame(index=df.index)
        
        # Keep key levels
        level_vars = ['FEDFUNDS', 'GS10', 'GS2', 'NFCI', 'VIXCLS', 'UMCSENT']
        for var in level_vars:
            if var in filled_df.columns:
                trans_data[var] = filled_df[var]
        
        # Create spreads
        if 'GS10' in filled_df.columns and 'GS2' in filled_df.columns:
            trans_data['yield_spread_10_2'] = filled_df['GS10'] - filled_df['GS2']
        
        if 'DBAA' in filled_df.columns and 'DAAA' in filled_df.columns:
            trans_data['credit_spread'] = filled_df['DBAA'] - filled_df['DAAA']
        
        # Create inflation rates
        inflation_vars = ['CPIAUCSL', 'CPILFESL', 'PCEPI']
        for var in inflation_vars:
            if var in filled_df.columns:
                inflation = filled_df[var].pct_change(periods=4) * 100
                trans_data[f'{var}_inflation'] = inflation
        
        # Create returns
        if 'SP500' in filled_df.columns:
            trans_data['SP500_return'] = filled_df['SP500'].pct_change(periods=1) * 100
        
        # Fill any new nulls
        for col in trans_data.columns:
            if trans_data[col].isnull().any():
                trans_data[col] = trans_data[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return trans_data
    
    def _process_state_data(self, df):
        """Process state-level variables with state-specific PCA"""
        print("Processing state variables...")
        
        # Fill missing values
        filled_df = self._fill_missing_values(df)
        
        # Group by state
        states = list(STATE_ABBREVIATIONS.values())
        all_state_processed = pd.DataFrame(index=df.index)
        
        for state in states:
            state_cols = [col for col in filled_df.columns if col.startswith(f'{state}_')]
            
            if len(state_cols) < 2:
                continue
            
            state_df = filled_df[state_cols].copy()
            
            # Create state-specific transformations
            state_processed = self._create_state_transformations(state_df, state)
            
            # Apply PCA to state data
            state_pca = self._apply_state_pca(state_processed, state)
            
            # Add to combined data
            all_state_processed = pd.concat([all_state_processed, state_pca], axis=1)
        
        return all_state_processed
    
    def _create_state_transformations(self, state_df, state):
        """Create transformations for a specific state"""
        trans_data = pd.DataFrame(index=state_df.index)
        
        # Map variable types for this state
        for col in state_df.columns:
            base_name = col.replace(f'{state}_', '')
            
            # Keep levels for certain variables
            if base_name in ['unemployment_rate', 'coincident_index']:
                trans_data[col] = state_df[col]
            
            # Create growth rates for economic activity
            elif base_name in ['nonfarm_payrolls', 'labor_force', 'total_wages', 'gdp', 'housing_permits']:
                growth = state_df[col].pct_change(periods=4) * 100
                trans_data[f'{col}_growth'] = growth
            
            # Create inflation for price indices
            elif base_name in ['house_price_index', 'per_capita_income']:
                inflation = state_df[col].pct_change(periods=4) * 100
                trans_data[f'{col}_inflation'] = inflation
            
            else:
                # Default: keep as level
                trans_data[col] = state_df[col]
        
        # Fill nulls from transformations
        for col in trans_data.columns:
            if trans_data[col].isnull().any():
                trans_data[col] = trans_data[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return trans_data
    
    def _apply_state_pca(self, state_df, state):
        """Apply PCA to state-specific data"""
        if state_df.shape[1] < 2:
            return state_df
        
        # Remove zero variance columns
        state_df = state_df.loc[:, state_df.var() != 0]
        
        if state_df.shape[1] < 2:
            return state_df
        
        # Standardize
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(state_df)
        
        # Apply PCA (max 3 components for state data)
        n_components = min(3, state_df.shape[1])
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(scaled_data)
        
        # Create result DataFrame
        pca_data = pd.DataFrame(index=state_df.index)
        
        for i in range(n_components):
            pca_data[f'{state}_PC{i+1}'] = pca_result[:, i]
        
        # Store models
        self.scalers[state] = scaler
        self.pca_models[state] = pca
        
        # Add key original variables
        key_vars = [f'{state}_unemployment_rate', f'{state}_coincident_index']
        for var in key_vars:
            if var in state_df.columns:
                pca_data[var] = state_df[var]
        
        print(f"{state}: {state_df.shape[1]} vars -> {n_components} PCs "
              f"(var explained: {pca.explained_variance_ratio_.sum():.2f})")
        
        return pca_data
    
    def _fill_missing_values(self, df):
        """Fill missing values - more aggressive for state data"""
        print(f"Filling missing values for {df.shape[1]} variables...")
        
        filled_df = df.copy()
        
        for col in filled_df.columns:
            series = filled_df[col]
            
            # Forward fill (up to 4 quarters for state data)
            filled_df[col] = series.fillna(method='ffill', limit=4)
            
            # Backward fill (up to 4 quarters)
            filled_df[col] = filled_df[col].fillna(method='bfill', limit=4)
            
            # Linear interpolation
            filled_df[col] = filled_df[col].interpolate(method='linear')
            
            # Final fallback: median imputation
            if filled_df[col].isnull().any():
                median_val = filled_df[col].median()
                if pd.isna(median_val):
                    # If median is also null, use 0
                    filled_df[col] = filled_df[col].fillna(0)
                else:
                    filled_df[col] = filled_df[col].fillna(median_val)
        
        return filled_df
    
    def _final_cleanup(self, df):
        """Final cleanup - GUARANTEE NO NULLS"""
        print("Final cleanup...")
        
        clean_df = df.copy()
        
        # Handle numeric variables only
        numeric_cols = clean_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if clean_df[col].isnull().any():
                median_val = clean_df[col].median()
                if pd.isna(median_val):
                    clean_df[col] = clean_df[col].fillna(0)
                else:
                    clean_df[col] = clean_df[col].fillna(median_val)
        
        # Remove zero variance columns
        for col in numeric_cols:
            if clean_df[col].var() == 0:
                clean_df = clean_df.drop(columns=[col])
                print(f"Removed zero variance column: {col}")
        
        # Final null check
        null_count = clean_df.isnull().sum().sum()
        if null_count > 0:
            print(f"WARNING: {null_count} nulls remaining - applying final imputation")
            imputer = SimpleImputer(strategy='median')
            numeric_cols = clean_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                clean_df[numeric_cols] = imputer.fit_transform(clean_df[numeric_cols])
        
        return clean_df
    
    def map_to_quarter(self, date):
        """Map date to standard quarter end"""
        if pd.isna(date):
            return None
        
        dt = pd.to_datetime(date)
        year = dt.year
        quarter = dt.quarter
        
        if quarter == 1:
            return pd.Timestamp(f'{year}-03-31')
        elif quarter == 2:
            return pd.Timestamp(f'{year}-06-30')
        elif quarter == 3:
            return pd.Timestamp(f'{year}-09-30')
        else:
            return pd.Timestamp(f'{year}-12-31')
    
    def _fix_data_types_for_parquet(self, df):
        """Fix data types before saving to Parquet"""
        print("Fixing data types for Parquet compatibility...")
        
        fixed_df = df.copy()
        protected_cols = [
            'quarter', 'macro_quarter', 'STATE',
            'PURPOSE', 'PROP', 'OCC_STAT', 
            'FIRST_FLAG', 'CHANNEL', 'ifrs9_stage', 
            'credit_score_bucket', 'ltv_bucket'
        ]
        
        for col in fixed_df.columns:
            if col in protected_cols:
                continue
            if fixed_df[col].dtype == 'object':
                try:
                    fixed_df[col] = pd.to_numeric(fixed_df[col], errors='coerce')
                    if fixed_df[col].isnull().any():
                        fixed_df[col] = fixed_df[col].fillna(0)
                except:
                    fixed_df[col] = fixed_df[col].astype(str)
        
        # Handle remaining nulls
        for col in fixed_df.columns:
            if fixed_df[col].isnull().any():
                if fixed_df[col].dtype in ['float64', 'int64']:
                    fixed_df[col] = fixed_df[col].fillna(0)
                elif fixed_df[col].dtype == 'object':
                    fixed_df[col] = fixed_df[col].fillna('unknown')
                else:
                    fixed_df[col] = fixed_df[col].fillna(pd.Timestamp('1900-01-01'))
        
        return fixed_df
    
    def integrate_with_survival(self, survival_df):
        """Integrate with survival data using state-specific macro conditions"""
        print("Integrating with survival data using state-level macro data...")

        if self.macro_data is None:
            raise ValueError("Must process macro data first! Call load_and_process() first.")

        # Preserve original columns
        if 'quarter' in survival_df.columns:
            original_quarter = survival_df['quarter'].copy()
        else:
            original_quarter = None

        # Map origination dates to macro quarters
        survival_df['macro_quarter'] = survival_df['ORIG_DATE'].apply(self.map_to_quarter)
        
        # Get state abbreviations for survival data
        if 'STATE' in survival_df.columns:
            # Map state names to abbreviations if needed
            survival_df['state_abbrev'] = survival_df['STATE'].map(
                lambda x: STATE_ABBREVIATIONS.get(x, x) if x in STATE_ABBREVIATIONS else x
            )
        else:
            print("WARNING: No STATE column found in survival data. Using national averages only.")
            survival_df['state_abbrev'] = 'NATIONAL'

        # Create state-specific macro features
        merged_data = []
        
        for state in survival_df['state_abbrev'].unique():
            if pd.isna(state):
                continue
                
            state_mask = survival_df['state_abbrev'] == state
            state_survival = survival_df[state_mask].copy()
            
            # Get relevant macro columns for this state
            if state != 'NATIONAL':
                state_macro_cols = [col for col in self.macro_data.columns 
                                  if col.startswith(f'{state}_') or not any(col.startswith(f'{s}_') 
                                  for s in STATE_ABBREVIATIONS.values())]
            else:
                # Use national variables only
                state_macro_cols = [col for col in self.macro_data.columns 
                                  if not any(col.startswith(f'{s}_') for s in STATE_ABBREVIATIONS.values())]
            
            # Select relevant macro data
            relevant_macro = self.macro_data[state_macro_cols]
            
            # Merge with state survival data
            state_merged = state_survival.merge(
                relevant_macro,
                left_on='macro_quarter',
                right_index=True,
                how='left'
            )
            
            merged_data.append(state_merged)
        
        # Combine all states
        merged = pd.concat(merged_data, ignore_index=True)
        
        # Re-attach quarter column if it existed
        if original_quarter is not None:
            # Need to ensure the index aligns or re-index appropriately
            # If survival_df was already indexed, this might need more robust handling
            merged['quarter'] = original_quarter.reindex(survival_df.index).loc[merged.index]
        
        # Handle missing macro data
        macro_cols = [col for col in merged.columns if col not in survival_df.columns and '_PC' in col or any(v in col for v in ['unemployment_rate', 'coincident_index', 'yield_spread', 'credit_spread', 'inflation', 'return', 'WAGES', 'gdp', 'LF', 'BPPRIVSA', 'PCPI', 'HPI', 'NA', 'UR', 'PHCI', 'GS', 'FEDFUNDS', 'DBAA', 'DAAA', 'BAMLH0A0HYM2', 'CPIAUCSL', 'CPILFESL', 'PCEPI', 'NFCI', 'VIXCLS', 'TEDRATE', 'UMCSENT', 'SP500'])]
        macro_start = self.macro_data.index.min()
        macro_end = self.macro_data.index.max()
        
        for col in macro_cols:
            if merged[col].isnull().any():
                # For dates before macro start, use first available values
                before_mask = merged['macro_quarter'] < macro_start
                if before_mask.any() and col in self.macro_data.columns:
                    first_val = self.macro_data[col].iloc[0]
                    merged.loc[before_mask, col] = first_val
                
                # For dates after macro end, use last available values
                after_mask = merged['macro_quarter'] > macro_end
                if after_mask.any() and col in self.macro_data.columns:
                    last_val = self.macro_data[col].iloc[-1]
                    merged.loc[after_mask, col] = last_val
                
                # For remaining nulls, use median or 0
                if merged[col].isnull().any():
                    if col in self.macro_data.columns:
                        median_val = self.macro_data[col].median()
                        merged[col] = merged[col].fillna(median_val if not pd.isna(median_val) else 0)
                    else:
                        merged[col] = merged[col].fillna(0)
        
        # Fix data types
        merged = self._fix_data_types_for_parquet(merged)
        
        # Final cleanup
        null_count = merged.isnull().sum().sum()
        print(f"Integration complete. Shape: {merged.shape}")
        print(f"Null values: {null_count}")
        
        if null_count > 0:
            print("WARNING: Applying final null cleanup")
            for col in merged.columns:
                if merged[col].isnull().any():
                    if merged[col].dtype in ['float64', 'int64']:
                        merged[col] = merged[col].fillna(0)
                    elif merged[col].dtype == 'object':
                        merged[col] = merged[col].fillna('unknown')
                    else:
                        merged[col] = merged[col].fillna(pd.Timestamp('1900-01-01'))
        
        return merged
    
    def get_summary(self):
        """Get processing summary"""
        if self.macro_data is None:
            return "No processed data available"
        
        # Count features by type
        state_features = len([c for c in self.macro_data.columns if any(c.startswith(f'{s}_') for s in STATE_ABBREVIATIONS.values())])
        national_features = len(self.macro_data.columns) - state_features
        pca_features = len([c for c in self.macro_data.columns if '_PC' in c])
        
        summary = {
            'total_features': len(self.macro_data.columns),
            'state_features': state_features,
            'national_features': national_features,
            'pca_features': pca_features,
            'observations': len(self.macro_data),
            'date_range': f"{self.macro_data.index.min()} to {self.macro_data.index.max()}",
            'null_values': self.macro_data.isnull().sum().sum(),
            'states_with_pca': len(self.pca_models),
        }
        
        return summary

# Main execution
if __name__ == "__main__":
    # Load survival data first to determine date range
    print("\n=== LOADING SURVIVAL DATA TO DETERMINE DATE RANGE ===")
    survival_df = pd.read_parquet('/Users/sundargodina/Downloads/survival_files/survival_all.parquet')
    
    if 'ORIG_DATE' not in survival_df.columns:
        raise ValueError("survival_df must contain an 'ORIG_DATE' column for dynamic date range determination.")
    
    # Ensure ORIG_DATE is datetime
    survival_df['ORIG_DATE'] = pd.to_datetime(survival_df['ORIG_DATE'])
    
    # Calculate min and max origination dates
    min_orig_date = survival_df['ORIG_DATE'].min()
    max_orig_date = survival_df['ORIG_DATE'].max()
    
    # Determine effective start date for macro data
    # Need at least 4 quarters of lookback for pct_change(periods=4)
    # Adding a buffer to be safe, e.g., 1 year prior to the earliest origination
    effective_start_date = (min_orig_date - pd.DateOffset(years=1)).strftime('%Y-%m-%d')
    effective_end_date = max_orig_date.strftime('%Y-%m-%d')
    
    print(f"Determined Macro Fetch Start Date: {effective_start_date}")
    print(f"Determined Macro Fetch End Date: {effective_end_date}")

    # Fetch fresh state-level data using dynamic dates
    macro_data = fetch_macro_data_by_state(effective_start_date, effective_end_date)
    
    # Initialize processor
    processor = StateMacroSurvivalProcessor(RAW_MACRO_FILE)
    

    # Process macro data
    processed_macro_data = processor.load_and_process() # This will load the data saved by fetch_macro_data_by_state
    
    # Get summary
    summary = processor.get_summary()
    print("\n=== PROCESSING SUMMARY ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
