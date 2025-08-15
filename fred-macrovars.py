# Simplified Domain-Specific PCA on Macroeconomic Data for Survival Analysis

import pandas as pd

import numpy as np

from fredapi import Fred

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer

import warnings

warnings.filterwarnings('ignore')



FRED_API_KEY = 'putyourkeyhere'

START_DATE = '2012-01-01'

END_DATE = '2023-12-31'

RAW_MACRO_FILE = '/Users/sundargodina/Downloads/fred/raw_macro_data.csv'



fred = Fred(api_key=FRED_API_KEY)



# Core macro variables based on IFRS 9 research

core_variables = {

    'rates': ['FEDFUNDS', 'GS10', 'GS2', 'TB3MS'],

    'credit': ['DBAA', 'DAAA', 'BAMLH0A0HYM2'],

    'employment': ['UNRATE', 'PAYEMS', 'ICSA'],

    'inflation': ['CPIAUCSL', 'CPILFESL', 'PCEPI'],

    'activity': ['GDPC1', 'INDPRO', 'CFNAI'],

    'housing': ['HOUST', 'CSUSHPISA', 'MORTGAGE30US'],

    'stress': ['NFCI', 'VIXCLS', 'TEDRATE'],

    'sentiment': ['UMCSENT', 'SP500']

}



def fetch_macro_data():

    """Fetch macro data with robust error handling"""

    print("Fetching macro data...")

    all_data = pd.DataFrame()

    

    for domain, codes in core_variables.items():

        for code in codes:

            try:

                print(f"Fetching {code}...")

                data = fred.get_series(code, observation_start=START_DATE, observation_end=END_DATE)

                if not data.empty:

                    all_data[code] = data

                    print(f"✓ {code}: {len(data)} observations")

                else:

                    print(f"✗ {code}: No data")

            except Exception as e:

                print(f"✗ {code}: Failed - {str(e)}")

    

    if all_data.empty:

        raise ValueError("No macro data could be fetched!")

    

    # Convert to quarterly (standard quarterly end dates)

    quarterly_data = all_data.resample('Q').last()

    

    # Save raw data

    quarterly_data.to_csv(RAW_MACRO_FILE)

    print(f"Saved {quarterly_data.shape[0]} quarterly observations with {quarterly_data.shape[1]} variables")

    

    return quarterly_data



class MacroSurvivalProcessor:

    def __init__(self, raw_file):

        self.raw_file = raw_file

        self.macro_data = None

        self.scalers = {}

        self.pca_models = {}

        

    def load_and_process(self):

        """Load and process macro data - NO NULL VALUES GUARANTEED"""

        print("Loading and processing macro data...")

        

        # Load raw data

        raw_data = pd.read_csv(self.raw_file, index_col=0, parse_dates=True)

        print(f"Raw data shape: {raw_data.shape}")

        

        # Step 1: Clean data - remove columns with >20% missing

        missing_pct = raw_data.isnull().sum() / len(raw_data)

        good_cols = missing_pct[missing_pct <= 0.20].index.tolist()

        clean_data = raw_data[good_cols].copy()

        print(f"After removing high-missing columns: {clean_data.shape}")

        

        # Step 2: Fill missing values systematically

        clean_data = self._fill_missing_values(clean_data)

        

        # Step 3: Create transformations

        transformed_data = self._create_transformations(clean_data)

        

        # Step 4: Apply PCA by domain

        pca_features = self._apply_domain_pca(transformed_data)

        

        # Step 5: Create economic indicators

        final_data = self._create_indicators(pca_features)

        

        # Step 6: Final cleanup - ENSURE NO NULLS

        final_data = self._final_cleanup(final_data)

        

        self.macro_data = final_data

        print(f"Final macro data shape: {final_data.shape}")

        print(f"Null values: {final_data.isnull().sum().sum()}")

        

        return final_data

    

    def _fill_missing_values(self, df):

        """Fill missing values - NO NULLS ALLOWED"""

        print("Filling missing values...")

        

        filled_df = df.copy()

        

        for col in filled_df.columns:

            series = filled_df[col]

            

            # Forward fill (up to 2 quarters)

            filled_df[col] = series.fillna(method='ffill', limit=2)

            

            # Backward fill (up to 2 quarters)

            filled_df[col] = filled_df[col].fillna(method='bfill', limit=2)

            

            # Linear interpolation for remaining gaps

            filled_df[col] = filled_df[col].interpolate(method='linear')

            

            # Final fallback: median imputation

            if filled_df[col].isnull().any():

                median_val = filled_df[col].median()

                filled_df[col] = filled_df[col].fillna(median_val)

        

        return filled_df

    

    def _create_transformations(self, df):

        """Create economic transformations"""

        print("Creating transformations...")

        

        trans_data = pd.DataFrame(index=df.index)

        

        # Keep key levels

        level_vars = ['UNRATE', 'FEDFUNDS', 'GS10', 'GS2', 'CFNAI', 'NFCI', 'VIXCLS', 'UMCSENT']

        for var in level_vars:

            if var in df.columns:

                trans_data[var] = df[var]

        

        # Create growth rates (4-quarter)

        growth_vars = ['GDPC1', 'INDPRO', 'PAYEMS', 'HOUST']

        for var in growth_vars:

            if var in df.columns:

                growth = df[var].pct_change(periods=4) * 100

                trans_data[f'{var}_growth'] = growth

        

        # Create inflation rates (4-quarter)

        inflation_vars = ['CPIAUCSL', 'CPILFESL', 'PCEPI', 'CSUSHPISA']

        for var in inflation_vars:

            if var in df.columns:

                inflation = df[var].pct_change(periods=4) * 100

                trans_data[f'{var}_inflation'] = inflation

        

        # Create returns (1-quarter)

        return_vars = ['SP500']

        for var in return_vars:

            if var in df.columns:

                returns = df[var].pct_change(periods=1) * 100

                trans_data[f'{var}_return'] = returns

        

        # Create spreads

        if 'GS10' in df.columns and 'GS2' in df.columns:

            trans_data['yield_spread_10_2'] = df['GS10'] - df['GS2']

        

        if 'DBAA' in df.columns and 'DAAA' in df.columns:

            trans_data['credit_spread'] = df['DBAA'] - df['DAAA']

        

        if 'MORTGAGE30US' in df.columns and 'GS10' in df.columns:

            trans_data['mortgage_spread'] = df['MORTGAGE30US'] - df['GS10']

        

        # Fill any new nulls from transformations

        for col in trans_data.columns:

            if trans_data[col].isnull().any():

                # Use forward/backward fill

                trans_data[col] = trans_data[col].fillna(method='ffill').fillna(method='bfill')

                # Final fallback

                if trans_data[col].isnull().any():

                    trans_data[col] = trans_data[col].fillna(0)

        

        return trans_data

    

    def _apply_domain_pca(self, df):

        """Apply domain-specific PCA"""

        print("Applying domain PCA...")

        

        # Define domain mappings for transformed variables

        domain_map = {

            'rates': ['FEDFUNDS', 'GS10', 'GS2', 'yield_spread_10_2'],

            'credit': ['DBAA', 'DAAA', 'credit_spread', 'mortgage_spread'],

            'employment': ['UNRATE', 'PAYEMS_growth', 'ICSA'],

            'inflation': ['CPIAUCSL_inflation', 'CPILFESL_inflation', 'PCEPI_inflation'],

            'activity': ['GDPC1_growth', 'INDPRO_growth', 'CFNAI'],

            'housing': ['HOUST_growth', 'CSUSHPISA_inflation'],

            'stress': ['NFCI', 'VIXCLS', 'TEDRATE'],

            'sentiment': ['UMCSENT', 'SP500_return']

        }

        

        pca_data = pd.DataFrame(index=df.index)

        

        for domain, vars_list in domain_map.items():

            # Find available variables

            available_vars = [v for v in vars_list if v in df.columns]

            

            if len(available_vars) < 2:

                print(f"Skipping {domain} - insufficient variables")

                continue

            

            # Get domain data

            domain_df = df[available_vars].copy()

            

            # Remove zero variance columns

            domain_df = domain_df.loc[:, domain_df.var() != 0]

            

            if domain_df.shape[1] < 2:

                print(f"Skipping {domain} - insufficient variance")

                continue

            

            # Standardize

            scaler = StandardScaler()

            scaled_data = scaler.fit_transform(domain_df)

            

            # Apply PCA (2 components max)

            n_components = min(2, domain_df.shape[1])

            pca = PCA(n_components=n_components)

            pca_result = pca.fit_transform(scaled_data)

            

            # Store results

            for i in range(n_components):

                pca_data[f'{domain}_PC{i+1}'] = pca_result[:, i]

            

            # Store models

            self.scalers[domain] = scaler

            self.pca_models[domain] = pca

            

            print(f"{domain}: {len(available_vars)} vars -> {n_components} PCs "

                  f"(var explained: {pca.explained_variance_ratio_.sum():.2f})")

        

        # Add key original variables

        key_vars = ['UNRATE', 'FEDFUNDS', 'GS10', 'CFNAI', 'NFCI', 'yield_spread_10_2']

        for var in key_vars:

            if var in df.columns:

                pca_data[var] = df[var]

        

        return pca_data

    

    def _create_indicators(self, df):

        """Create economic indicators"""

        print("Creating economic indicators...")

        

        indicator_data = df.copy()

        

        # Economic regimes (convert to numeric instead of categorical)

        if 'UNRATE' in df.columns:

            # Create numerical regime indicators

            indicator_data['unemployment_regime_low'] = (df['UNRATE'] <= 4).astype(int)

            indicator_data['unemployment_regime_normal'] = ((df['UNRATE'] > 4) & (df['UNRATE'] <= 6)).astype(int)

            indicator_data['unemployment_regime_high'] = ((df['UNRATE'] > 6) & (df['UNRATE'] <= 8)).astype(int)

            indicator_data['unemployment_regime_crisis'] = (df['UNRATE'] > 8).astype(int)

        

        if 'FEDFUNDS' in df.columns:

            # Create numerical regime indicators

            indicator_data['interest_rate_regime_zero'] = (df['FEDFUNDS'] <= 1).astype(int)

            indicator_data['interest_rate_regime_low'] = ((df['FEDFUNDS'] > 1) & (df['FEDFUNDS'] <= 3)).astype(int)

            indicator_data['interest_rate_regime_normal'] = ((df['FEDFUNDS'] > 3) & (df['FEDFUNDS'] <= 6)).astype(int)

            indicator_data['interest_rate_regime_high'] = (df['FEDFUNDS'] > 6).astype(int)

        

        if 'yield_spread_10_2' in df.columns:

            # Create numerical regime indicators

            indicator_data['yield_curve_regime_inverted'] = (df['yield_spread_10_2'] < 0).astype(int)

            indicator_data['yield_curve_regime_flat'] = ((df['yield_spread_10_2'] >= 0) & (df['yield_spread_10_2'] <= 1)).astype(int)

            indicator_data['yield_curve_regime_normal'] = ((df['yield_spread_10_2'] > 1) & (df['yield_spread_10_2'] <= 2)).astype(int)

            indicator_data['yield_curve_regime_steep'] = (df['yield_spread_10_2'] > 2).astype(int)

        

        # Economic stress composite

        stress_components = []

        if 'NFCI' in df.columns:

            stress_components.append((df['NFCI'] > 0).astype(int))

        if 'UNRATE' in df.columns:

            stress_components.append((df['UNRATE'] > df['UNRATE'].median()).astype(int))

        if 'VIXCLS' in df.columns:

            stress_components.append((df['VIXCLS'] > df['VIXCLS'].quantile(0.75)).astype(int))

        

        if stress_components:

            indicator_data['economic_stress_composite'] = sum(stress_components) / len(stress_components)

        

        # Credit conditions (convert to numeric)

        if 'credit_spread' in df.columns:

            indicator_data['credit_conditions_good'] = (df['credit_spread'] <= df['credit_spread'].quantile(0.33)).astype(int)

            indicator_data['credit_conditions_moderate'] = ((df['credit_spread'] > df['credit_spread'].quantile(0.33)) & 

                                                          (df['credit_spread'] <= df['credit_spread'].quantile(0.67))).astype(int)

            indicator_data['credit_conditions_stressed'] = (df['credit_spread'] > df['credit_spread'].quantile(0.67)).astype(int)

        

        # Yield curve inversion duration

        if 'yield_spread_10_2' in df.columns:

            indicator_data['yield_curve_inversion_duration'] = (

                df['yield_spread_10_2'] < 0

            ).astype(int).rolling(4, min_periods=1).sum()

        

        return indicator_data

    

    def _final_cleanup(self, df):

        """Final cleanup - GUARANTEE NO NULLS"""

        print("Final cleanup...")

        

        clean_df = df.copy()

        

        # Handle numeric variables only (no categorical)

        numeric_cols = clean_df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:

            if clean_df[col].isnull().any():

                # Use median for final imputation

                median_val = clean_df[col].median()

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

            

            # Final aggressive imputation

            imputer = SimpleImputer(strategy='median')

            numeric_cols = clean_df.select_dtypes(include=[np.number]).columns

            if len(numeric_cols) > 0:

                clean_df[numeric_cols] = imputer.fit_transform(clean_df[numeric_cols])

        

        return clean_df

    

    def map_to_quarter(self, date):

        """Map date to standard quarter end - MODIFIED FOR STANDARD QUARTERLY MAPPING"""

        if pd.isna(date):

            return None

        

        dt = pd.to_datetime(date)

        year = dt.year

        quarter = dt.quarter

        

        # Standard quarterly mapping - map to actual quarter end

        if quarter == 1:

            return pd.Timestamp(f'{year}-03-31')

        elif quarter == 2:

            return pd.Timestamp(f'{year}-06-30')

        elif quarter == 3:

            return pd.Timestamp(f'{year}-09-30')

        else:  # quarter == 4

            return pd.Timestamp(f'{year}-12-31')

    

    def _fix_data_types_for_parquet(self, df):

        """Fix data types before saving to Parquet"""

        print("Fixing data types for Parquet compatibility...")

        

        fixed_df = df.copy()

        protected_cols = [
        'quarter', 'macro_quarter',
        'PURPOSE', 'STATE', 'PROP', 'OCC_STAT', 
        'FIRST_FLAG', 'CHANNEL', 'ifrs9_stage', 
        'credit_score_bucket', 'ltv_bucket'
        ]

        

        # Convert object columns to appropriate types

        for col in fixed_df.columns:

            if col in protected_cols:

                continue  

            if fixed_df[col].dtype == 'object':

                # Try to convert to datetime first

                if 'DATE' in col.upper() or 'DT' in col.upper():

                    try:

                        fixed_df[col] = pd.to_datetime(fixed_df[col], errors='coerce')

                        print(f"Converted {col} to datetime")

                    except:

                        # If datetime conversion fails, convert to string

                        fixed_df[col] = fixed_df[col].astype(str)

                        print(f"Converted {col} to string")

                else:

                    # For non-date object columns, try to convert to numeric

                    try:

                        fixed_df[col] = pd.to_numeric(fixed_df[col], errors='coerce')

                        # Fill any nulls created by conversion

                        if fixed_df[col].isnull().any():

                            fixed_df[col] = fixed_df[col].fillna(0)

                        print(f"Converted {col} to numeric")

                    except:

                        # If numeric conversion fails, keep as string

                        fixed_df[col] = fixed_df[col].astype(str)

                        print(f"Converted {col} to string")

        

        # Handle any remaining nulls

        for col in fixed_df.columns:

            if fixed_df[col].isnull().any():

                if fixed_df[col].dtype in ['float64', 'int64']:

                    fixed_df[col] = fixed_df[col].fillna(0)

                elif fixed_df[col].dtype == 'object':

                    fixed_df[col] = fixed_df[col].fillna('unknown')

                else:

                    # For datetime columns

                    fixed_df[col] = fixed_df[col].fillna(pd.Timestamp('1900-01-01'))

        

        return fixed_df

    

    def integrate_with_survival(self, survival_df):

        """Integrate with survival data - NO NULLS GUARANTEED"""

        print("Integrating with survival data...")



        if self.macro_data is None:

            raise ValueError("Must process macro data first!")



        # Preserve 'quarter' if present

        if 'quarter' in survival_df.columns:

            original_quarter = survival_df['quarter'].copy()

        else:

            original_quarter = None



        # Map origination dates to macro quarters using standard quarterly mapping

        survival_df['macro_quarter'] = survival_df['ORIG_DATE'].apply(self.map_to_quarter)



        # Merge with macro data

        merged = survival_df.merge(

            self.macro_data,

            left_on='macro_quarter',

            right_index=True,

            how='left'

        )



        # Re-attach quarter column if it existed

        if original_quarter is not None:

            merged['quarter'] = original_quarter

        

        # Handle missing macro data for out-of-range dates

        macro_cols = [col for col in merged.columns if col not in survival_df.columns]

        macro_start = self.macro_data.index.min()

        macro_end = self.macro_data.index.max()

        

        for col in macro_cols:

            if merged[col].isnull().any():

                # For dates before macro start, use first available values

                before_mask = merged['macro_quarter'] < macro_start

                if before_mask.any():

                    first_val = self.macro_data[col].iloc[0]

                    merged.loc[before_mask, col] = first_val

                

                # For dates after macro end, use last available values

                after_mask = merged['macro_quarter'] > macro_end

                if after_mask.any():

                    last_val = self.macro_data[col].iloc[-1]

                    merged.loc[after_mask, col] = last_val

                

                # For any remaining nulls, use median

                if merged[col].isnull().any():

                    median_val = self.macro_data[col].median()

                    merged[col] = merged[col].fillna(median_val)

        

        # Fix data types for Parquet compatibility

        merged = self._fix_data_types_for_parquet(merged)

        

        # Final null check

        null_count = merged.isnull().sum().sum()

        print(f"Integration complete. Shape: {merged.shape}")

        print(f"Null values: {null_count}")

        

        if null_count > 0:

            print("WARNING: Nulls found after integration - applying final cleanup")

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

        

        summary = {

            'total_features': len(self.macro_data.columns),

            'observations': len(self.macro_data),

            'date_range': f"{self.macro_data.index.min()} to {self.macro_data.index.max()}",

            'null_values': self.macro_data.isnull().sum().sum(),

            'pca_domains': len(self.pca_models),

            'feature_types': {

                'pca_features': len([c for c in self.macro_data.columns if '_PC' in c]),

                'level_vars': len([c for c in self.macro_data.columns if c in ['UNRATE', 'FEDFUNDS', 'GS10', 'CFNAI', 'NFCI']]),

                'spreads': len([c for c in self.macro_data.columns if 'spread' in c]),

                'indicators': len([c for c in self.macro_data.columns if any(x in c for x in ['regime', 'composite', 'conditions', 'duration'])])

            }

        }

        

        return summary



# Main execution

if __name__ == "__main__":

    # Fetch fresh data (uncomment if needed)

    # fetch_macro_data()

    

    # Initialize processor

    processor = MacroSurvivalProcessor(RAW_MACRO_FILE)

    

    # Process macro data

    macro_data = processor.load_and_process()

    

    # Get summary

    summary = processor.get_summary()

    print("\n=== PROCESSING SUMMARY ===")

    for key, value in summary.items():

        print(f"{key}: {value}")

    

    # Load survival data

    print("\n=== LOADING SURVIVAL DATA ===")

    survival_df = pd.read_parquet('/Users/sundargodina/Downloads/survival_files/survival_all.parquet')

    

    # Integrate

    final_df = processor.integrate_with_survival(survival_df)

    

    # Save result

    output_file = '/Users/sundargodina/Downloads/fred/survival_with_macro_no_nulls.parquet'

    final_df.to_parquet(output_file)

    

    print(f"\n=== FINAL RESULT ===")

    print(f"File saved: {output_file}")

    print(f"Final shape: {final_df.shape}")

    print(f"Null values: {final_df.isnull().sum().sum()}")

    print(f"Macro columns added: {len([c for c in final_df.columns if c not in survival_df.columns])}")

    

    # Show sample of features

    macro_cols = [col for col in final_df.columns if col not in survival_df.columns]

    print(f"\nSample macro features:")

    print(final_df[macro_cols[:10]].head())

    

    # Show data types

    print(f"\nData types summary:")

    print(final_df.dtypes.value_counts())

    

    # Variance check

    print(f"\nFeature variance check:")

    for col in macro_cols[:10]:

        if final_df[col].dtype in ['float64', 'int64']:

            print(f"{col}: variance = {final_df[col].var():.4f}")
