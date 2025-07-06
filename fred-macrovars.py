# Domain-Specific PCA on Macroeconomic Data for Survival Analysis (Split: Fetching and PCA)
import pandas as pd
import numpy as np
from scipy import stats)
from fredapi import Fred
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,RobustScaler
import warnings
import os
warnings.filterwarnings('ignore')

FRED_API_KEY = 'nope'
START_DATE = '2017-01-01'
END_DATE = '2023-06-30'
TARGET_FREQ = 'Q'
N_COMPONENTS_PER_DOMAIN = 2
RAW_MACRO_FILE = '/Users/sme/Downloads/fred/raw_macro_data.csv'

fred = Fred(api_key=FRED_API_KEY)


domain_groups = {
    'interest_rates': ['FEDFUNDS', 'GS10', 'GS2', 'GS5', 'TB3MS', 'T10Y2Y', 'T10Y3M'],
    'credit_spreads': ['DBAA', 'DAAA', 'BAA10Y', 'AAA10Y', 'BAMLH0A0HYM2', 'BAMLC0A0CM'],
    'labor_market': ['UNRATE', 'PAYEMS', 'ICSA', 'U6RATE', 'CIVPART'],
    'inflation': ['CPIAUCSL', 'CPILFESL', 'PCEPI', 'PCEPILFE', 'T5YIE', 'T10YIE'],
    'economic_activity': ['GDPC1', 'INDPRO', 'CFNAI', 'CFNAIDIFF', 'RECPROUSM156N'],
    'housing': ['HOUST', 'USSTHPI', 'CSUSHPISA', 'MORTGAGE30US', 'RHORUSQ156N'],
    'financial_stress': ['NFCI', 'ANFCI', 'NFCINONFINLEVERAGE', 'STLFSI4', 'TEDRATE', 'VIXCLS'],
    'markets_sentiment': ['SP500', 'NASDAQCOM', 'UMCSENT', 'USSLIND', 'BFCIUS'],
    'money_credit': ['M2SL', 'TOTLL', 'CONSUMER', 'DRSFRMACBS', 'DRBLACBS', 'DRCCLACBS'],
    'commod_fx': ['DCOILWTICO', 'GOLDAMGBD228NLBM', 'DEXUSEU', 'DTWEXBGS']
}


def fetch_series(series_list):
    df = pd.DataFrame()
    for code in series_list:
        try:
            print(f"Fetching {code}...")
            data = fred.get_series(code, observation_start=START_DATE, observation_end=END_DATE)
            df[code] = data
        except:
            print(f"Failed: {code}")
    df.index = pd.to_datetime(df.index)
    return df

def process_macro_to_quarterly(df):
    quarterly_df = df.resample('Q').mean().ffill().bfill()
    quarterly_df = quarterly_df[quarterly_df.index.quarter.isin([1, 3])]
    return quarterly_df

def fetch_and_save_all_macro():
    all_data = {}
    for domain, codes in domain_groups.items():
        df = fetch_series(codes)
        quarterly_df = process_macro_to_quarterly(df)
        for col in quarterly_df.columns:
            all_data[col] = quarterly_df[col]
    full_df = pd.DataFrame(all_data)
    full_df.index.name = 'date'
    full_df.to_csv(RAW_MACRO_FILE)
    print(f"Saved raw quarterly macro data to {RAW_MACRO_FILE} with shape {full_df.shape}")


#saves raw macro data from 2017 first quarter to 2022 3rd quarter, only saving Q1 AND Q3 data.
fetch_and_save_all_macro()
    


class MacroSurvivalProcessor:
    def __init__(self, raw_macro_file, n_components_per_domain=2):
        self.raw_macro_file = raw_macro_file
        self.n_components_per_domain = n_components_per_domain
        self.domain_groups = {
            'interest_rates': ['FEDFUNDS', 'GS10', 'GS2', 'GS5', 'TB3MS', 'T10Y2Y', 'T10Y3M'],
            'credit_spreads': ['DBAA', 'DAAA', 'BAA10Y', 'AAA10Y', 'BAMLH0A0HYM2', 'BAMLC0A0CM'],
            'labor_market': ['UNRATE', 'PAYEMS', 'ICSA', 'U6RATE', 'CIVPART'],
            'inflation': ['CPIAUCSL', 'CPILFESL', 'PCEPI', 'PCEPILFE', 'T5YIE', 'T10YIE'],
            'economic_activity': ['GDPC1', 'INDPRO', 'CFNAI', 'CFNAIDIFF', 'RECPROUSM156N'],
            'housing': ['HOUST', 'USSTHPI', 'CSUSHPISA', 'MORTGAGE30US', 'RHORUSQ156N'],
            'financial_stress': ['NFCI', 'ANFCI', 'NFCINONFINLEVERAGE', 'STLFSI4', 'TEDRATE', 'VIXCLS'],
            'markets_sentiment': ['SP500', 'NASDAQCOM', 'UMCSENT', 'USSLIND', 'BFCIUS'],
            'money_credit': ['M2SL', 'TOTLL', 'CONSUMER', 'DRSFRMACBS', 'DRBLACBS', 'DRCCLACBS'],
            'commod_fx': ['DCOILWTICO', 'GOLDAMGBD228NLBM', 'DEXUSEU', 'DTWEXBGS']
        }
        self.macro_processed = None
        self.scalers = {}
        self.pca_models = {}
        
    def load_and_preprocess_macro(self):
        """Load and preprocess raw macro data"""
        print("Loading raw macro data...")
        macro_df = pd.read_csv(self.raw_macro_file, index_col=0, parse_dates=True)
        
        # Ensure quarterly Q1 and Q3 only
        macro_df = macro_df[macro_df.index.quarter.isin([1, 3])]
        
        print(f"Loaded macro data shape: {macro_df.shape}")
        print(f"Date range: {macro_df.index.min()} to {macro_df.index.max()}")
        
        # Handle missing values
        macro_df = self._handle_missing_values(macro_df)
        
        # Apply transformations to make series stationary and meaningful
        macro_df = self._apply_transformations(macro_df)
        
        # Detect and handle outliers
        macro_df = self._handle_outliers(macro_df)
        
        # Create lagged and trend features
        macro_df = self._create_temporal_features(macro_df)
        
        # Apply PCA by domain
        macro_df = self._apply_domain_pca(macro_df)
        
        # Create economic regime indicators
        macro_df = self._create_economic_indicators(macro_df)
        
        self.macro_processed = macro_df
        print(f"Final processed macro shape: {macro_df.shape}")
        return macro_df
    
    def _handle_missing_values(self, df):
        """Handle missing values in macro data"""
        print("Handling missing values...")
        
        # Forward fill then backward fill (max 2 periods each)
        df = df.fillna(method='ffill', limit=2)
        df = df.fillna(method='bfill', limit=2)
        
        # For remaining missing values, use interpolation
        df = df.interpolate(method='linear', limit=4)
        
        # Drop columns with >20% missing values
        missing_pct = df.isnull().sum() / len(df)
        cols_to_drop = missing_pct[missing_pct > 0.2].index.tolist()
        if cols_to_drop:
            print(f"Dropping columns with >20% missing: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)
        
        return df
    
    def _apply_transformations(self, df):
        """Apply economic transformations to make series stationary"""
        print("Applying economic transformations...")
        
        transformations = {
            # GDP and economic activity - use growth rates
            'GDPC1': lambda x: x.pct_change(periods=4) * 100,  # YoY growth
            'INDPRO': lambda x: x.pct_change(periods=4) * 100,
            
            # Price indices - use inflation rates
            'CPIAUCSL': lambda x: x.pct_change(periods=4) * 100,
            'CPILFESL': lambda x: x.pct_change(periods=4) * 100,
            'PCEPI': lambda x: x.pct_change(periods=4) * 100,
            'PCEPILFE': lambda x: x.pct_change(periods=4) * 100,
            'USSTHPI': lambda x: x.pct_change(periods=4) * 100,
            'CSUSHPISA': lambda x: x.pct_change(periods=4) * 100,
            
            # Employment - use levels and growth
            'PAYEMS': lambda x: x.pct_change(periods=4) * 100,
            
            # Money supply - use growth rates
            'M2SL': lambda x: x.pct_change(periods=4) * 100,
            'TOTLL': lambda x: x.pct_change(periods=4) * 100,
            'CONSUMER': lambda x: x.pct_change(periods=4) * 100,
            
            # Stock markets - use returns
            'SP500': lambda x: x.pct_change(periods=1) * 100,
            'NASDAQCOM': lambda x: x.pct_change(periods=1) * 100,
            
            # Housing starts
            'HOUST': lambda x: x.pct_change(periods=4) * 100,
            
            # Commodities - use returns
            'DCOILWTICO': lambda x: x.pct_change(periods=1) * 100,
            'GOLDAMGBD228NLBM': lambda x: x.pct_change(periods=1) * 100,
        }
        
        # Apply transformations
        for col, transform_func in transformations.items():
            if col in df.columns:
                df[f'{col}_transformed'] = transform_func(df[col])
                # Keep original for some key indicators
                if col in ['UNRATE', 'FEDFUNDS', 'GS10', 'GS2']:
                    continue
                else:
                    df = df.drop(columns=[col])
        
        return df
    
    def _handle_outliers(self, df):
        """Handle outliers using robust methods"""
        print("Handling outliers...")
        
        # Use IQR method for outlier detection and winsorization
        for col in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            # Winsorize extreme outliers
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df
    
    def _create_temporal_features(self, df):
        """Create lagged and trend features"""
        print("Creating temporal features...")
        
        # Key indicators for temporal features
        key_indicators = ['UNRATE', 'FEDFUNDS', 'GS10', 'CFNAI', 'NFCI']
        
        for indicator in key_indicators:
            if indicator in df.columns:
                # 1-quarter lag
                df[f'{indicator}_lag1'] = df[indicator].shift(1)
                
                # 4-quarter change
                df[f'{indicator}_yoy_change'] = df[indicator] - df[indicator].shift(4)
                
                # Volatility (4-quarter rolling std)
                df[f'{indicator}_volatility'] = df[indicator].rolling(window=4).std()
                
                # Moving average
                df[f'{indicator}_ma4'] = df[indicator].rolling(window=4).mean()
                
                # Trend (slope of last 4 quarters)
                df[f'{indicator}_trend'] = df[indicator].rolling(window=4).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 4 else np.nan
                )
        
        return df
    
    def _apply_domain_pca(self, df):
        """Apply PCA within each domain"""
        print("Applying domain-specific PCA...")
        
        pca_features = pd.DataFrame(index=df.index)
        
        for domain, original_codes in self.domain_groups.items():
            # Find available columns for this domain (including transformed)
            domain_cols = []
            for code in original_codes:
                # Check for original column
                if code in df.columns:
                    domain_cols.append(code)
                # Check for transformed column
                elif f'{code}_transformed' in df.columns:
                    domain_cols.append(f'{code}_transformed')
            
            if len(domain_cols) < 2:
                print(f"Skipping PCA for {domain} - insufficient columns")
                continue
            
            # Get domain data
            domain_data = df[domain_cols].dropna()
            
            
            if len(domain_data) < 8:  # Need enough observations
                print(f"Skipping PCA for {domain} - insufficient observations")
                continue
            
            # Standardize
            scaler = RobustScaler()
            domain_scaled = scaler.fit_transform(domain_data)
            
            # Apply PCA
            n_components = min(self.n_components_per_domain, len(domain_cols))
            pca = PCA(n_components=n_components)
            pca_result = pca.fit_transform(domain_scaled)
            
            # Store results
            for i in range(n_components):
                col_name = f'{domain}_PC{i+1}'
                pca_features[col_name] = np.nan
                pca_features.loc[domain_data.index, col_name] = pca_result[:, i]
            
            # Store models for later use
            self.scalers[domain] = scaler
            self.pca_models[domain] = pca
            
            print(f"{domain}: {len(domain_cols)} variables -> {n_components} PCs "
                  f"(explained variance: {pca.explained_variance_ratio_.sum():.3f})")
        
        # Combine with original key indicators
        key_originals = ['UNRATE', 'FEDFUNDS', 'GS10', 'CFNAI', 'NFCI', 'RECPROUSM156N']
        for col in key_originals:
            if col in df.columns:
                pca_features[col] = df[col]
        
        return pca_features
    
    def _create_economic_indicators(self, df):
        """Create economic regime and stress indicators"""
        print("Creating economic indicators...")
        
        # Recession indicator
        if 'RECPROUSM156N' in df.columns:
            df['recession_indicator'] = (df['RECPROUSM156N'] == 1).astype(int)
        
        # Economic stress indicators
        if 'UNRATE' in df.columns:
            df['unemployment_stress'] = (
                df['UNRATE'] > df['UNRATE'].rolling(window=8).quantile(0.75)
            ).astype(int)
        
        if 'NFCI' in df.columns:
            df['financial_stress'] = (df['NFCI'] > 0).astype(int)
        
        # Interest rate environment
        if 'FEDFUNDS' in df.columns:
            df['fed_funds_rising'] = (df['FEDFUNDS'] > df['FEDFUNDS'].shift(1)).astype(int)
            df['fed_funds_level'] = pd.cut(df['FEDFUNDS'], 
                                         bins=[0, 1, 3, 6, float('inf')], 
                                         labels=['very_low', 'low', 'medium', 'high'])
        
        # Yield curve indicators
        if 'GS10' in df.columns and 'GS2' in df.columns:
            df['yield_curve_slope'] = df['GS10'] - df['GS2']
            df['yield_curve_inverted'] = (df['yield_curve_slope'] < 0).astype(int)
        
        # Credit market stress
        if 'DBAA' in df.columns and 'DAAA' in df.columns:
            df['credit_spread'] = df['DBAA'] - df['DAAA']
            df['credit_stress'] = (
                df['credit_spread'] > df['credit_spread'].rolling(window=8).quantile(0.75)
            ).astype(int)
        
        return df
    
    def map_orig_date_to_quarter(self, orig_date):
        """Map origination date to corresponding quarter"""
        if pd.isna(orig_date):
            return None
        
        date = pd.to_datetime(orig_date)
        year = date.year
        quarter = date.quarter
        
        # Map to Q1 or Q3
        if quarter in [1, 2]:
            return pd.Timestamp(f'{year}-03-31')  # Q1
        else:
            return pd.Timestamp(f'{year}-09-30')  # Q3
    
    def integrate_with_survival_data(self, survival_df, chunk_size=500000):
        """Integrate processed macro data with survival data"""
        print("Integrating macro data with survival data...")
        
        if self.macro_processed is None:
            raise ValueError("Must run load_and_preprocess_macro() first")
        
        # Ensure ORIG_DATE is datetime
        survival_df['ORIG_DATE'] = pd.to_datetime(survival_df['ORIG_DATE'])
        
        # Map origination dates to quarters
        survival_df['macro_quarter'] = survival_df['ORIG_DATE'].apply(
            self.map_orig_date_to_quarter
        )
        
        # Process in chunks if dataset is large
        if len(survival_df) > chunk_size:
            print(f"Processing in chunks of {chunk_size:,} rows...")
            chunks = []
            for i in range(0, len(survival_df), chunk_size):
                chunk = survival_df.iloc[i:i+chunk_size].copy()
                chunk_integrated = self._merge_chunk_with_macro(chunk)
                chunks.append(chunk_integrated)
                print(f"Processed chunk {i//chunk_size + 1}/{len(survival_df)//chunk_size + 1}")
            
            result = pd.concat(chunks, ignore_index=True)
        else:
            result = self._merge_chunk_with_macro(survival_df)
        
        print(f"Integration complete. Final shape: {result.shape}")
        return result
    
    def _merge_chunk_with_macro(self, chunk):
        """Merge a chunk with macro data"""
        # Merge with macro data
        merged = chunk.merge(
            self.macro_processed,
            left_on='macro_quarter',
            right_index=True,
            how='left'
        )
        
        # Handle missing macro data (for dates outside macro range)
        macro_cols = [col for col in merged.columns if col not in chunk.columns]
        
        # Forward fill missing macro data within reasonable limits
        for col in macro_cols:
            if merged[col].isna().any():
                # Use the nearest available quarter's data
                merged[col] = merged[col].fillna(method='ffill', limit=2)
                merged[col] = merged[col].fillna(method='bfill', limit=2)
        
        return merged
    
    def create_ifrs9_features(self, df):
        """Create IFRS 9 specific features"""
        print("Creating IFRS 9 specific features...")
        
        # Economic deterioration indicators
        if 'economic_activity_PC1' in df.columns:
            df['economic_deterioration'] = (
                df['economic_activity_PC1'] < df['economic_activity_PC1'].shift(1)
            ).astype(int)
        
        # Forward-looking stress indicators
        if 'financial_stress_PC1' in df.columns:
            df['forward_stress_indicator'] = (
                df['financial_stress_PC1'] > df['financial_stress_PC1'].rolling(4).mean()
            ).astype(int)
        
        # Staging support features
        if 'credit_spreads_PC1' in df.columns:
            df['credit_conditions'] = pd.cut(
                df['credit_spreads_PC1'],
                bins=[-np.inf, df['credit_spreads_PC1'].quantile(0.33), 
                      df['credit_spreads_PC1'].quantile(0.67), np.inf],
                labels=['good', 'moderate', 'stressed']
            )
        
        return df
    
    def get_feature_summary(self):
        """Get summary of created features"""
        if self.macro_processed is None:
            return "No processed macro data available"
        
        summary = {
            'total_features': len(self.macro_processed.columns),
            'pca_features': len([col for col in self.macro_processed.columns if '_PC' in col]),
            'economic_indicators': len([col for col in self.macro_processed.columns 
                                      if any(x in col for x in ['stress', 'recession', 'indicator'])]),
            'original_indicators': len([col for col in self.macro_processed.columns 
                                      if col in ['UNRATE', 'FEDFUNDS', 'GS10', 'CFNAI', 'NFCI']]),
            'date_range': f"{self.macro_processed.index.min()} to {self.macro_processed.index.max()}"
        }
        
        return summary

processor = MacroSurvivalProcessor('/Users/sundargodina/Downloads/fred/raw_macro_data.csv')
    

macro_processed = processor.load_and_preprocess_macro()
    

print("Loading survival data...")
survival_df = pd.read_parquet('/Users/sundargodina/Downloads/survival_files/survival_all.parquet')

integrated_df = processor.integrate_with_survival_data(survival_df)
    

final_df = processor.create_ifrs9_features(integrated_df)
    
    # Get summary
summary = processor.get_feature_summary()
print("\nFeature Summary:")
for key, value in summary.items():
    print(f"{key}: {value}")
    

final_df.to_parquet('/Users/sundargodina/Downloads/fred/survival_data_with_macro.parquet')
