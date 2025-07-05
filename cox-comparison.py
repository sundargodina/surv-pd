import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import gc
import warnings
warnings.filterwarnings('ignore')

# MEMORY OPTIMIZATION SETTINGS
pd.set_option('mode.copy_on_write', True)

# STEP 1: Memory-Efficient Data Loading with Stratified Sampling
def load_and_stratified_sample(file_path, sample_size=800000, random_state=42):
    """
    Load parquet data with stratified sampling to preserve performance
    Key insight: Stratified sampling by default event maintains model performance better
    """
    print("Loading parquet data with stratified sampling for better performance...")
    
    # Read parquet file efficiently
    print("Reading parquet file...")
    df = pd.read_parquet(file_path)
    
    print(f"Original dataset size: {len(df):,} rows")
    
    # Apply stratified sampling if dataset is large
    if len(df) > sample_size:
        print(f"Applying stratified sampling to {sample_size:,} rows...")
        
        # Stratified sampling by default_event to preserve event distribution
        if 'default_event' in df.columns:
            # Sample separately for events and non-events
            events = df[df['default_event'] == 1]
            non_events = df[df['default_event'] == 0]
            
            # Calculate proportional sample sizes
            event_ratio = len(events) / len(df)
            event_sample_size = int(sample_size * event_ratio)
            non_event_sample_size = sample_size - event_sample_size
            
            # Sample from each group
            event_sample = events.sample(n=min(event_sample_size, len(events)), 
                                       random_state=random_state) if len(events) > 0 else pd.DataFrame()
            non_event_sample = non_events.sample(n=min(non_event_sample_size, len(non_events)), 
                                               random_state=random_state) if len(non_events) > 0 else pd.DataFrame()
            
            df = pd.concat([event_sample, non_event_sample], ignore_index=True)
            
            # Clean up
            del events, non_events, event_sample, non_event_sample
        else:
            # Fallback to simple sampling if default_event not available
            df = df.sample(n=sample_size, random_state=random_state)
    
    # FIXED: Parse quarter column properly
    if 'quarter' in df.columns:
        # Extract year and quarter from '2017Q1' format
        df['orig_year'] = df['quarter'].str[:4].astype('int16')
        df['quarter_num'] = df['quarter'].str[-1].astype('int8')
        # Keep original quarter for time-based splitting
        df['quarter_year'] = df['quarter'].astype('category')
    
    # Optimize data types for memory efficiency
    dtype_dict = {
        'LOAN_AGE': 'int16',
        'CSCORE_B': 'int16', 
        'DTI': 'float32',
        'ORIG_RATE': 'float32',
        'ORIG_UPB': 'float32',
        'ORIG_TERM': 'int16',
        'OLTV': 'float32',
        'NUM_BO': 'int8',
        'FIRST_FLAG': 'category',
        'PURPOSE': 'category',
        'PROP': 'category',
        'OCC_STAT': 'category',
        'CHANNEL': 'category',
        'STATE': 'category',
        'default_event': 'int8',
        'survival_time': 'float32',
        'quarter_num': 'int8',
        'orig_year': 'int16',
        'credit_score_bucket': 'category',
        'dti_bucket': 'category',
        'ltv_bucket': 'category',
        # Macro economic features
        'interest_rates_PC1': 'float32',
        'credit_spreads_PC1': 'float32',
        'housing_PC1': 'float32',
        'labor_market_PC1': 'float32',
        'financial_stress_PC1': 'float32',
        'economic_activity_PC2': 'float32',
        'inflation_PC1': 'float32',
        'markets_sentiment_PC2': 'float32'
    }
    
    # Apply optimized dtypes where columns exist
    for col, dtype in dtype_dict.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype)
            except:
                print(f"Warning: Could not convert {col} to {dtype}")
    
    gc.collect()
    
    print(f"Loaded {len(df):,} loans with stratified sampling")
    if 'default_event' in df.columns:
        print(f"Default rate preserved: {df['default_event'].mean():.4f}")
    return df

# STEP 2: Curated Feature Selection (Removing Data Leakage & Useless Features)
core_features = [
    'CSCORE_B', 'DTI', 'ORIG_RATE', 'ORIG_UPB', 'ORIG_TERM', 'OLTV', 
    'NUM_BO', 'FIRST_FLAG', 'PURPOSE', 'PROP', 'OCC_STAT', 'CHANNEL', 
    'STATE', 'credit_score_bucket', 'dti_bucket', 'ltv_bucket'
]

macro_features = [
    'interest_rates_PC1', 'credit_spreads_PC1', 'housing_PC1',
    'labor_market_PC1', 'financial_stress_PC1', 'economic_activity_PC2',
    'inflation_PC1', 'markets_sentiment_PC2'
]

duration_col = 'survival_time'
event_col = 'default_event'

# STEP 3: Enhanced Train/Test Split with Time-Based Validation
def enhanced_time_split(df):
    """Enhanced time-based split with Q1 and Q3 data only from 2017-2022"""
    if 'quarter_year' not in df.columns:
        df['quarter_year'] = df['quarter'].astype(str) if 'quarter' in df.columns else 'Unknown'
    
    train_quarters = ['2017Q1', '2017Q3', '2018Q1', '2018Q3', '2019Q1', '2019Q3', '2020Q1']
    val_quarters = ['2020Q3', '2021Q1']
    test_quarters = ['2021Q3', '2022Q1', '2022Q3']
    
    train_mask = df['quarter_year'].isin(train_quarters)
    val_mask = df['quarter_year'].isin(val_quarters)
    test_mask = df['quarter_year'].isin(test_quarters)
    
    print(f"Train quarters: {train_quarters}")
    print(f"Val quarters: {val_quarters}")
    print(f"Test quarters: {test_quarters}")
    print(f"Available quarters in data: {sorted(df['quarter_year'].unique())}")
    
    # Create datasets for both models
    core_cols = [col for col in core_features if col in df.columns] + [duration_col, event_col]
    macro_cols = [col for col in core_features + macro_features if col in df.columns] + [duration_col, event_col]
    
    # Core model data
    train_core = df[train_mask][core_cols].copy()
    val_core = df[val_mask][core_cols].copy()
    test_core = df[test_mask][core_cols].copy()
    
    # Macro model data  
    train_macro = df[train_mask][macro_cols].copy()
    val_macro = df[val_mask][macro_cols].copy()
    test_macro = df[test_mask][macro_cols].copy()
    
    print(f"Train: {len(train_core):,}, Val: {len(val_core):,}, Test: {len(test_core):,}")
    
    # If no data in time splits, use random split as fallback
    if len(train_core) == 0 or len(val_core) == 0 or len(test_core) == 0:
        print("Warning: Time-based split resulted in empty datasets. Using random split as fallback.")
        train_size = int(0.7 * len(df))
        val_size = int(0.15 * len(df))
        
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        train_core = df_shuffled[:train_size][core_cols].copy()
        val_core = df_shuffled[train_size:train_size+val_size][core_cols].copy()
        test_core = df_shuffled[train_size+val_size:][core_cols].copy()
        
        train_macro = df_shuffled[:train_size][macro_cols].copy()
        val_macro = df_shuffled[train_size:train_size+val_size][macro_cols].copy()
        test_macro = df_shuffled[train_size+val_size:][macro_cols].copy()
        
        print(f"Fallback split - Train: {len(train_core):,}, Val: {len(val_core):,}, Test: {len(test_core):,}")
    
    gc.collect()
    return (train_core, val_core, test_core), (train_macro, val_macro, test_macro)

# STEP 4: FIXED Preprocessing with Enhanced NaN Handling
def optimized_preprocessing(datasets):
    """
    Enhanced preprocessing with comprehensive NaN handling and data validation
    """
    train_df, val_df, test_df = datasets
    
    print(f"Initial data shapes: Train={train_df.shape}, Val={val_df.shape}, Test={test_df.shape}")
    
    # Check for NaN values before processing
    print("Checking for NaN values in datasets...")
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        nan_counts = df.isnull().sum()
        total_nans = nan_counts.sum()
        if total_nans > 0:
            print(f"{name} dataset has {total_nans} NaN values:")
            print(nan_counts[nan_counts > 0])
    
    # Compute statistics only from training data
    train_stats = {}
    
    # Separate numeric and categorical columns
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in [duration_col, event_col]]
    
    categorical_cols = train_df.select_dtypes(include=['category', 'object']).columns.tolist()
    
    print(f"Numeric columns: {len(numeric_cols)}")
    print(f"Categorical columns: {len(categorical_cols)}")
    
    # Store training statistics for imputation
    for col in numeric_cols:
        if col in train_df.columns:
            # Use median for numeric, but handle edge cases
            median_val = train_df[col].median()
            if pd.isna(median_val):
                # If median is NaN, use 0 or column mean
                mean_val = train_df[col].mean()
                train_stats[col] = 0 if pd.isna(mean_val) else mean_val
            else:
                train_stats[col] = median_val
    
    for col in categorical_cols:
        if col in train_df.columns:
            mode_vals = train_df[col].mode()
            train_stats[col] = mode_vals[0] if len(mode_vals) > 0 else 'Unknown'
    
    # Apply preprocessing to all datasets
    clean_datasets = []
    for dataset_name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(f"\nProcessing {dataset_name} dataset...")
        df_clean = df.copy()
        
        # Fill missing values using training statistics
        for col in numeric_cols:
            if col in df_clean.columns:
                missing_before = df_clean[col].isnull().sum()
                df_clean[col] = df_clean[col].fillna(train_stats[col])
                missing_after = df_clean[col].isnull().sum()
                if missing_before > 0:
                    print(f"  {col}: filled {missing_before} missing values")
        
        for col in categorical_cols:
            if col in df_clean.columns:
                missing_before = df_clean[col].isnull().sum()
                df_clean[col] = df_clean[col].fillna(train_stats[col])
                missing_after = df_clean[col].isnull().sum()
                if missing_before > 0:
                    print(f"  {col}: filled {missing_before} missing values")
        
        # Handle survival time and event columns specifically
        print(f"  Checking survival_time and default_event columns...")
        
        # Remove rows with invalid survival times or events
        before_cleaning = len(df_clean)
        
        # Remove rows where survival_time is NaN, <= 0, or infinite
        valid_survival = (
            df_clean[duration_col].notna() & 
            (df_clean[duration_col] > 0) & 
            np.isfinite(df_clean[duration_col])
        )
        
        # Remove rows where event is NaN or not 0/1
        valid_event = (
            df_clean[event_col].notna() & 
            df_clean[event_col].isin([0, 1])
        )
        
        # Combine conditions
        valid_rows = valid_survival & valid_event
        df_clean = df_clean[valid_rows].copy()
        
        after_cleaning = len(df_clean)
        removed_rows = before_cleaning - after_cleaning
        
        if removed_rows > 0:
            print(f"  Removed {removed_rows} rows with invalid survival_time or default_event")
        
        # Final check for any remaining NaN values
        remaining_nans = df_clean.isnull().sum().sum()
        if remaining_nans > 0:
            print(f"  Warning: {remaining_nans} NaN values remain in {dataset_name}")
            # Show which columns still have NaN
            nan_cols = df_clean.columns[df_clean.isnull().any()].tolist()
            for col in nan_cols:
                nan_count = df_clean[col].isnull().sum()
                print(f"    {col}: {nan_count} NaN values")
            
            # Drop rows with any remaining NaN values
            df_clean = df_clean.dropna()
            print(f"  Dropped rows with NaN, final shape: {df_clean.shape}")
        
        clean_datasets.append(df_clean)
    
    print(f"\nAfter preprocessing - Train: {len(clean_datasets[0]):,}, "
          f"Val: {len(clean_datasets[1]):,}, Test: {len(clean_datasets[2]):,}")
    
    # Categorical encoding
    train_clean, val_clean, test_clean = clean_datasets
    
    # Get categorical columns that exist in the data
    categorical_cols_existing = [col for col in categorical_cols if col in train_clean.columns]
    
    if categorical_cols_existing:
        print(f"\nEncoding {len(categorical_cols_existing)} categorical features...")
        
        # Create dummy variables using training data to define categories
        train_encoded = pd.get_dummies(train_clean, columns=categorical_cols_existing, 
                                     drop_first=True, dtype='int8')
        
        # Get the column names after encoding
        encoded_cols = train_encoded.columns.tolist()
        
        # Apply same encoding to validation and test (align columns)
        val_encoded = pd.get_dummies(val_clean, columns=categorical_cols_existing, 
                                   drop_first=True, dtype='int8')
        test_encoded = pd.get_dummies(test_clean, columns=categorical_cols_existing, 
                                    drop_first=True, dtype='int8')
        
        # Align columns (add missing columns with 0s)
        for df_encoded in [val_encoded, test_encoded]:
            for col in encoded_cols:
                if col not in df_encoded.columns:
                    df_encoded[col] = 0
        
        # Reorder columns to match training data
        val_encoded = val_encoded[encoded_cols]
        test_encoded = test_encoded[encoded_cols]
        
        # Final NaN check after encoding
        for name, df_enc in [("Train", train_encoded), ("Val", val_encoded), ("Test", test_encoded)]:
            nan_count = df_enc.isnull().sum().sum()
            if nan_count > 0:
                print(f"  Warning: {name} has {nan_count} NaN values after encoding")
                # Remove any remaining NaN rows
                df_enc = df_enc.dropna()
        
        print(f"After encoding - Features: {len(encoded_cols) - 2}")  # -2 for duration and event cols
        
        return [train_encoded, val_encoded, test_encoded]
    else:
        print("No categorical features to encode")
        return clean_datasets

# STEP 5: FIXED Cox Model Fitting with Better Error Handling
def fit_optimized_cox(train_data, val_data, duration_col, event_col, model_name="Model"):
    """
    Fit optimized Cox model with comprehensive error handling and data validation
    """
    print(f"\nFitting {model_name}...")
    
    # Comprehensive data validation
    print(f"Data shape: {train_data.shape}")
    print(f"Duration column '{duration_col}' stats:")
    print(f"  Min: {train_data[duration_col].min():.4f}")
    print(f"  Max: {train_data[duration_col].max():.4f}")
    print(f"  Mean: {train_data[duration_col].mean():.4f}")
    print(f"  NaN count: {train_data[duration_col].isnull().sum()}")
    
    print(f"Event column '{event_col}' stats:")
    print(f"  Value counts: {train_data[event_col].value_counts().to_dict()}")
    print(f"  NaN count: {train_data[event_col].isnull().sum()}")
    
    # Check for any remaining NaN values
    total_nans = train_data.isnull().sum().sum()
    if total_nans > 0:
        print(f"ERROR: Found {total_nans} NaN values in training data")
        nan_cols = train_data.columns[train_data.isnull().any()].tolist()
        for col in nan_cols:
            nan_count = train_data[col].isnull().sum()
            print(f"  {col}: {nan_count} NaN values")
        
        # Remove rows with NaN values
        print("Removing rows with NaN values...")
        train_data = train_data.dropna()
        val_data = val_data.dropna()
        print(f"After NaN removal - Train: {len(train_data)}, Val: {len(val_data)}")
    
    # Check for infinite values
    numeric_cols = train_data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        inf_count = np.isinf(train_data[col]).sum()
        if inf_count > 0:
            print(f"WARNING: Found {inf_count} infinite values in {col}")
            # Replace infinite values with median
            median_val = train_data[col][np.isfinite(train_data[col])].median()
            train_data[col] = train_data[col].replace([np.inf, -np.inf], median_val)
            val_data[col] = val_data[col].replace([np.inf, -np.inf], median_val)
    
    # Ensure event column is properly coded
    if not train_data[event_col].isin([0, 1]).all():
        print(f"ERROR: Event column contains values other than 0 and 1")
        return None, 0.0
    
    # Check for variance in features
    feature_cols = [col for col in train_data.columns if col not in [duration_col, event_col]]
    zero_var_cols = []
    for col in feature_cols:
        if train_data[col].var() == 0:
            zero_var_cols.append(col)
    
    if zero_var_cols:
        print(f"Removing {len(zero_var_cols)} zero-variance features: {zero_var_cols}")
        train_data = train_data.drop(columns=zero_var_cols)
        val_data = val_data.drop(columns=zero_var_cols)
    
    # Final data validation
    print(f"Final training data shape: {train_data.shape}")
    print(f"Event rate: {train_data[event_col].mean():.4f}")
    
    # Fit Cox model with more conservative settings
    try:
        cox_model = CoxPHFitter(
            penalizer=0.1,     # Increased regularization
            l1_ratio=0.1,      # Mostly Ridge regularization
            alpha=0.05
        )
        
        # Fit the model
        cox_model.fit(
            train_data, 
            duration_col=duration_col, 
            event_col=event_col,
            show_progress=True
        )
        
        # Validation performance
        val_c_index = concordance_index(
            val_data[duration_col], 
            -cox_model.predict_partial_hazard(val_data), 
            val_data[event_col]
        )
        
        print(f"{model_name} - Validation C-index: {val_c_index:.4f}")
        print(f"{model_name} - Coefficients: {cox_model.summary.shape[0]}")
        
        return cox_model, val_c_index
        
    except Exception as e:
        print(f"Error fitting {model_name}: {str(e)}")
        print("Trying with higher regularization...")
        
        # Try with higher regularization
        try:
            cox_model = CoxPHFitter(
                penalizer=0.5,     # Much higher regularization
                l1_ratio=0.0,      # Pure Ridge regularization
                alpha=0.05
            )
            
            cox_model.fit(
                train_data, 
                duration_col=duration_col, 
                event_col=event_col,
                show_progress=True
            )
            
            val_c_index = concordance_index(
                val_data[duration_col], 
                -cox_model.predict_partial_hazard(val_data), 
                val_data[event_col]
            )
            
            print(f"{model_name} (high reg) - Validation C-index: {val_c_index:.4f}")
            return cox_model, val_c_index
            
        except Exception as e2:
            print(f"Failed even with high regularization: {str(e2)}")
            return None, 0.0

# STEP 6: Compare Models
def compare_models(core_model, macro_model, core_data, macro_data):
    """Comprehensive comparison of both models"""
    print("\n" + "="*60)
    print("MODEL COMPARISON RESULTS")
    print("="*60)
    
    results = {}
    
    for model_name, model, data in [("Core", core_model, core_data), ("Macro", macro_model, macro_data)]:
        if model is None:
            print(f"\n{model_name} Model: FAILED TO FIT")
            continue
            
        train_data, val_data, test_data = data
        
        # Calculate performance metrics
        train_c = concordance_index(
            train_data[duration_col], 
            -model.predict_partial_hazard(train_data), 
            train_data[event_col]
        )
        
        val_c = concordance_index(
            val_data[duration_col], 
            -model.predict_partial_hazard(val_data), 
            val_data[event_col]
        )
        
        test_c = concordance_index(
            test_data[duration_col], 
            -model.predict_partial_hazard(test_data), 
            test_data[event_col]
        )
        
        results[model_name] = {
            'train_c': train_c,
            'val_c': val_c, 
            'test_c': test_c,
            'features': model.summary.shape[0]
        }
        
        print(f"\n{model_name} Model Results:")
        print(f"  Features: {model.summary.shape[0]}")
        print(f"  Train C-index: {train_c:.4f}")
        print(f"  Val C-index: {val_c:.4f}")
        print(f"  Test C-index: {test_c:.4f}")
        print(f"  Overfit (Train-Test): {(train_c - test_c):.4f}")
        
        # Top features
        if hasattr(model, 'summary') and len(model.summary) > 0:
            print(f"  Top 5 Features:")
            top_features = model.summary['coef'].abs().sort_values(ascending=False).head(5)
            for i, (feature, coef) in enumerate(top_features.items(), 1):
                direction = "↑" if model.summary.loc[feature, 'coef'] > 0 else "↓"
                print(f"    {i}. {feature}: {coef:.3f} {direction}")
    
    # Model comparison summary
    print(f"\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    if 'Core' in results and 'Macro' in results:
        core_test = results['Core']['test_c']
        macro_test = results['Macro']['test_c']
        improvement = macro_test - core_test
        
        print(f"Core Model Test C-index: {core_test:.4f}")
        print(f"Macro Model Test C-index: {macro_test:.4f}")
        print(f"Improvement from Macro: {improvement:.4f} ({improvement/core_test*100:.1f}%)")
        
        if improvement > 0.005:
            print("✅ Macro features provide meaningful improvement")
        else:
            print("❌ Macro features don't provide significant improvement")
    
    return results

# STEP 7: Generate Predictions
def generate_predictions(model, test_data, model_name="Model"):
    """Generate predictions for IFRS9 compliance"""
    print(f"\nGenerating predictions from {model_name}...")
    
    # Sample for memory efficiency
    sample_size = min(2000, len(test_data))
    pred_sample = test_data.sample(n=sample_size, random_state=42)
    
    predictions = pred_sample.copy()
    
    # Risk scores
    predictions['risk_score'] = model.predict_partial_hazard(pred_sample)
    
    # Survival probabilities for key horizons
    for horizon in [6, 12, 24]:
        survival_probs = []
        for i in range(len(pred_sample)):
            try:
                survival_func = model.predict_survival_function(pred_sample.iloc[[i]])
                times = survival_func.index
                
                if horizon <= times.max():
                    # Find closest time
                    closest_time = min(times, key=lambda x: abs(x - horizon))
                    prob = survival_func.loc[closest_time].iloc[0]
                else:
                    prob = 0.0  # Assume default if beyond observation
                
                survival_probs.append(prob)
            except:
                survival_probs.append(np.nan)
        
        predictions[f'survival_prob_{horizon}m'] = survival_probs
        predictions[f'default_prob_{horizon}m'] = 1 - np.array(survival_probs)
    
    # Risk tiers
    predictions['risk_tier'] = pd.cut(
        predictions['risk_score'],
        bins=[0, 0.3, 0.7, 1.5, float('inf')],
        labels=['Low', 'Medium', 'High', 'Very High']
    )
    
    return predictions

# MAIN EXECUTION
print("Starting Cox Regression Analysis with Enhanced NaN Handling...")

# Load data
df = load_and_stratified_sample('/Users/sundargodina/new.parquet')

# Apply time-based split
(train_core, val_core, test_core), (train_macro, val_macro, test_macro) = enhanced_time_split(df)

# Preprocess data with enhanced NaN handling
print("\n" + "="*60)
print("PREPROCESSING CORE MODEL DATA")
print("="*60)
core_clean = optimized_preprocessing([train_core, val_core, test_core])

print("\n" + "="*60)
print("PREPROCESSING MACRO MODEL DATA")
print("="*60)
macro_clean = optimized_preprocessing([train_macro, val_macro, test_macro])

# Fit models
core_model, core_val_c = fit_optimized_cox(
    core_clean[0], core_clean[1], duration_col, event_col, "Core Model"
)

macro_model, macro_val_c = fit_optimized_cox(
    macro_clean[0], macro_clean[1], duration_col, event_col, "Macro Model"
)

# Compare models
comparison_results = compare_models(core_model, macro_model, core_clean, macro_clean)

# Generate predictions
if core_model:
    core_predictions = generate_predictions(core_model, core_clean[2], "Core Model")
    print(f"\nCore model predictions sample:")
    print(core_predictions[['risk_score', 'default_prob_12m', 'risk_tier']].head())
    
if macro_model:
    macro_predictions = generate_predictions(macro_model, macro_clean[2], "Macro Model")
    print(f"\nMacro model predictions sample:")
    print(macro_predictions[['risk_score', 'default_prob_12m', 'risk_tier']].head())

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
