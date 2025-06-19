# Domain-Specific PCA on Macroeconomic Data for Survival Analysis (Split: Fetching and PCA)
import pandas as pd
import numpy as np
from fredapi import Fred
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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
    


