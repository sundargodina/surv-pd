import polars as pl

column_names = [  # Full list not needed since we're selecting subset
    "POOL_ID", "LOAN_ID", "ACT_PERIOD", "CHANNEL", "SELLER", "SERVICER",
    "MASTER_SERVICER", "ORIG_RATE", "CURR_RATE", "ORIG_UPB", "ISSUANCE_UPB",
    "CURRENT_UPB", "ORIG_TERM", "ORIG_DATE", "FIRST_PAY", "LOAN_AGE",
    "REM_MONTHS", "ADJ_REM_MONTHS", "MATR_DT", "OLTV", "OCLTV",
    "NUM_BO", "DTI", "CSCORE_B", "CSCORE_C", "FIRST_FLAG", "PURPOSE",
    "PROP", "NO_UNITS", "OCC_STAT", "STATE", "MSA", "ZIP", "MI_PCT",
    "PRODUCT", "PPMT_FLG", "IO", "FIRST_PAY_IO", "MNTHS_TO_AMTZ_IO",
    "DLQ_STATUS", "PMT_HISTORY", "MOD_FLAG", "MI_CANCEL_FLAG", "Zero_Bal_Code",
    "ZB_DTE", "LAST_UPB", "RPRCH_DTE", "CURR_SCHD_PRNCPL", "TOT_SCHD_PRNCPL",
    "UNSCHD_PRNCPL_CURR", "LAST_PAID_INSTALLMENT_DATE", "FORECLOSURE_DATE",
    "DISPOSITION_DATE", "FORECLOSURE_COSTS", "PROPERTY_PRESERVATION_AND_REPAIR_COSTS",
    "ASSET_RECOVERY_COSTS", "MISCELLANEOUS_HOLDING_EXPENSES_AND_CREDITS",
    "ASSOCIATED_TAXES_FOR_HOLDING_PROPERTY", "NET_SALES_PROCEEDS",
    "CREDIT_ENHANCEMENT_PROCEEDS", "REPURCHASES_MAKE_WHOLE_PROCEEDS",
    "OTHER_FORECLOSURE_PROCEEDS", "NON_INTEREST_BEARING_UPB", "PRINCIPAL_FORGIVENESS_AMOUNT",
    "ORIGINAL_LIST_START_DATE", "ORIGINAL_LIST_PRICE", "CURRENT_LIST_START_DATE",
    "CURRENT_LIST_PRICE", "ISSUE_SCOREB", "ISSUE_SCOREC", "CURR_SCOREB",
    "CURR_SCOREC", "MI_TYPE", "SERV_IND", "CURRENT_PERIOD_MODIFICATION_LOSS_AMOUNT",
    "CUMULATIVE_MODIFICATION_LOSS_AMOUNT", "CURRENT_PERIOD_CREDIT_EVENT_NET_GAIN_OR_LOSS",
    "CUMULATIVE_CREDIT_EVENT_NET_GAIN_OR_LOSS", "HOMEREADY_PROGRAM_INDICATOR",
    "FORECLOSURE_PRINCIPAL_WRITE_OFF_AMOUNT", "RELOCATION_MORTGAGE_INDICATOR",
    "ZERO_BALANCE_CODE_CHANGE_DATE", "LOAN_HOLDBACK_INDICATOR", "LOAN_HOLDBACK_EFFECTIVE_DATE",
    "DELINQUENT_ACCRUED_INTEREST", "PROPERTY_INSPECTION_WAIVER_INDICATOR",
    "HIGH_BALANCE_LOAN_INDICATOR", "ARM_5_YR_INDICATOR", "ARM_PRODUCT_TYPE",
    "MONTHS_UNTIL_FIRST_PAYMENT_RESET", "MONTHS_BETWEEN_SUBSEQUENT_PAYMENT_RESET",
    "INTEREST_RATE_CHANGE_DATE", "PAYMENT_CHANGE_DATE", "ARM_INDEX",
    "ARM_CAP_STRUCTURE", "INITIAL_INTEREST_RATE_CAP", "PERIODIC_INTEREST_RATE_CAP",
    "LIFETIME_INTEREST_RATE_CAP", "MARGIN", "BALLOON_INDICATOR",
    "PLAN_NUMBER", "FORBEARANCE_INDICATOR", "HIGH_LOAN_TO_VALUE_HLTV_REFINANCE_OPTION_INDICATOR",
    "DEAL_NAME", "RE_PROCS_FLAG", "ADR_TYPE", "ADR_COUNT", "ADR_UPB", "PAYMENT_DEFERRAL_MOD_EVENT_FLAG", "INTEREST_BEARING_UPB"
]
selected_cols = [
    "LOAN_ID", "ACT_PERIOD", "ORIG_DATE", "FIRST_PAY", "LOAN_AGE", "DLQ_STATUS",
    "Zero_Bal_Code", "ZB_DTE", "FORECLOSURE_DATE", "DISPOSITION_DATE",
    "CSCORE_B", "CSCORE_C", "DTI", "NUM_BO", "ORIG_RATE", "CURR_RATE",
    "ORIG_UPB", "CURRENT_UPB", "ORIG_TERM", "OLTV", "OCLTV", "PURPOSE",
    "FORBEARANCE_INDICATOR", "STATE", "MSA", "ZIP", "PROP", "OCC_STAT",
    "MI_PCT", "PPMT_FLG", "MOD_FLAG", "PMT_HISTORY", "PRODUCT", "CHANNEL",
    "FIRST_FLAG", "SELLER", "SERVICER", "NO_UNITS", "CURR_SCOREB", "CURR_SCOREC",
    "SERV_IND", "REM_MONTHS", "MATR_DT"
]

date_cols = [
    "ORIG_DATE", "FIRST_PAY", "ZB_DTE", "FORECLOSURE_DATE", "DISPOSITION_DATE", "MATR_DT"
]

categorical_cols = [
    "DLQ_STATUS", "Zero_Bal_Code", "PURPOSE", "FORBEARANCE_INDICATOR", "STATE", "MSA", "ZIP",
    "PROP", "OCC_STAT", "PPMT_FLG", "MOD_FLAG", "PMT_HISTORY", "PRODUCT", "CHANNEL", "FIRST_FLAG",
    "SELLER", "SERVICER", "SERV_IND"
]
dtype_spec = {
    "LOAN_ID": pl.Utf8,
    "ACT_PERIOD": pl.Utf8,
    "LOAN_AGE": pl.Float64,
    "DLQ_STATUS": pl.Utf8,
    "Zero_Bal_Code": pl.Utf8,
    "CSCORE_B": pl.Float64,
    "CSCORE_C": pl.Float64,
    "DTI": pl.Float64,
    "NUM_BO": pl.Float64,
    "ORIG_RATE": pl.Float64,
    "CURR_RATE": pl.Float64,
    "ORIG_UPB": pl.Float64,
    "CURRENT_UPB": pl.Float64,
    "ORIG_TERM": pl.Float64,
    "OLTV": pl.Float64,
    "OCLTV": pl.Float64,
    "PURPOSE": pl.Utf8,
    "FORBEARANCE_INDICATOR": pl.Utf8,
    "STATE": pl.Utf8,
    "MSA": pl.Utf8,
    "ZIP": pl.Utf8,
    "PROP": pl.Utf8,
    "OCC_STAT": pl.Utf8,
    "MI_PCT": pl.Float64,
    "PPMT_FLG": pl.Utf8,
    "MOD_FLAG": pl.Utf8,
    "PMT_HISTORY": pl.Utf8,
    "PRODUCT": pl.Utf8,
    "CHANNEL": pl.Utf8, "FIRST_FLAG": pl.Utf8,
    "SELLER": pl.Utf8,
    "SERVICER": pl.Utf8,
    "NO_UNITS": pl.Float64,
    "CURR_SCOREB": pl.Float64,
    "CURR_SCOREC": pl.Float64,
    "SERV_IND": pl.Utf8,
    "REM_MONTHS": pl.Float64,
    "MATR_DT": pl.Utf8,
}



quarters = [f"{year}Q{q}" for year in range(2017, 2023) for q in (1, 3)]

base_input_dir = "/Users/sundargodina/Downloads/"
base_output_dir = "/Users/sundargodina/Downloads/project/"
chunksize = 500_000

for quarter in quarters:
    file_path = os.path.join(base_input_dir, f"{quarter}.csv")
    output_path = os.path.join(base_output_dir, f"{quarter}.parquet")

    if not os.path.exists(file_path):
        print(f" File not found: {file_path}. Skipping.")
        continue

    print(f"Processing {file_path}")
    offset = 0
    processed_dfs = []

    while True:
        print(f"Reading rows {offset} to {offset + chunksize - 1}")
        try:
            df = pl.read_csv(
                file_path,
                separator="|",
                new_columns=column_names,
                schema_overrides=dtype_spec,
                has_header=False,
                skip_rows=offset,
                n_rows=chunksize,
                ignore_errors=True,
            )
        except pl.exceptions.NoDataError:
            print(" No more data or malformed chunk. Stopping read.")
            break

        if df.is_empty():
            break

        df = df.select(selected_cols)

        # Date parsing
        for col in date_cols:
            if col in df.columns:
                str_col = f"{col}_str"
                df = df.with_columns([
                    pl.col(col).cast(pl.Utf8).map_elements(convert_myyyy, return_dtype=pl.Utf8).alias(str_col)
                ])
                df = df.with_columns([
                    pl.col(str_col).str.strptime(pl.Date, format="%Y-%m-%d", strict=False).alias(col)
                ])

        # Fill missing categoricals
        for col in categorical_cols:
            if col in df.columns:
                df = df.with_columns([
                    pl.when(pl.col(col).is_null()).then(pl.lit("Missing")).otherwise(pl.col(col)).alias(col)
                ])

        # Fill missing numerics
        numeric_cols = [col for col, dtype in dtype_spec.items() if dtype in (pl.Float64, pl.Int64) and col in df.columns]
        for col in numeric_cols:
            col_median = df[col].median()
            df = df.with_columns([pl.col(col).fill_null(col_median if col_median is not None else 0.0)])

        processed_dfs.append(df)
        offset += chunksize

    if processed_dfs:
        df_cleaned = pl.concat(processed_dfs)
        df_cleaned.write_parquet(output_path, compression="snappy")
        print(f" Done processing {quarter} and saved to: {output_path}")
    else:
        print(f" No data processed for {quarter}")
