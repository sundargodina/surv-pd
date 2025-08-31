import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import gc
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Set style for academic plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class LongitudinalLoanEDA:

       
        self.file_path = file_path
        self.sample_loans = sample_loans
        self.df_sample = None
        self.loan_summary = None
        
    def load_longitudinal_sample(self):
        """
        Load stratified sample preserving longitudinal structure
        """
        print("Loading longitudinal sample with Polars...")
        print(f"Target sample: {self.sample_loans:,} unique loans")
        
        # Use lazy loading and scan to get loan metadata efficiently
        print("Scanning file for loan metadata...")
        df_meta = (
            pl.scan_parquet(self.file_path)
            .select([
                'LOAN_ID', 
                'event', 
                'any_delinquent', 
                'data_quarter',
                'time'
            ])
            .collect(streaming=True)
        )
        
        print(f"Scanned {len(df_meta):,} total observations")
        
        # Get loan-level characteristics for stratified sampling
        loan_info = (
            df_meta
            .group_by('LOAN_ID')
            .agg([
                pl.col('event').max().alias('max_event'),
                pl.col('any_delinquent').max().alias('ever_delinquent'),
                pl.col('data_quarter').count().alias('n_observations'),
                pl.col('data_quarter').min().alias('first_quarter'),
                pl.col('data_quarter').max().alias('last_quarter'),
                pl.col('time').min().alias('min_time'),
                pl.col('time').max().alias('max_time')
            ])
        )
        
        print(f"Unique loans: {len(loan_info):,}")
        print(f"Average observations per loan: {len(df_meta)/len(loan_info):.1f}")
        
        # Stratified sampling ensuring representation across key dimensions
        np.random.seed(42)  # Reproducibility
        
        # Sample by delinquency status to ensure representation
        never_dlq_loans = loan_info.filter(pl.col('ever_delinquent') == 0)['LOAN_ID'].to_list()
        ever_dlq_loans = loan_info.filter(pl.col('ever_delinquent') == 1)['LOAN_ID'].to_list()
        
        # Proportional sampling
        dlq_rate = len(ever_dlq_loans) / len(loan_info)
        target_dlq_loans = min(int(self.sample_loans * dlq_rate), len(ever_dlq_loans))
        target_current_loans = self.sample_loans - target_dlq_loans
        
        sampled_dlq = np.random.choice(ever_dlq_loans, 
                                      size=min(target_dlq_loans, len(ever_dlq_loans)), 
                                      replace=False).tolist()
        sampled_current = np.random.choice(never_dlq_loans, 
                                          size=min(target_current_loans, len(never_dlq_loans)), 
                                          replace=False).tolist()
        
        sampled_loan_ids = sampled_dlq + sampled_current
        
        print(f"Sampled {len(sampled_dlq):,} ever-delinquent loans")
        print(f"Sampled {len(sampled_current):,} never-delinquent loans")
        
        # Clean up metadata
        del df_meta, loan_info
        gc.collect()
        
        # Load full data for sampled loans using lazy evaluation
        print("Loading full data for sampled loans...")
        self.df_sample = (
            pl.scan_parquet(self.file_path)
            .filter(pl.col('LOAN_ID').is_in(sampled_loan_ids))
            .collect(streaming=True)
        )
        
        # Sort for proper panel structure
        self.df_sample = self.df_sample.sort(['LOAN_ID', 'time'])
        
        print(f"Final sample: {len(self.df_sample):,} observations")
        print(f"Unique loans: {self.df_sample['LOAN_ID'].n_unique():,}")
        print(f"Memory usage: {self.df_sample.estimated_size('mb'):.1f} MB")
        
        # Create loan-level summary
        self.create_loan_summary()
        
        return self.df_sample
    
    def create_loan_summary(self):
        """
        Create loan-level summary for cross-sectional analysis using Polars
        """
        print("Creating loan-level summary...")
        
        # Define aggregation expressions
        agg_exprs = [
            # Origination characteristics (first observation)
            pl.col('ORIG_DATE').first().alias('ORIG_DATE'),
            pl.col('CSCORE_B').first().alias('CSCORE_B'),
            pl.col('DTI').first().alias('DTI'),
            pl.col('ORIG_RATE').first().alias('ORIG_RATE'),
            pl.col('ORIG_UPB').first().alias('ORIG_UPB'),
            pl.col('OLTV').first().alias('OLTV'),
            pl.col('PURPOSE').first().alias('PURPOSE'),
            pl.col('STATE').first().alias('STATE'),
            pl.col('PROP').first().alias('PROP'),
            
            # Outcome variables (worst case)
            pl.col('event').max().alias('event_max'),
            pl.col('any_delinquent').max().alias('any_delinquent_max'),
            
            # Panel characteristics
            pl.col('time').count().alias('time_count'),
            pl.col('time').min().alias('time_min'),
            pl.col('time').max().alias('time_max'),
            pl.col('data_quarter').min().alias('first_quarter'),
            pl.col('data_quarter').max().alias('last_quarter'),
        ]
        
        # Add time-varying variable aggregations if they exist
        tv_vars = ['balance_ratio', 'credit_score_change', 'rate_change']
        for var in tv_vars:
            if var in self.df_sample.columns:
                agg_exprs.extend([
                    pl.col(var).last().alias(f'{var}_last'),
                    pl.col(var).mean().alias(f'{var}_mean'),
                    pl.col(var).std().alias(f'{var}_std')
                ])
        
        # Add economic variables if they exist
        econ_vars = ['interest_rates_PC1', 'credit_markets_PC1', 'labor_market_PC1', 'housing_market_PC1']
        for var in econ_vars:
            if var in self.df_sample.columns:
                agg_exprs.append(pl.col(var).mean().alias(f'{var}_mean'))
        
        # Check for serious_delinquent
        if 'serious_delinquent' in self.df_sample.columns:
            agg_exprs.append(pl.col('serious_delinquent').max().alias('serious_delinquent_max'))
        
        self.loan_summary = (
            self.df_sample
            .group_by('LOAN_ID')
            .agg(agg_exprs)
        )
        
        print(f"Loan summary created: {len(self.loan_summary)} loans, {len(self.loan_summary.columns)} variables")
    
    def panel_data_overview(self):
        """
        Comprehensive overview of panel data structure
        """
        print("\n" + "="*70)
        print("PANEL DATA STRUCTURE ANALYSIS")
        print("="*70)
        
        # Basic panel statistics
        n_loans = self.df_sample['LOAN_ID'].n_unique()
        n_obs = len(self.df_sample)
        avg_obs = n_obs / n_loans
        
        print(f"Unique loans (N): {n_loans:,}")
        print(f"Total observations: {n_obs:,}")
        print(f"Average observations per loan: {avg_obs:.1f}")
        
        # Time period coverage
        time_stats = (
            self.df_sample
            .group_by('LOAN_ID')
            .agg([
                pl.col('time').min().alias('min_time'),
                pl.col('time').max().alias('max_time'),
                pl.col('time').count().alias('count_time')
            ])
        )
        
        print(f"Time observations per loan - Min: {time_stats['count_time'].min()}, Max: {time_stats['count_time'].max()}")
        print(f"Time span - Min: {time_stats['min_time'].min():.1f}, Max: {time_stats['max_time'].max():.1f}")
        
        # Quarter coverage
        quarter_range = self.df_sample['data_quarter'].unique().sort()
        print(f"Quarter coverage: {quarter_range[0]} to {quarter_range[-1]} ({len(quarter_range)} quarters)")
        
        # Panel balance
        obs_counts = time_stats['count_time'].value_counts().sort('count_time')
        print(f"\nPanel balance analysis:")
        for row in obs_counts.head(10).iter_rows():
            obs_count, loan_count = row
            pct = loan_count / n_loans * 100
            print(f"  {obs_count} observations: {loan_count:,} loans ({pct:.1f}%)")
        
        # Event transitions
        event_transitions = (
            self.df_sample
            .group_by('LOAN_ID')
            .agg(pl.col('event').n_unique().alias('unique_events'))
        )
        
        transition_counts = event_transitions['unique_events'].value_counts().sort('unique_events')
        print(f"\nEvent transitions per loan:")
        for row in transition_counts.iter_rows():
            n_events, loan_count = row
            pct = loan_count / n_loans * 100
            print(f"  {n_events} different events: {loan_count:,} loans ({pct:.1f}%)")
    
    def longitudinal_delinquency_patterns(self):
        """
        Analyze delinquency patterns over time
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Longitudinal Delinquency Patterns', fontsize=16, y=0.95)
        
        # 1. Delinquency rates over time (quarterly)
        quarterly_dlq = (
            self.df_sample
            .group_by('data_quarter')
            .agg([
                pl.col('any_delinquent').mean().alias('any_delinquent_rate'),
                pl.col('LOAN_ID').n_unique().alias('active_loans')
            ])
            .sort('data_quarter')
        )
        
        # Add serious delinquent if available
        if 'serious_delinquent' in self.df_sample.columns:
            quarterly_dlq = quarterly_dlq.with_columns(
                self.df_sample.group_by('data_quarter')
                .agg(pl.col('serious_delinquent').mean().alias('serious_delinquent_rate'))
                .sort('data_quarter')['serious_delinquent_rate']
            )
        
        # Convert to pandas for plotting
        quarterly_pd = quarterly_dlq.to_pandas()
        quarterly_pd['data_quarter'] = pd.to_datetime(quarterly_pd['data_quarter'])
        quarterly_pd.set_index('data_quarter', inplace=True)
        
        ax1 = axes[0, 0]
        ax1.plot(quarterly_pd.index, quarterly_pd['any_delinquent_rate'] * 100, 
                marker='o', linewidth=2, label='Any Delinquent', color='orange')
        
        if 'serious_delinquent_rate' in quarterly_pd.columns:
            ax1.plot(quarterly_pd.index, quarterly_pd['serious_delinquent_rate'] * 100, 
                    marker='s', linewidth=2, label='Serious Delinquent', color='red')
        
        ax1.set_xlabel('Quarter')
        ax1.set_ylabel('Delinquency Rate (%)')
        ax1.set_title('Delinquency Rates Over Time')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Time to first delinquency
        first_dlq = (
            self.df_sample
            .filter(pl.col('any_delinquent') == 1)
            .group_by('LOAN_ID')
            .agg(pl.col('time').min().alias('first_dlq_time'))
        )['first_dlq_time'].to_numpy()
        
        if len(first_dlq) > 0:
            axes[0, 1].hist(first_dlq, bins=30, alpha=0.7, color='coral', edgecolor='black')
            median_time = np.median(first_dlq)
            axes[0, 1].axvline(median_time, color='red', linestyle='--', 
                              label=f'Median: {median_time:.1f}')
            axes[0, 1].set_xlabel('Time to First Delinquency')
            axes[0, 1].set_ylabel('Number of Loans')
            axes[0, 1].set_title('Distribution of Time to First Delinquency')
            axes[0, 1].legend()
        
        # 3. Survival analysis - efficient computation
        max_time = int(self.df_sample['time'].max())
        time_points = np.arange(0, max_time + 1, 1)
        survival_rates = []
        
        print("Computing survival curve...")
        for t in time_points[::2]:  # Sample every other time point for efficiency
            # Loans observed at time t or before
            loans_by_t = (
                self.df_sample
                .filter(pl.col('time') <= t)
                .select('LOAN_ID')
                .unique()
                .height
            )
            
            # Loans that became delinquent by time t
            dlq_by_t = (
                self.df_sample
                .filter((pl.col('time') <= t) & (pl.col('any_delinquent') == 1))
                .select('LOAN_ID')
                .unique()
                .height
            )
            
            if loans_by_t > 0:
                survival_rate = 1 - (dlq_by_t / loans_by_t)
                survival_rates.append(survival_rate)
            else:
                survival_rates.append(1.0)
        
        time_points_sampled = time_points[::2][:len(survival_rates)]
        axes[1, 0].plot(time_points_sampled, survival_rates, 
                       linewidth=2, color='green', marker='o', markersize=3)
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Survival Probability (No Delinquency)')
        axes[1, 0].set_title('Kaplan-Meier Style Survival Curve')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. State transition matrix (efficient computation)
        print("Computing transition matrix...")
        df_transitions = (
            self.df_sample
            .sort(['LOAN_ID', 'time'])
            .with_columns([
                pl.col('any_delinquent').shift(1).over('LOAN_ID').alias('dlq_lag')
            ])
            .filter(pl.col('dlq_lag').is_not_null())
        )
        
        # Compute transition probabilities
        transitions = (
            df_transitions
            .group_by(['dlq_lag', 'any_delinquent'])
            .agg(pl.len().alias('count'))
            .with_columns([
                pl.col('count').sum().over('dlq_lag').alias('total_from_state')
            ])
            .with_columns([
                (pl.col('count') / pl.col('total_from_state') * 100).alias('transition_prob')
            ])
        )
        
        # Create transition matrix for plotting
        transition_matrix = np.zeros((2, 2))
        for row in transitions.iter_rows():
            from_state, to_state, count, total, prob = row
            transition_matrix[int(from_state), int(to_state)] = prob
        
        sns.heatmap(transition_matrix, annot=True, fmt='.1f', cmap='Blues', 
                   ax=axes[1, 1], cbar_kws={'label': 'Transition Probability (%)'})
        axes[1, 1].set_title('Delinquency State Transition Matrix')
        axes[1, 1].set_xlabel('Current State (0=Current, 1=Delinquent)')
        axes[1, 1].set_ylabel('Previous State (0=Current, 1=Delinquent)')
        
        plt.tight_layout()
        plt.show()
    
    def time_varying_risk_analysis(self):
        """
        Analyze how risk factors evolve over time
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Time-Varying Risk Factor Evolution', fontsize=16, y=0.95)
        
        # 1. Credit score changes over time
        if 'credit_score_change' in self.df_sample.columns:
            credit_by_time = (
                self.df_sample
                .group_by('time')
                .agg(pl.col('credit_score_change').mean().alias('avg_credit_change'))
                .sort('time')
            ).to_pandas()
            
            axes[0, 0].plot(credit_by_time['time'], credit_by_time['avg_credit_change'], 
                           linewidth=2, marker='o', markersize=3, color='blue')
            axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
            axes[0, 0].set_xlabel('Time')
            axes[0, 0].set_ylabel('Average Credit Score Change')
            axes[0, 0].set_title('Credit Score Evolution')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Balance ratio over time by delinquency status
        if 'balance_ratio' in self.df_sample.columns:
            balance_by_dlq = (
                self.df_sample
                .group_by(['time', 'any_delinquent'])
                .agg(pl.col('balance_ratio').mean().alias('avg_balance_ratio'))
                .sort(['time', 'any_delinquent'])
            ).to_pandas()
            
            for dlq_status in [0, 1]:
                subset = balance_by_dlq[balance_by_dlq['any_delinquent'] == dlq_status]
                label = 'Current' if dlq_status == 0 else 'Delinquent'
                color = 'green' if dlq_status == 0 else 'red'
                axes[0, 1].plot(subset['time'], subset['avg_balance_ratio'], 
                               label=label, linewidth=2, color=color)
            
            axes[0, 1].set_xlabel('Time')
            axes[0, 1].set_ylabel('Average Balance Ratio')
            axes[0, 1].set_title('Balance Ratio by Delinquency Status')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Interest rate changes over time
        if 'rate_change' in self.df_sample.columns:
            rate_by_time = (
                self.df_sample
                .group_by('time')
                .agg(pl.col('rate_change').mean().alias('avg_rate_change'))
                .sort('time')
            ).to_pandas()
            
            axes[0, 2].plot(rate_by_time['time'], rate_by_time['avg_rate_change'], 
                           linewidth=2, marker='o', markersize=3, color='purple')
            axes[0, 2].axhline(y=0, color='red', linestyle='--', alpha=0.7)
            axes[0, 2].set_xlabel('Time')
            axes[0, 2].set_ylabel('Average Rate Change')
            axes[0, 2].set_title('Interest Rate Changes Over Time')
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4-6. Economic conditions over time
        econ_vars = [col for col in self.df_sample.columns if '_PC1' in col][:3]
        
        for i, var in enumerate(econ_vars):
            if i < 3:  # Only plot first 3 economic variables
                econ_data = (
                    self.df_sample
                    .group_by('data_quarter')
                    .agg(pl.col(var).mean().alias(f'avg_{var}'))
                    .sort('data_quarter')
                ).to_pandas()
                
                econ_data['data_quarter'] = pd.to_datetime(econ_data['data_quarter'])
                
                axes[1, i].plot(econ_data['data_quarter'], econ_data[f'avg_{var}'], 
                               linewidth=2, marker='o', markersize=3)
                axes[1, i].set_xlabel('Quarter')
                axes[1, i].set_ylabel(var.replace('_', ' ').title())
                axes[1, i].set_title(f'{var.replace("_", " ").title()} Over Time')
                axes[1, i].tick_params(axis='x', rotation=45)
                axes[1, i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(econ_vars), 3):
            axes[1, i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def within_loan_analysis(self):
        """
        Analyze within-loan variation and patterns
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Within-Loan Variation Analysis', fontsize=16, y=0.95)
        
        # 1. Distribution of observations per loan
        obs_per_loan = (
            self.df_sample
            .group_by('LOAN_ID')
            .agg(pl.len().alias('obs_count'))
        )['obs_count'].to_numpy()
        
        axes[0, 0].hist(obs_per_loan, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        mean_obs = np.mean(obs_per_loan)
        axes[0, 0].axvline(mean_obs, color='red', linestyle='--', 
                          label=f'Mean: {mean_obs:.1f}')
        axes[0, 0].set_xlabel('Number of Observations per Loan')
        axes[0, 0].set_ylabel('Number of Loans')
        axes[0, 0].set_title('Distribution of Panel Length')
        axes[0, 0].legend()
        
        # 2. Within-loan variation in key variables
        key_vars = ['credit_score_change', 'balance_ratio', 'rate_change']
        available_vars = [var for var in key_vars if var in self.df_sample.columns]
        
        if available_vars:
            within_std_stats = []
            var_names = []
            
            for var in available_vars:
                within_std = (
                    self.df_sample
                    .group_by('LOAN_ID')
                    .agg(pl.col(var).std().alias('within_std'))
                ).filter(pl.col('within_std').is_not_null())['within_std'].mean()
                
                within_std_stats.append(within_std)
                var_names.append(var.replace('_', ' ').title())
            
            x_pos = range(len(within_std_stats))
            axes[0, 1].bar(x_pos, within_std_stats, alpha=0.7, color='orange')
            axes[0, 1].set_xticks(x_pos)
            axes[0, 1].set_xticklabels(var_names, rotation=45)
            axes[0, 1].set_ylabel('Mean Within-Loan Standard Deviation')
            axes[0, 1].set_title('Within-Loan Variability')
        
        # 3. Loan age effects
        loan_age_effects = (
            self.df_sample
            .group_by('time')
            .agg([
                pl.col('any_delinquent').mean().alias('dlq_rate'),
                pl.col('credit_score_change').mean().alias('avg_credit_change') if 'credit_score_change' in self.df_sample.columns else pl.lit(0).alias('avg_credit_change'),
                pl.col('balance_ratio').mean().alias('avg_balance_ratio') if 'balance_ratio' in self.df_sample.columns else pl.lit(0).alias('avg_balance_ratio')
            ])
            .sort('time')
        ).to_pandas()
        
        ax1 = axes[1, 0]
        ax2 = ax1.twinx()
        
        line1 = ax1.plot(loan_age_effects['time'], loan_age_effects['dlq_rate'] * 100, 
                        color='red', marker='o', label='Delinquency Rate (%)', linewidth=2)
        
        if 'credit_score_change' in self.df_sample.columns:
            line2 = ax2.plot(loan_age_effects['time'], loan_age_effects['avg_credit_change'], 
                            color='blue', marker='s', label='Credit Score Change', linewidth=2)
        
        ax1.set_xlabel('Loan Age (Time)')
        ax1.set_ylabel('Delinquency Rate (%)', color='red')
        ax2.set_ylabel('Average Credit Score Change', color='blue')
        ax1.set_title('Loan Age Effects')
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # 4. Event sequence complexity
        event_complexity = (
            self.df_sample
            .group_by('LOAN_ID')
            .agg([
                pl.col('event').count().alias('total_obs'),
                pl.col('event').n_unique().alias('unique_events')
            ])
        ).to_pandas()
        
        axes[1, 1].scatter(event_complexity['total_obs'], event_complexity['unique_events'], 
                          alpha=0.6, s=30)
        axes[1, 1].set_xlabel('Number of Observations')
        axes[1, 1].set_ylabel('Number of Unique Events')
        axes[1, 1].set_title('Event Sequence Complexity')
        
        # Add diagonal line
        max_val = max(event_complexity['total_obs'].max(), event_complexity['unique_events'].max())
        axes[1, 1].plot([0, max_val], [0, max_val], 'r--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
    
    def cross_sectional_analysis(self):
        """
        Cross-sectional analysis using loan-level aggregated data
        """
        if self.loan_summary is None:
            print("Loan summary not available. Run load_longitudinal_sample first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Cross-Sectional Analysis (Loan-Level Aggregated)', fontsize=16, y=0.95)
        
        # Convert to pandas for plotting
        loan_summary_pd = self.loan_summary.to_pandas()
        
        # 1. Credit score vs final outcome
        if 'CSCORE_B' in loan_summary_pd.columns and 'any_delinquent_max' in loan_summary_pd.columns:
            current_loans = loan_summary_pd[loan_summary_pd['any_delinquent_max'] == 0]['CSCORE_B'].dropna()
            dlq_loans = loan_summary_pd[loan_summary_pd['any_delinquent_max'] == 1]['CSCORE_B'].dropna()
            
            axes[0, 0].hist(current_loans, bins=30, alpha=0.6, 
                           label=f'Never Delinquent (n={len(current_loans):,})', 
                           color='green', density=True)
            axes[0, 0].hist(dlq_loans, bins=30, alpha=0.6, 
                           label=f'Ever Delinquent (n={len(dlq_loans):,})', 
                           color='red', density=True)
            axes[0, 0].set_xlabel('Origination Credit Score')
            axes[0, 0].set_ylabel('Density')
            axes[0, 0].set_title('Credit Score Distribution by Final Outcome')
            axes[0, 0].legend()
        
        # 2. Time-varying variable summaries
        if 'credit_score_change_mean' in loan_summary_pd.columns:
            current_cc = loan_summary_pd[loan_summary_pd['any_delinquent_max'] == 0]['credit_score_change_mean'].dropna()
            dlq_cc = loan_summary_pd[loan_summary_pd['any_delinquent_max'] == 1]['credit_score_change_mean'].dropna()
            
            axes[0, 1].hist(current_cc, bins=30, alpha=0.6, 
                           label='Never Delinquent', color='green', density=True)
            axes[0, 1].hist(dlq_cc, bins=30, alpha=0.6, 
                           label='Ever Delinquent', color='red', density=True)
            axes[0, 1].set_xlabel('Average Credit Score Change')
            axes[0, 1].set_ylabel('Density')
            axes[0, 1].set_title('Average Credit Score Change by Outcome')
            axes[0, 1].legend()
        
        # 3. Panel length vs outcome
        if 'time_count' in loan_summary_pd.columns:
            panel_length = loan_summary_pd['time_count']
            length_bins = pd.cut(panel_length, bins=10)
            dlq_rate_by_length = []
            
            for interval in length_bins.cat.categories:
                mask = length_bins == interval
                if mask.sum() > 0:
                    dlq_rate = loan_summary_pd.loc[mask, 'any_delinquent_max'].mean() * 100
                    dlq_rate_by_length.append(dlq_rate)
                else:
                    dlq_rate_by_length.append(0)
            
            x_pos = range(len(dlq_rate_by_length))
            axes[1, 0].bar(x_pos, dlq_rate_by_length, alpha=0.7, color='coral')
            axes[1, 0].set_xticks(x_pos)
            axes[1, 0].set_xticklabels([f'{int(cat.left)}-{int(cat.right)}' 
                                       for cat in length_bins.cat.categories], rotation=45)
            axes[1, 0].set_xlabel('Number of Observations')
            axes[1, 0].set_ylabel('Delinquency Rate (%)')
            axes[1, 0].set_title('Delinquency Rate by Panel Length')
        
        # 4. State-level summary
        if 'STATE' in loan_summary_pd.columns:
            state_summary = loan_summary_pd.groupby('STATE').agg({
                'any_delinquent_max': ['count', 'mean'],
                'ORIG_UPB': 'mean'
            })
            
            state_summary.columns = ['loan_count', 'dlq_rate', 'avg_balance']
            top_states = state_summary.nlargest(15, 'loan_count')
            
            ax1 = axes[1, 1]
            ax2 = ax1.twinx()
            
            x_pos = range(len(top_states))
            bars = ax1.bar(x_pos, top_states['loan_count'], alpha=0.6, color='lightblue')
            line = ax2.plot(x_pos, top_states['dlq_rate'] * 100, color='red', 
                           marker='o', linewidth=2)
            
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(top_states.index, rotation=45)
            ax1.set_ylabel('Number of Loans', color='blue')
            ax2.set_ylabel('Delinquency Rate (%)', color='red')
            ax1.set_title('Top States: Volume vs Delinquency Rate')
        
        plt.tight_layout()
        plt.show()
    
    def panel_correlation_analysis(self):
        """
        Correlation analysis accounting for panel structure using Polars
        """
        print("\n" + "="*70)
        print("PANEL CORRELATION ANALYSIS")
        print("="*70)
        
        # Key variables for correlation
        key_vars = ['credit_score_change', 'balance_ratio', 'rate_change', 'any_delinquent']
        econ_vars = [col for col in self.df_sample.columns if '_PC1' in col]
        analysis_vars = key_vars + econ_vars
        
        available_vars = [var for var in analysis_vars if var in self.df_sample.columns]
        
        if len(available_vars) > 2:
            # Overall correlation matrix (convert to pandas for correlation)
            overall_corr_data = self.df_sample.select(available_vars).to_pandas()
            overall_corr = overall_corr_data.corr()
            
            # Within-group correlations (demeaned data) - efficient with Polars
            print("Computing within-loan correlations...")
            
            # Create demeaned variables using Polars window functions
            demean_exprs = []
            for var in available_vars:
                demean_exprs.append(
                    (pl.col(var) - pl.col(var).mean().over('LOAN_ID')).alias(f'{var}_within')
                )
            
            df_demeaned = (
                self.df_sample
                .with_columns(demean_exprs)
                .select([f'{var}_within' for var in available_vars])
            ).to_pandas()
            
            # Rename columns back for correlation matrix
            df_demeaned.columns = available_vars
            within_corr = df_demeaned.corr()
            
            # Plot both correlation matrices
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            sns.heatmap(overall_corr, annot=True, fmt='.2f', cmap='RdBu_r', 
                       center=0, square=True, ax=ax1, cbar_kws={'shrink': 0.8})
            ax1.set_title('Overall Correlation Matrix')
            
            sns.heatmap(within_corr, annot=True, fmt='.2f', cmap='RdBu_r', 
                       center=0, square=True, ax=ax2, cbar_kws={'shrink': 0.8})
            ax2.set_title('Within-Loan Correlation Matrix')
            
            plt.tight_layout()
            plt.show()
            
            # Print key differences
            print("Key correlation differences (Overall vs Within-loan):")
            if 'any_delinquent' in available_vars:
                for var in available_vars:
                    if var != 'any_delinquent':
                        overall_val = overall_corr.loc['any_delinquent', var]
                        within_val = within_corr.loc['any_delinquent', var]
                        diff = overall_val - within_val
                        print(f"  {var}: Overall={overall_val:.3f}, Within={within_val:.3f}, Diff={diff:.3f}")
    
    def economic_cycles_analysis(self):
        """
        Analyze relationship between economic cycles and loan performance
        """
        econ_factors = [col for col in self.df_sample.columns if '_PC1' in col or '_PC2' in col]
        
        if not econ_factors:
            print("No economic factors available for analysis.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Economic Cycles and Loan Performance', fontsize=16, y=0.95)
        
        # 1. Economic conditions over time
        quarterly_econ = (
            self.df_sample
            .group_by('data_quarter')
            .agg([pl.col(factor).mean().alias(f'avg_{factor}') for factor in econ_factors[:4]])
            .sort('data_quarter')
        ).to_pandas()
        
        quarterly_econ['data_quarter'] = pd.to_datetime(quarterly_econ['data_quarter'])
        quarterly_econ.set_index('data_quarter', inplace=True)
        
        for factor in econ_factors[:4]:  # Plot first 4 factors
            col_name = f'avg_{factor}'
            if col_name in quarterly_econ.columns:
                axes[0, 0].plot(quarterly_econ.index, quarterly_econ[col_name], 
                               label=factor.replace('_PC1', '').replace('_', ' ').title(), 
                               linewidth=2, marker='o', markersize=3)
        
        axes[0, 0].set_xlabel('Quarter')
        axes[0, 0].set_ylabel('Factor Score')
        axes[0, 0].set_title('Economic Factors Over Time')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Economic stress vs delinquency (if financial stress available)
        stress_factor = None
        for factor in econ_factors:
            if 'stress' in factor.lower() or 'financial' in factor.lower():
                stress_factor = factor
                break
        
        if not stress_factor and econ_factors:
            stress_factor = econ_factors[0]  # Use first economic factor
        
        if stress_factor:
            # Create stress bins and compute delinquency rates
            stress_data = (
                self.df_sample
                .with_columns([
                    pl.col(stress_factor).qcut(10, labels=[f'Q{i+1}' for i in range(10)]).alias('stress_decile')
                ])
                .group_by('stress_decile')
                .agg([
                    pl.col('any_delinquent').mean().alias('dlq_rate'),
                    pl.col(stress_factor).mean().alias('avg_stress')
                ])
                .sort('avg_stress')
            ).to_pandas()
            
            x_pos = range(len(stress_data))
            axes[0, 1].bar(x_pos, stress_data['dlq_rate'] * 100, alpha=0.7, color='red')
            axes[0, 1].set_xticks(x_pos)
            axes[0, 1].set_xticklabels(stress_data['stress_decile'], rotation=45)
            axes[0, 1].set_xlabel(f'{stress_factor.replace("_", " ").title()} Deciles')
            axes[0, 1].set_ylabel('Delinquency Rate (%)')
            axes[0, 1].set_title(f'{stress_factor.replace("_", " ").title()} vs Delinquency Rate')
        
        # 3. Housing market conditions
        if 'housing_market_PC1' in self.df_sample.columns:
            housing_quarterly = (
                self.df_sample
                .group_by('data_quarter')
                .agg([
                    pl.col('housing_market_PC1').mean().alias('housing_avg'),
                    pl.col('any_delinquent').mean().alias('dlq_rate')
                ])
                .sort('data_quarter')
            ).to_pandas()
            
            housing_quarterly['data_quarter'] = pd.to_datetime(housing_quarterly['data_quarter'])
            housing_quarterly.set_index('data_quarter', inplace=True)
            
            ax1 = axes[1, 0]
            ax2 = ax1.twinx()
            
            line1 = ax1.plot(housing_quarterly.index, housing_quarterly['housing_avg'], 
                            color='green', linewidth=2, label='Housing Market PC1')
            line2 = ax2.plot(housing_quarterly.index, housing_quarterly['dlq_rate'] * 100, 
                            color='red', linewidth=2, label='Delinquency Rate (%)')
            
            ax1.set_xlabel('Quarter')
            ax1.set_ylabel('Housing Market Score', color='green')
            ax2.set_ylabel('Delinquency Rate (%)', color='red')
            ax1.set_title('Housing Market vs Delinquency')
            ax1.tick_params(axis='x', rotation=45)
            
            # Combined legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # 4. Credit market conditions impact
        if 'credit_markets_PC1' in self.df_sample.columns:
            credit_impact = (
                self.df_sample
                .group_by('data_quarter')
                .agg([
                    pl.col('credit_markets_PC1').mean().alias('credit_avg'),
                    pl.col('any_delinquent').sum().alias('dlq_count'),
                    pl.col('LOAN_ID').n_unique().alias('active_loans')
                ])
                .with_columns([
                    (pl.col('dlq_count') / pl.col('active_loans') * 100).alias('dlq_rate')
                ])
                .sort('data_quarter')
            ).to_pandas()
            
            axes[1, 1].scatter(credit_impact['credit_avg'], credit_impact['dlq_rate'], 
                              alpha=0.7, s=50, color='purple')
            
            # Add trend line if we have enough data points
            if len(credit_impact) > 2:
                z = np.polyfit(credit_impact['credit_avg'], credit_impact['dlq_rate'], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(credit_impact['credit_avg'].min(), 
                                     credit_impact['credit_avg'].max(), 100)
                axes[1, 1].plot(x_trend, p(x_trend), "r--", alpha=0.8)
                
                # Calculate correlation
                corr = credit_impact['credit_avg'].corr(credit_impact['dlq_rate'])
                axes[1, 1].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                               transform=axes[1, 1].transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            axes[1, 1].set_xlabel('Credit Markets PC1')
            axes[1, 1].set_ylabel('Quarterly Delinquency Rate (%)')
            axes[1, 1].set_title('Credit Market Conditions vs Delinquency')
        
        plt.tight_layout()
        plt.show()
    
    def summary_statistics_panel(self):
        
        # Basic panel statistics
        n_loans = self.df_sample['LOAN_ID'].n_unique()
        n_obs = len(self.df_sample)
        
        print(f"Panel Structure:")
        print(f"  Unique loans (N): {n_loans:,}")
        print(f"  Total observations: {n_obs:,}")
        print(f"  Average observations per loan: {n_obs/n_loans:.1f}")
        
        # Time-invariant variables (origination characteristics)
        time_invariant = ['CSCORE_B', 'DTI', 'ORIG_RATE', 'ORIG_UPB', 'OLTV']
        available_ti = [var for var in time_invariant if var in self.df_sample.columns]
        
        if available_ti:
            print(f"\nTime-Invariant Variables (Origination Characteristics):")
            ti_data = (
                self.df_sample
                .group_by('LOAN_ID')
                .agg([pl.col(var).first() for var in available_ti])
                .select(available_ti)
            ).to_pandas()
            
            ti_stats = ti_data.describe()
            print(ti_stats.round(3).to_string())
        
        # Time-varying variables
        time_varying = ['credit_score_change', 'balance_ratio', 'rate_change', 'any_delinquent']
        available_tv = [var for var in time_varying if var in self.df_sample.columns]
        
        if available_tv:
            print(f"\nTime-Varying Variables (All Observations):")
            tv_data = self.df_sample.select(available_tv).to_pandas()
            tv_stats = tv_data.describe()
            print(tv_stats.round(3).to_string())
            
            # Within-loan variation using Polars
            print(f"\nWithin-Loan Variation (Standard Deviation):")
            within_stats_data = []
            
            for var in available_tv:
                within_std = (
                    self.df_sample
                    .group_by('LOAN_ID')
                    .agg(pl.col(var).std().alias('within_std'))
                    .filter(pl.col('within_std').is_not_null())
                )['within_std'].to_pandas()
                
                within_stats_data.append({
                    'Variable': var,
                    'count': len(within_std),
                    'mean': within_std.mean(),
                    'std': within_std.std(),
                    'min': within_std.min(),
                    '25%': within_std.quantile(0.25),
                    '50%': within_std.median(),
                    '75%': within_std.quantile(0.75),
                    'max': within_std.max()
                })
            
            within_stats_df = pd.DataFrame(within_stats_data).set_index('Variable')
            print(within_stats_df.round(3).to_string())
        
        # Outcome variable summary
        if 'any_delinquent' in self.df_sample.columns:
            print(f"\nOutcome Variables:")
            
            # Overall delinquency rate
            overall_dlq_rate = self.df_sample['any_delinquent'].mean() * 100
            print(f"  Overall delinquency rate: {overall_dlq_rate:.2f}%")
            
            # Ever delinquent (loan level)
            ever_dlq_rate = (
                self.df_sample
                .group_by('LOAN_ID')
                .agg(pl.col('any_delinquent').max())
            )['any_delinquent'].mean() * 100
            print(f"  Ever delinquent rate (loan-level): {ever_dlq_rate:.2f}%")
            print("Computing transition probabilities...")
            transitions = (
                self.df_sample
                .sort(['LOAN_ID', 'time'])
                .with_columns([
                    pl.col('any_delinquent').shift(1).over('LOAN_ID').alias('dlq_lag')
                ])
                .filter(pl.col('dlq_lag').is_not_null())
                .group_by(['dlq_lag', 'any_delinquent'])
                .agg(pl.len().alias('count'))
                .with_columns([
                    pl.col('count').sum().over('dlq_lag').alias('total_from_state')
                ])
                .with_columns([
                    (pl.col('count') / pl.col('total_from_state')).alias('transition_prob')
                ])
            )
            
            print(f"  Transition probabilities:")
            for row in transitions.iter_rows():
                from_state, to_state, count, total, prob = row
                state_names = {0: 'Current', 1: 'Delinquent'}
                print(f"    {state_names[from_state]} -> {state_names[to_state]}: {prob:.3f}")
    
    def advanced_panel_diagnostics(self):
        """
        Advanced diagnostics specific to panel data
        """
        print("\n" + "="*70)
        print("ADVANCED PANEL DIAGNOSTICS")
        print("="*70)
      
        exit_analysis = (
            self.df_sample
            .group_by('LOAN_ID')
            .agg([
                pl.col('time').max().alias('last_observed_time'),
                pl.col('any_delinquent').last().alias('final_status'),
                pl.col('event').last().alias('final_event')
            ])
        )
        
        # Analyze exit patterns
        max_possible_time = self.df_sample['time'].max()
        early_exit = exit_analysis.filter(pl.col('last_observed_time') < max_possible_time * 0.9)
        
        print(f"  • Loans with early exit: {len(early_exit):,} ({len(early_exit)/len(exit_analysis)*100:.1f}%)")
        
        # Exit by final status
        exit_by_status = (
            early_exit
            .group_by('final_status')
            .agg(pl.len().alias('count'))
        )
        
        for row in exit_by_status.iter_rows():
            status, count = row
            status_name = 'Current' if status == 0 else 'Delinquent'
            print(f"    - {status_name} at exit: {count:,}")
        
        # 2. Time gaps analysis
        print("\nTime Gaps Analysis:")
        time_gaps = (
            self.df_sample
            .sort(['LOAN_ID', 'time'])
            .with_columns([
                pl.col('time').diff().over('LOAN_ID').alias('time_gap')
            ])
            .filter(pl.col('time_gap').is_not_null() & (pl.col('time_gap') != 1))
        )
        
        if len(time_gaps) > 0:
            gap_stats = time_gaps['time_gap'].to_pandas().describe()
            print(f"  • Loans with time gaps: {time_gaps['LOAN_ID'].n_unique():,}")
            print(f"  • Average gap size: {gap_stats['mean']:.1f} periods")
            print(f"  • Max gap size: {gap_stats['max']:.0f} periods")
        else:
            print("  • No significant time gaps detected")
        
        # 3. Panel balance assessment
        print("\nPanel Balance Assessment:")
        
        # Check for balanced vs unbalanced panel
        obs_per_loan = (
            self.df_sample
            .group_by('LOAN_ID')
            .agg(pl.len().alias('obs_count'))
        )['obs_count']
        
        mode_obs = obs_per_loan.mode()[0]  # Most common number of observations
        balanced_loans = (obs_per_loan == mode_obs).sum()
        
        print(f"  • Most common panel length: {mode_obs} observations")
        print(f"  • Loans with modal length: {balanced_loans:,} ({balanced_loans/len(obs_per_loan)*100:.1f}%)")
        print(f"  • Panel type: {'Strongly Balanced' if balanced_loans/len(obs_per_loan) > 0.8 else 'Unbalanced'}")
        
        return time_gaps, exit_analysis
    
    def run_complete_longitudinal_analysis(self):
        """
        Run complete longitudinal EDA analysis optimized for large datasets
        """
        start_time = datetime.now()
        print("Starting comprehensive longitudinal loan EDA with Polars optimization...")
        print(f"Targeting {self.sample_loans:,} loans from 346M observations")
        
        # Load data
        self.load_longitudinal_sample()
        
        # Panel structure overview
        self.panel_data_overview()
        
        # Core longitudinal analyses
        print("\nGenerating longitudinal delinquency patterns...")
        self.longitudinal_delinquency_patterns()
        
        print("\nAnalyzing time-varying risk factors...")
        self.time_varying_risk_analysis()
        
        print("\nPerforming within-loan analysis...")
        self.within_loan_analysis()
        
        # Cross-sectional perspective
        print("\nConducting cross-sectional analysis...")
        self.cross_sectional_analysis()
        
        # Advanced analyses
        print("\nPerforming panel correlation analysis...")
        self.panel_correlation_analysis()
        
        print("\nAnalyzing economic cycles...")
        self.economic_cycles_analysis()
        
        # Panel diagnostics
        print("\nRunning advanced panel diagnostics...")
        time_gaps, exit_analysis = self.advanced_panel_diagnostics()
        
        # Summary statistics
        self.summary_statistics_panel()
        
        # Performance summary
        end_time = datetime.now()
        runtime = (end_time - start_time).total_seconds()
        
        # Key insights for dissertation
        print(f"\n" + "="*70)
        print("KEY LONGITUDINAL INSIGHTS FOR DISSERTATION")
        print("="*70)
        
        n_loans = self.df_sample['LOAN_ID'].n_unique()
        n_obs = len(self.df_sample)
        avg_obs = n_obs / n_loans
        
        print(f"Panel Data Structure:")
        print(f"  • {n_loans:,} unique loans tracked longitudinally")
        print(f"  • {n_obs:,} total loan-period observations")
        print(f"  • Average {avg_obs:.1f} observations per loan")
        print(f"  • Represents {(n_loans/346_000_000)*100:.4f}% sample of universe")
        
        # Time patterns from loan summary
        if self.loan_summary is not None:
            time_stats = self.loan_summary.select(['time_count', 'time_min', 'time_max']).to_pandas()
            print(f"  • Loans observed for {time_stats['time_count'].min()}-{time_stats['time_count'].max()} periods")
            print(f"  • Time span: {time_stats['time_min'].min():.1f} to {time_stats['time_max'].max():.1f}")
        
        # Delinquency dynamics
        overall_dlq = self.df_sample['any_delinquent'].mean() * 100
        ever_dlq = (
            self.df_sample
            .group_by('LOAN_ID')
            .agg(pl.col('any_delinquent').max())
        )['any_delinquent'].mean() * 100
        
        print(f"\nDelinquency Patterns:")
        print(f"  • {overall_dlq:.1f}% of all loan-periods show delinquency")
        print(f"  • {ever_dlq:.1f}% of loans ever become delinquent")
        print(f"  • Enables survival analysis and hazard modeling")
        
        # Time-varying characteristics
        if 'credit_score_change' in self.df_sample.columns:
            credit_stats = self.df_sample['credit_score_change'].to_pandas()
            avg_credit_change = credit_stats.mean()
            
            within_std = (
                self.df_sample
                .group_by('LOAN_ID')
                .agg(pl.col('credit_score_change').std())
                .filter(pl.col('credit_score_change').is_not_null())
            )['credit_score_change'].mean()
            
            print(f"  • Average credit score change: {avg_credit_change:.1f} points")
            print(f"  • Average within-loan credit score volatility: {within_std:.1f}")
        
        # Economic sensitivity
        econ_factors = [col for col in self.df_sample.columns if '_PC1' in col]
        if econ_factors:
            print(f"  • {len(econ_factors)} economic factors captured")
            print(f"  • Economic cycle effects identifiable through time variation")

        
        # Data quality assessment
        print(f"\nData Quality Assessment:")
        missing_summary = []
        for col in self.df_sample.columns:
            missing_pct = (self.df_sample[col].null_count() / len(self.df_sample)) * 100
            if missing_pct > 0:
                missing_summary.append(f"  • {col}: {missing_pct:.1f}% missing")
        
        if missing_summary:
            print("Missing data patterns:")
            for item in missing_summary[:10]:  # Show top 10
                print(item)
        else:
            print("  • No missing data detected in key variables")
        
        return self.df_sample, self.loan_summary
         
      

if __name__ == "__main__":
    
    eda = LongitudinalLoanEDA(
        "path", 
        sample_loans=75000
    )
    
    # Run complete longitudinal analysis
       
    panel_data, loan_summary = eda.run_complete_longitudinal_analysis()
    
    # Export results for further analysis
    panel_file, summary_file = eda.export_results_for_analysis("dissertation_loan_analysis")  
    # Memory cleanup
    del panel_data, loan_summary
    gc.collect()
