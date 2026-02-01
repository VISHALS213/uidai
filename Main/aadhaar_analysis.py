"""
Aadhaar Enrolment and Updates Analysis
Comprehensive analysis to identify patterns, trends, anomalies, and predictive indicators
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import glob
import os

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class AadhaarAnalyzer:
    """Main class for Aadhaar data analysis"""
    
    def __init__(self, bio_path='api_data_aadhar_biometric', demo_path='api_data_aadhar_demographic'):
        self.bio_path = bio_path
        self.demo_path = demo_path
        self.bio_data = None
        self.demo_data = None
        self.merged_data = None
        
    def load_data(self):
        """Load and combine all CSV files from both datasets"""
        print("Loading biometric data...")
        bio_files = glob.glob(f"{self.bio_path}/*.csv")
        bio_dfs = []
        for file in bio_files:
            df = pd.read_csv(file)
            bio_dfs.append(df)
        self.bio_data = pd.concat(bio_dfs, ignore_index=True)
        
        print("Loading demographic data...")
        demo_files = glob.glob(f"{self.demo_path}/*.csv")
        demo_dfs = []
        for file in demo_files:
            df = pd.read_csv(file)
            demo_dfs.append(df)
        self.demo_data = pd.concat(demo_dfs, ignore_index=True)
        
        print(f"Biometric data shape: {self.bio_data.shape}")
        print(f"Demographic data shape: {self.demo_data.shape}")
        
    def clean_data(self):
        """Clean and preprocess the data"""
        print("\nCleaning data...")
        
        # Convert date columns
        self.bio_data['date'] = pd.to_datetime(self.bio_data['date'], format='%d-%m-%Y')
        self.demo_data['date'] = pd.to_datetime(self.demo_data['date'], format='%d-%m-%Y')
        
        # Remove duplicates
        self.bio_data = self.bio_data.drop_duplicates()
        self.demo_data = self.demo_data.drop_duplicates()
        
        # Handle missing values
        print(f"Biometric missing values:\n{self.bio_data.isnull().sum()}")
        print(f"\nDemographic missing values:\n{self.demo_data.isnull().sum()}")
        
        # Fill missing pincodes with 0 (will handle separately)
        self.bio_data['pincode'] = self.bio_data['pincode'].fillna(0).astype(int)
        self.demo_data['pincode'] = self.demo_data['pincode'].fillna(0).astype(int)
        
    def merge_datasets(self):
        """Merge biometric and demographic data"""
        print("\nMerging datasets...")
        
        # Merge on date, state, district, pincode
        self.merged_data = pd.merge(
            self.bio_data,
            self.demo_data,
            on=['date', 'state', 'district', 'pincode'],
            how='outer',
            suffixes=('_bio', '_demo')
        )
        
        # Fill NaN with 0 for counts
        count_columns = ['bio_age_5_17', 'bio_age_17_', 'demo_age_5_17', 'demo_age_17_']
        self.merged_data[count_columns] = self.merged_data[count_columns].fillna(0)
        
        # Create total columns
        self.merged_data['total_bio'] = self.merged_data['bio_age_5_17'] + self.merged_data['bio_age_17_']
        self.merged_data['total_demo'] = self.merged_data['demo_age_5_17'] + self.merged_data['demo_age_17_']
        self.merged_data['total_updates'] = self.merged_data['total_bio'] + self.merged_data['total_demo']
        
        # Calculate bio vs demo preference
        self.merged_data['bio_ratio'] = self.merged_data['total_bio'] / (self.merged_data['total_updates'] + 1)
        
        print(f"Merged data shape: {self.merged_data.shape}")
        
    def explore_data(self):
        """Generate basic exploratory statistics"""
        print("\n" + "="*80)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*80)
        
        print("\nBiometric Data Summary:")
        print(self.bio_data.describe())
        
        print("\nDemographic Data Summary:")
        print(self.demo_data.describe())
        
        print("\nMerged Data Summary:")
        print(self.merged_data.describe())
        
        # Top states by activity
        print("\nTop 10 States by Total Updates:")
        state_totals = self.merged_data.groupby('state')['total_updates'].sum().sort_values(ascending=False).head(10)
        print(state_totals)
        
        # Date range
        print(f"\nDate range: {self.merged_data['date'].min()} to {self.merged_data['date'].max()}")
        
        return {
            'bio_summary': self.bio_data.describe(),
            'demo_summary': self.demo_data.describe(),
            'merged_summary': self.merged_data.describe(),
            'top_states': state_totals
        }
    
    def temporal_analysis(self):
        """Analyze temporal patterns and trends"""
        print("\n" + "="*80)
        print("TEMPORAL PATTERN ANALYSIS")
        print("="*80)
        
        # Daily trends
        daily_trends = self.merged_data.groupby('date').agg({
            'total_bio': 'sum',
            'total_demo': 'sum',
            'total_updates': 'sum'
        }).reset_index()
        
        # Weekly trends
        self.merged_data['week'] = self.merged_data['date'].dt.isocalendar().week
        weekly_trends = self.merged_data.groupby('week').agg({
            'total_bio': 'sum',
            'total_demo': 'sum',
            'total_updates': 'sum'
        }).reset_index()
        
        # Day of week patterns
        self.merged_data['day_of_week'] = self.merged_data['date'].dt.day_name()
        dow_trends = self.merged_data.groupby('day_of_week').agg({
            'total_updates': 'sum'
        }).reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        
        print("\nDaily Update Statistics:")
        print(daily_trends.describe())
        
        print("\nDay of Week Pattern:")
        print(dow_trends)
        
        return {
            'daily_trends': daily_trends,
            'weekly_trends': weekly_trends,
            'dow_trends': dow_trends
        }
    
    def geographic_analysis(self):
        """Analyze geographic patterns"""
        print("\n" + "="*80)
        print("GEOGRAPHIC PATTERN ANALYSIS")
        print("="*80)
        
        # State-level analysis
        state_analysis = self.merged_data.groupby('state').agg({
            'total_bio': 'sum',
            'total_demo': 'sum',
            'total_updates': 'sum',
            'bio_ratio': 'mean',
            'district': 'nunique'
        }).reset_index()
        state_analysis.columns = ['state', 'total_bio', 'total_demo', 'total_updates', 
                                   'avg_bio_ratio', 'num_districts']
        state_analysis = state_analysis.sort_values('total_updates', ascending=False)
        
        # District-level analysis (top 20)
        district_analysis = self.merged_data.groupby(['state', 'district']).agg({
            'total_updates': 'sum'
        }).reset_index().sort_values('total_updates', ascending=False).head(20)
        
        print("\nTop 10 States by Total Updates:")
        print(state_analysis.head(10))
        
        print("\nTop 10 Districts by Total Updates:")
        print(district_analysis.head(10))
        
        # Bio vs Demo preference by state
        print("\nStates with Highest Biometric Preference:")
        print(state_analysis.nlargest(10, 'avg_bio_ratio')[['state', 'avg_bio_ratio']])
        
        print("\nStates with Highest Demographic Preference:")
        print(state_analysis.nsmallest(10, 'avg_bio_ratio')[['state', 'avg_bio_ratio']])
        
        return {
            'state_analysis': state_analysis,
            'district_analysis': district_analysis
        }
    
    def age_group_analysis(self):
        """Analyze patterns across age groups"""
        print("\n" + "="*80)
        print("AGE GROUP ANALYSIS")
        print("="*80)
        
        # Overall age distribution
        total_child_bio = self.merged_data['bio_age_5_17'].sum()
        total_adult_bio = self.merged_data['bio_age_17_'].sum()
        total_child_demo = self.merged_data['demo_age_5_17'].sum()
        total_adult_demo = self.merged_data['demo_age_17_'].sum()
        
        print(f"\nBiometric Updates:")
        print(f"  Children (5-17): {total_child_bio:,} ({total_child_bio/(total_child_bio+total_adult_bio)*100:.2f}%)")
        print(f"  Adults (17+): {total_adult_bio:,} ({total_adult_bio/(total_child_bio+total_adult_bio)*100:.2f}%)")
        
        print(f"\nDemographic Updates:")
        print(f"  Children (5-17): {total_child_demo:,} ({total_child_demo/(total_child_demo+total_adult_demo)*100:.2f}%)")
        print(f"  Adults (17+): {total_adult_demo:,} ({total_adult_demo/(total_child_demo+total_adult_demo)*100:.2f}%)")
        
        # State-wise age preferences
        state_age = self.merged_data.groupby('state').agg({
            'bio_age_5_17': 'sum',
            'bio_age_17_': 'sum',
            'demo_age_5_17': 'sum',
            'demo_age_17_': 'sum'
        })
        
        state_age['child_ratio'] = (state_age['bio_age_5_17'] + state_age['demo_age_5_17']) / \
                                    (state_age['bio_age_5_17'] + state_age['bio_age_17_'] + 
                                     state_age['demo_age_5_17'] + state_age['demo_age_17_'])
        
        print("\nStates with Highest Child Update Ratio:")
        print(state_age.nlargest(10, 'child_ratio')[['child_ratio']])
        
        return {
            'overall_age_stats': {
                'child_bio': total_child_bio,
                'adult_bio': total_adult_bio,
                'child_demo': total_child_demo,
                'adult_demo': total_adult_demo
            },
            'state_age_analysis': state_age
        }
    
    def detect_anomalies(self):
        """Detect anomalies in the data"""
        print("\n" + "="*80)
        print("ANOMALY DETECTION")
        print("="*80)
        
        # Statistical anomalies using IQR method
        def find_outliers(data, column):
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
            return outliers
        
        # Find days with unusually high activity
        daily_agg = self.merged_data.groupby('date').agg({
            'total_updates': 'sum'
        }).reset_index()
        
        high_activity_days = find_outliers(daily_agg, 'total_updates')
        print(f"\nDays with unusually high activity: {len(high_activity_days)}")
        if len(high_activity_days) > 0:
            print(high_activity_days.head(10))
        
        # Find locations with unusual patterns
        location_agg = self.merged_data.groupby(['state', 'district']).agg({
            'total_updates': 'sum'
        }).reset_index()
        
        unusual_locations = find_outliers(location_agg, 'total_updates')
        print(f"\nLocations with unusual activity levels: {len(unusual_locations)}")
        if len(unusual_locations) > 0:
            print(unusual_locations.head(10))
        
        # Find extreme bio/demo ratios
        extreme_bio_pref = self.merged_data[self.merged_data['bio_ratio'] > 0.95]
        extreme_demo_pref = self.merged_data[self.merged_data['bio_ratio'] < 0.05]
        
        print(f"\nLocations with extreme biometric preference (>95%): {len(extreme_bio_pref)}")
        print(f"Locations with extreme demographic preference (<5% bio): {len(extreme_demo_pref)}")
        
        return {
            'high_activity_days': high_activity_days,
            'unusual_locations': unusual_locations,
            'extreme_bio_pref': extreme_bio_pref,
            'extreme_demo_pref': extreme_demo_pref
        }
    
    def identify_trends(self):
        """Identify key trends in the data"""
        print("\n" + "="*80)
        print("TREND IDENTIFICATION")
        print("="*80)
        
        # Time-based trends
        daily_data = self.merged_data.groupby('date').agg({
            'total_bio': 'sum',
            'total_demo': 'sum',
            'bio_ratio': 'mean'
        }).reset_index()
        
        # Calculate growth rates
        daily_data['bio_growth'] = daily_data['total_bio'].pct_change() * 100
        daily_data['demo_growth'] = daily_data['total_demo'].pct_change() * 100
        
        print("\nAverage Daily Growth Rates:")
        print(f"  Biometric: {daily_data['bio_growth'].mean():.2f}%")
        print(f"  Demographic: {daily_data['demo_growth'].mean():.2f}%")
        
        # Trend in bio vs demo preference
        print("\nBio/Demo Preference Trend:")
        print(f"  Average bio ratio: {daily_data['bio_ratio'].mean():.3f}")
        print(f"  Trend (correlation with time): {daily_data['bio_ratio'].corr(pd.Series(range(len(daily_data)))):.3f}")
        
        # State-level trends
        state_growth = self.merged_data.groupby(['state', 'date']).agg({
            'total_updates': 'sum'
        }).reset_index()
        
        # Calculate growth for each state
        state_growth_rates = []
        for state in state_growth['state'].unique():
            state_data = state_growth[state_growth['state'] == state].sort_values('date')
            if len(state_data) > 1:
                first_week = state_data.head(7)['total_updates'].sum()
                last_week = state_data.tail(7)['total_updates'].sum()
                if first_week > 0:
                    growth = ((last_week - first_week) / first_week) * 100
                    state_growth_rates.append({'state': state, 'growth_rate': growth})
        
        state_growth_df = pd.DataFrame(state_growth_rates).sort_values('growth_rate', ascending=False)
        
        print("\nTop 10 Fastest Growing States:")
        print(state_growth_df.head(10))
        
        print("\nTop 10 Declining States:")
        print(state_growth_df.tail(10))
        
        return {
            'daily_trends': daily_data,
            'state_growth': state_growth_df
        }
    
    def generate_insights(self):
        """Generate actionable insights"""
        print("\n" + "="*80)
        print("KEY INSIGHTS & RECOMMENDATIONS")
        print("="*80)
        
        insights = []
        
        # 1. Overall adoption patterns
        total_updates = self.merged_data['total_updates'].sum()
        total_bio = self.merged_data['total_bio'].sum()
        total_demo = self.merged_data['total_demo'].sum()
        
        insights.append({
            'category': 'Overall Adoption',
            'insight': f'Total updates: {total_updates:,} (Bio: {total_bio:,}, Demo: {total_demo:,})',
            'recommendation': 'Focus resources on update type with lower adoption'
        })
        
        # 2. Geographic disparities
        state_stats = self.merged_data.groupby('state')['total_updates'].sum().sort_values(ascending=False)
        top_state = state_stats.index[0]
        bottom_state = state_stats.index[-1]
        
        insights.append({
            'category': 'Geographic Disparity',
            'insight': f'{top_state} has {state_stats.iloc[0]:,} updates vs {bottom_state} with {state_stats.iloc[-1]:,}',
            'recommendation': f'Increase awareness campaigns in {bottom_state} and similar low-adoption states'
        })
        
        # 3. Age group patterns
        child_total = (self.merged_data['bio_age_5_17'] + self.merged_data['demo_age_5_17']).sum()
        adult_total = (self.merged_data['bio_age_17_'] + self.merged_data['demo_age_17_']).sum()
        
        insights.append({
            'category': 'Age Demographics',
            'insight': f'Children updates: {child_total:,} ({child_total/(child_total+adult_total)*100:.1f}%)',
            'recommendation': 'Target campaigns based on underrepresented age groups'
        })
        
        # 4. Temporal patterns
        dow_stats = self.merged_data.groupby(self.merged_data['date'].dt.day_name())['total_updates'].sum()
        if len(dow_stats) > 0:
            peak_day = dow_stats.idxmax()
            low_day = dow_stats.idxmin()
            
            insights.append({
                'category': 'Temporal Patterns',
                'insight': f'Peak activity on {peak_day}, lowest on {low_day}',
                'recommendation': 'Optimize staffing and resources based on day-of-week patterns'
            })
        
        # 5. Update method preference
        bio_ratio_avg = self.merged_data['bio_ratio'].mean()
        
        insights.append({
            'category': 'Update Method',
            'insight': f'Average biometric preference: {bio_ratio_avg:.1%}',
            'recommendation': 'Ensure adequate infrastructure for preferred update method'
        })
        
        print("\nKey Insights:")
        for i, insight in enumerate(insights, 1):
            print(f"\n{i}. {insight['category']}")
            print(f"   Insight: {insight['insight']}")
            print(f"   Recommendation: {insight['recommendation']}")
        
        return insights
    
    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print("="*80)
        print("AADHAAR ENROLMENT & UPDATES ANALYSIS")
        print("="*80)
        
        # Load and prepare data
        self.load_data()
        self.clean_data()
        self.merge_datasets()
        
        # Run analyses
        eda_results = self.explore_data()
        temporal_results = self.temporal_analysis()
        geo_results = self.geographic_analysis()
        age_results = self.age_group_analysis()
        anomaly_results = self.detect_anomalies()
        trend_results = self.identify_trends()
        insights = self.generate_insights()
        
        return {
            'eda': eda_results,
            'temporal': temporal_results,
            'geographic': geo_results,
            'age': age_results,
            'anomalies': anomaly_results,
            'trends': trend_results,
            'insights': insights
        }


if __name__ == "__main__":
    # Initialize analyzer
    analyzer = AadhaarAnalyzer()
    
    # Run analysis
    results = analyzer.run_full_analysis()
    
    print("\n" + "="*80)
    print("Analysis complete! Results stored in memory.")
    print("="*80)