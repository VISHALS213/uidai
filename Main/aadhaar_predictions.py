"""
Predictive Modeling for Aadhaar Updates
Machine learning models to forecast trends and identify key factors
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings

warnings.filterwarnings('ignore')


class AadhaarPredictor:
    """Predictive modeling for Aadhaar data"""
    
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.data = analyzer.merged_data.copy()
        self.models = {}
        self.encoders = {}
        self.feature_importance = {}
        
    def prepare_features(self):
        """Prepare features for modeling"""
        print("\nPreparing features for modeling...")
        
        # Create temporal features
        self.data['day_of_week'] = self.data['date'].dt.dayofweek
        self.data['day_of_month'] = self.data['date'].dt.day
        self.data['week_of_year'] = self.data['date'].dt.isocalendar().week
        self.data['is_weekend'] = self.data['day_of_week'].isin([5, 6]).astype(int)
        
        # Encode categorical variables
        for col in ['state', 'district']:
            le = LabelEncoder()
            self.data[f'{col}_encoded'] = le.fit_transform(self.data[col].astype(str))
            self.encoders[col] = le
        
        # Create lag features (previous day's activity)
        self.data = self.data.sort_values(['state', 'district', 'date'])
        for col in ['total_updates', 'total_bio', 'total_demo']:
            self.data[f'{col}_lag1'] = self.data.groupby(['state', 'district'])[col].shift(1)
            self.data[f'{col}_lag7'] = self.data.groupby(['state', 'district'])[col].shift(7)
        
        # Fill NaN lag values with 0
        lag_cols = [col for col in self.data.columns if 'lag' in col]
        self.data[lag_cols] = self.data[lag_cols].fillna(0)
        
        # Calculate rolling averages
        for col in ['total_updates', 'total_bio', 'total_demo']:
            self.data[f'{col}_rolling_7'] = self.data.groupby(['state', 'district'])[col].transform(
                lambda x: x.rolling(window=7, min_periods=1).mean()
            )
        
        print(f"Feature preparation complete. Total features: {self.data.shape[1]}")
        
    def train_update_predictor(self):
        """Train model to predict total updates"""
        print("\n" + "="*80)
        print("TRAINING UPDATE PREDICTION MODEL")
        print("="*80)
        
        # Select features
        feature_cols = [
            'state_encoded', 'district_encoded',
            'day_of_week', 'day_of_month', 'week_of_year', 'is_weekend',
            'total_updates_lag1', 'total_updates_lag7', 'total_updates_rolling_7',
            'total_bio_lag1', 'total_demo_lag1',
            'bio_age_5_17', 'bio_age_17_', 'demo_age_5_17', 'demo_age_17_'
        ]
        
        target = 'total_updates'
        
        # Prepare data
        df_model = self.data[feature_cols + [target]].dropna()
        X = df_model[feature_cols]
        y = df_model[target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"\nTraining set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # Train Random Forest
        print("\nTraining Random Forest...")
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = rf_model.predict(X_train)
        y_pred_test = rf_model.predict(X_test)
        
        # Evaluate
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        print("\nModel Performance:")
        print(f"  Training MAE: {train_mae:.2f}")
        print(f"  Testing MAE: {test_mae:.2f}")
        print(f"  Training RMSE: {train_rmse:.2f}")
        print(f"  Testing RMSE: {test_rmse:.2f}")
        print(f"  Training R²: {train_r2:.4f}")
        print(f"  Testing R²: {test_r2:.4f}")
        
        # Feature importance
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(importance_df.head(10))
        
        # Store model and results
        self.models['update_predictor'] = rf_model
        self.feature_importance['update_predictor'] = importance_df
        
        return {
            'model': rf_model,
            'metrics': {
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_r2': train_r2,
                'test_r2': test_r2
            },
            'feature_importance': importance_df
        }
    
    def train_bio_preference_predictor(self):
        """Train model to predict biometric preference"""
        print("\n" + "="*80)
        print("TRAINING BIOMETRIC PREFERENCE MODEL")
        print("="*80)
        
        # Filter data where updates exist
        df_filtered = self.data[self.data['total_updates'] > 0].copy()
        
        # Select features
        feature_cols = [
            'state_encoded', 'district_encoded',
            'day_of_week', 'week_of_year',
            'total_updates',
            'bio_age_5_17', 'bio_age_17_',
            'demo_age_5_17', 'demo_age_17_'
        ]
        
        target = 'bio_ratio'
        
        # Prepare data
        df_model = df_filtered[feature_cols + [target]].dropna()
        X = df_model[feature_cols]
        y = df_model[target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"\nTraining set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # Train Gradient Boosting
        print("\nTraining Gradient Boosting...")
        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=7,
            learning_rate=0.1,
            random_state=42
        )
        gb_model.fit(X_train, y_train)
        
        # Predictions
        y_pred_test = gb_model.predict(X_test)
        
        # Evaluate
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_r2 = r2_score(y_test, y_pred_test)
        
        print("\nModel Performance:")
        print(f"  Testing MAE: {test_mae:.4f}")
        print(f"  Testing RMSE: {test_rmse:.4f}")
        print(f"  Testing R²: {test_r2:.4f}")
        
        # Feature importance
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': gb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop Features Influencing Bio Preference:")
        print(importance_df.head(10))
        
        # Store model
        self.models['bio_preference'] = gb_model
        self.feature_importance['bio_preference'] = importance_df
        
        return {
            'model': gb_model,
            'metrics': {
                'test_mae': test_mae,
                'test_rmse': test_rmse,
                'test_r2': test_r2
            },
            'feature_importance': importance_df
        }
    
    def forecast_next_week(self):
        """Forecast updates for next 7 days"""
        print("\n" + "="*80)
        print("FORECASTING NEXT 7 DAYS")
        print("="*80)
        
        if 'update_predictor' not in self.models:
            print("Error: Update predictor not trained yet")
            return None
        
        model = self.models['update_predictor']
        
        # Get latest data for each state-district combination
        latest_data = self.data.sort_values('date').groupby(['state', 'district']).tail(1)
        
        forecasts = []
        
        for _, row in latest_data.iterrows():
            # Create features for next 7 days
            last_date = row['date']
            
            for days_ahead in range(1, 8):
                next_date = last_date + pd.Timedelta(days=days_ahead)
                
                features = {
                    'state_encoded': row['state_encoded'],
                    'district_encoded': row['district_encoded'],
                    'day_of_week': next_date.dayofweek,
                    'day_of_month': next_date.day,
                    'week_of_year': next_date.isocalendar().week,
                    'is_weekend': 1 if next_date.dayofweek in [5, 6] else 0,
                    'total_updates_lag1': row['total_updates'],
                    'total_updates_lag7': row.get('total_updates_lag7', 0),
                    'total_updates_rolling_7': row['total_updates_rolling_7'],
                    'total_bio_lag1': row['total_bio'],
                    'total_demo_lag1': row['total_demo'],
                    'bio_age_5_17': row['bio_age_5_17'],
                    'bio_age_17_': row['bio_age_17_'],
                    'demo_age_5_17': row['demo_age_5_17'],
                    'demo_age_17_': row['demo_age_17_']
                }
                
                X_pred = pd.DataFrame([features])
                predicted_updates = max(0, model.predict(X_pred)[0])
                
                forecasts.append({
                    'state': row['state'],
                    'district': row['district'],
                    'date': next_date,
                    'predicted_updates': predicted_updates
                })
        
        forecast_df = pd.DataFrame(forecasts)
        
        # Aggregate by date
        daily_forecast = forecast_df.groupby('date')['predicted_updates'].sum().reset_index()
        
        print("\nDaily Forecast Summary:")
        print(daily_forecast)
        
        # Top states forecast
        state_forecast = forecast_df.groupby('state')['predicted_updates'].sum().sort_values(ascending=False).head(10)
        
        print("\nTop 10 States - 7-Day Forecast:")
        print(state_forecast)
        
        return {
            'daily_forecast': daily_forecast,
            'state_forecast': state_forecast,
            'full_forecast': forecast_df
        }
    
    def identify_high_impact_factors(self):
        """Identify factors with highest impact on updates"""
        print("\n" + "="*80)
        print("HIGH IMPACT FACTOR ANALYSIS")
        print("="*80)
        
        insights = []
        
        if 'update_predictor' in self.feature_importance:
            print("\nFactors Driving Total Updates:")
            top_features = self.feature_importance['update_predictor'].head(5)
            print(top_features)
            
            for _, row in top_features.iterrows():
                insights.append({
                    'model': 'Update Prediction',
                    'factor': row['feature'],
                    'importance': row['importance']
                })
        
        if 'bio_preference' in self.feature_importance:
            print("\nFactors Driving Biometric Preference:")
            top_features = self.feature_importance['bio_preference'].head(5)
            print(top_features)
            
            for _, row in top_features.iterrows():
                insights.append({
                    'model': 'Bio Preference',
                    'factor': row['feature'],
                    'importance': row['importance']
                })
        
        return pd.DataFrame(insights)
    
    def save_models(self, path='models'):
        """Save trained models"""
        import os
        os.makedirs(path, exist_ok=True)
        
        for name, model in self.models.items():
            filepath = f"{path}/{name}.joblib"
            joblib.dump(model, filepath)
            print(f"Saved {name} to {filepath}")
        
        # Save encoders
        for name, encoder in self.encoders.items():
            filepath = f"{path}/{name}_encoder.joblib"
            joblib.dump(encoder, filepath)
            print(f"Saved {name} encoder to {filepath}")
    
    def run_predictive_analysis(self):
        """Run complete predictive analysis"""
        print("="*80)
        print("PREDICTIVE ANALYSIS")
        print("="*80)
        
        self.prepare_features()
        
        update_results = self.train_update_predictor()
        bio_results = self.train_bio_preference_predictor()
        forecast = self.forecast_next_week()
        impact_factors = self.identify_high_impact_factors()
        
        self.save_models()
        
        return {
            'update_model': update_results,
            'bio_model': bio_results,
            'forecast': forecast,
            'impact_factors': impact_factors
        }


if __name__ == "__main__":
    from aadhaar_analysis import AadhaarAnalyzer
    
    # Load data and run analysis
    analyzer = AadhaarAnalyzer()
    analyzer.run_full_analysis()
    
    # Run predictive modeling
    predictor = AadhaarPredictor(analyzer)
    results = predictor.run_predictive_analysis()
    
    print("\n" + "="*80)
    print("Predictive analysis complete!")
    print("="*80)
