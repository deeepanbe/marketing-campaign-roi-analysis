#!/usr/bin/env python3
"""
Marketing Campaign ROI Analysis
Author: Deepanraj A - Data Analyst
GitHub: deeepanbe

Analyzes marketing campaign performance with attribution modeling and ROI optimization.
Includes multi-touch attribution, conversion tracking, and budget allocation recommendations.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def load_campaign_data(file_path='campaign_data.csv'):
    """
    Load marketing campaign data
    Expected columns: campaign_id, channel, spend, impressions, clicks, conversions, revenue
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Campaign data loaded: {df.shape[0]} records")
        return df
    except FileNotFoundError:
        print(f"File {file_path} not found. Generating sample data...")
        np.random.seed(42)
        n_campaigns = 500
        
        channels = ['Social Media', 'Email', 'Display', 'Search', 'Video']
        data = {
            'campaign_id': [f'CMP_{i:04d}' for i in range(n_campaigns)],
            'channel': np.random.choice(channels, n_campaigns),
            'spend': np.random.uniform(1000, 50000, n_campaigns),
            'impressions': np.random.randint(10000, 1000000, n_campaigns),
            'clicks': np.random.randint(100, 50000, n_campaigns),
            'conversions': np.random.randint(10, 2000, n_campaigns),
        }
        df = pd.DataFrame(data)
        df['revenue'] = df['conversions'] * np.random.uniform(50, 500, n_campaigns)
        return df

def calculate_campaign_metrics(df):
    """
    Calculate key marketing performance metrics
    """
    df = df.copy()
    
    # ROI metrics
    df['roi'] = ((df['revenue'] - df['spend']) / df['spend']) * 100
    df['roi_category'] = pd.cut(df['roi'], bins=[-np.inf, 0, 100, 200, np.inf],
                                labels=['Negative', 'Low', 'Medium', 'High'])
    
    # Efficiency metrics
    df['ctr'] = (df['clicks'] / df['impressions']) * 100
    df['conversion_rate'] = (df['conversions'] / df['clicks']) * 100
    df['cpc'] = df['spend'] / df['clicks']
    df['cpa'] = df['spend'] / df['conversions']
    df['roas'] = df['revenue'] / df['spend']
    
    # Performance indicators
    df['high_ctr'] = (df['ctr'] > df['ctr'].median()).astype(int)
    df['high_conversion'] = (df['conversion_rate'] > df['conversion_rate'].median()).astype(int)
    df['profitable'] = (df['roi'] > 0).astype(int)
    
    print(f"\nMetrics calculated for {len(df)} campaigns")
    return df

def channel_performance_analysis(df):
    """
    Analyze performance by marketing channel
    """
    print("\n=== Channel Performance Analysis ===")
    
    channel_stats = df.groupby('channel').agg({
        'spend': 'sum',
        'revenue': 'sum',
        'conversions': 'sum',
        'roi': 'mean',
        'ctr': 'mean',
        'conversion_rate': 'mean'
    }).round(2)
    
    channel_stats['total_roi'] = ((channel_stats['revenue'] - channel_stats['spend']) / 
                                  channel_stats['spend'] * 100).round(2)
    
    print(channel_stats)
    return channel_stats

def attribution_modeling(df):
    """
    Build predictive model for campaign success
    """
    print("\n=== Attribution Model Training ===")
    
    # Create binary target: high-performing campaigns
    df['high_performer'] = (df['roi'] > df['roi'].quantile(0.75)).astype(int)
    
    # Prepare features
    feature_cols = ['spend', 'impressions', 'clicks', 'ctr', 'conversion_rate', 
                   'cpc', 'high_ctr', 'high_conversion']
    
    # Add channel encoding
    channel_dummies = pd.get_dummies(df['channel'], prefix='channel')
    X = pd.concat([df[feature_cols], channel_dummies], axis=1)
    y = df['high_performer']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print("\n=== Model Performance ===")
    print(classification_report(y_test, y_pred, target_names=['Low Performer', 'High Performer']))
    print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n=== Top 5 Success Factors ===")
    print(feature_importance.head())
    
    return model, feature_importance

def optimize_budget_allocation(df, total_budget):
    """
    Recommend optimal budget allocation across channels
    """
    print(f"\n=== Budget Optimization (Total: ${total_budget:,.0f}) ===")
    
    # Calculate channel efficiency scores
    channel_efficiency = df.groupby('channel').agg({
        'roi': 'mean',
        'roas': 'mean',
        'conversion_rate': 'mean'
    })
    
    # Weighted scoring
    channel_efficiency['efficiency_score'] = (
        channel_efficiency['roi'] * 0.4 +
        channel_efficiency['roas'] * 100 * 0.4 +
        channel_efficiency['conversion_rate'] * 0.2
    )
    
    # Normalize scores to allocation percentages
    total_score = channel_efficiency['efficiency_score'].sum()
    channel_efficiency['recommended_allocation_pct'] = (
        channel_efficiency['efficiency_score'] / total_score * 100
    ).round(2)
    
    channel_efficiency['recommended_budget'] = (
        channel_efficiency['recommended_allocation_pct'] / 100 * total_budget
    ).round(2)
    
    print(channel_efficiency[['roi', 'roas', 'efficiency_score', 
                             'recommended_allocation_pct', 'recommended_budget']])
    
    return channel_efficiency

def generate_insights(df):
    """
    Generate actionable marketing insights
    """
    print("\n=== Key Insights ===")
    
    # Overall performance
    total_spend = df['spend'].sum()
    total_revenue = df['revenue'].sum()
    overall_roi = ((total_revenue - total_spend) / total_spend * 100)
    
    print(f"Total Spend: ${total_spend:,.2f}")
    print(f"Total Revenue: ${total_revenue:,.2f}")
    print(f"Overall ROI: {overall_roi:.2f}%")
    
    # Best performing campaigns
    top_campaigns = df.nlargest(5, 'roi')[['campaign_id', 'channel', 'spend', 'revenue', 'roi']]
    print("\nTop 5 Campaigns by ROI:")
    print(top_campaigns)
    
    # Worst performers
    bottom_campaigns = df[df['roi'] < 0].shape[0]
    print(f"\nUnprofitable Campaigns: {bottom_campaigns} ({bottom_campaigns/len(df)*100:.1f}%)")
    
    return {
        'total_spend': total_spend,
        'total_revenue': total_revenue,
        'overall_roi': overall_roi,
        'profitable_campaigns_pct': (df['profitable'].sum() / len(df) * 100)
    }

def main():
    """
    Main execution pipeline for campaign ROI analysis
    """
    print("Marketing Campaign ROI Analysis")
    print("=" * 60)
    
    # Load and process data
    df = load_campaign_data()
    df = calculate_campaign_metrics(df)
    
    # Analysis
    channel_stats = channel_performance_analysis(df)
    model, feature_importance = attribution_modeling(df)
    
    # Budget optimization
    total_budget = 500000  # $500K example budget
    optimization = optimize_budget_allocation(df, total_budget)
    
    # Insights
    insights = generate_insights(df)
    
    print("\n" + "=" * 60)
    print(f"Analysis complete! Overall ROI: {insights['overall_roi']:.1f}%")
    print(f"Profitable campaigns: {insights['profitable_campaigns_pct']:.1f}%")

if __name__ == "__main__":
    main()
