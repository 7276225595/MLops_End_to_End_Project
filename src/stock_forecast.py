"""
Stock Price Prediction Model for RELIANCE.NS
Predicts next day's Open, High, Low, Close using Random Forest
Enhanced for MLOps with S3 integration
"""

import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import os
import sys
import json
import boto3
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error

def fetch_and_prepare_data(ticker="RELIANCE.NS", period="1y"):
    """Fetch stock data and create features"""
    print(f"📊 Fetching data for {ticker}...")
    df = yf.Ticker(ticker).history(period=period)
    
    if df.empty:
        raise ValueError(f"No data found for {ticker}")
    
    print(f"✅ Data fetched: {len(df)} rows from {df.index[0].date()} to {df.index[-1].date()}")
    
    # Feature engineering
    df['HL_diff'] = df['High'] - df['Low']
    df['CO_diff'] = df['Close'] - df['Open']
    df['MA_5'] = df['Close'].rolling(5).mean()
    df['MA_10'] = df['Close'].rolling(10).mean()
    
    # Targets for tomorrow
    df['O_next'] = df['Open'].shift(-1)
    df['H_next'] = df['High'].shift(-1)
    df['L_next'] = df['Low'].shift(-1)
    df['C_next'] = df['Close'].shift(-1)
    
    df.dropna(inplace=True)
    print(f"✅ After preprocessing: {len(df)} rows")
    
    return df

def train_model(df):
    """Train multi-output Random Forest model"""
    features = ['Open', 'High', 'Low', 'Close', 'HL_diff', 'CO_diff', 'MA_5', 'MA_10', 'Volume']
    X = df[features]
    y = df[['O_next', 'H_next', 'L_next', 'C_next']]
    
    # Train-test split (no shuffle for time series)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"📊 Training set: {len(X_train)} rows")
    print(f"📊 Test set: {len(X_test)} rows")
    
    # Train model
    print("🤖 Training Random Forest model...")
    base_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print("\n📈 Model Performance (MAE):")
    metrics = {}
    for col, true, pred in zip(y.columns, y_test.values.T, y_pred.T):
        mae = mean_absolute_error(true, pred)
        metrics[col] = mae
        print(f"  {col}: ₹{mae:.2f}")
    
    # Calculate average MAE
    avg_mae = sum(metrics.values()) / len(metrics)
    print(f"\n📊 Average MAE: ₹{avg_mae:.2f}")
    
    return model, X, y, features, metrics, avg_mae

def generate_forecast(model, df, features):
    """Predict tomorrow's prices"""
    X_latest = df.iloc[[-1]][features]
    o, h, l, c = model.predict(X_latest)[0]
    current_close = df['Close'].iloc[-1]
    
    print("\n" + "="*50)
    print("📅 TOMORROW'S FORECAST")
    print("="*50)
    print(f"Open:  ₹{o:.2f}")
    print(f"High:  ₹{h:.2f}")
    print(f"Low:   ₹{l:.2f}")
    print(f"Close: ₹{c:.2f}")
    print(f"Expected Range (H-L): ₹{(h - l):.2f}")
    print(f"Direction: {'📈 UP' if c > current_close else '📉 DOWN'}")
    print(f"Current Close: ₹{current_close:.2f}")
    print("="*50)
    
    # Save forecast to file
    with open('forecast.txt', 'w') as f:
        f.write(f"Stock: RELIANCE.NS\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Open: ₹{o:.2f}\n")
        f.write(f"High: ₹{h:.2f}\n")
        f.write(f"Low: ₹{l:.2f}\n")
        f.write(f"Close: ₹{c:.2f}\n")
        f.write(f"Direction: {'UP' if c > current_close else 'DOWN'}\n")
        f.write(f"Current Close: ₹{current_close:.2f}\n")
    
    return o, h, l, c

def upload_to_s3(local_path, s3_path, bucket, s3_client):
    """Upload file to S3 bucket"""
    try:
        s3_client.upload_file(local_path, bucket, s3_path)
        print(f"✅ Uploaded to s3://{bucket}/{s3_path}")
        return True
    except Exception as e:
        print(f"⚠️ S3 upload failed: {e}")
        return False

def main():
    """Main pipeline"""
    print("="*50)
    print("🏦 STOCK FORECAST MLOPS PIPELINE")
    print("="*50)
    print(f"Started at: {datetime.now().isoformat()}")
    
    # Fetch data
    df = fetch_and_prepare_data()
    
    # Train model
    model, X, y, features, metrics, avg_mae = train_model(df)
    
    # Check model quality
    if avg_mae > 30:
        print(f"\n⚠️ WARNING: High MAE (₹{avg_mae:.2f}) - Model may be unreliable")
    
    # Generate forecast
    forecast = generate_forecast(model, df, features)
    
    # Save model locally
    joblib.dump(model, 'model.pkl')
    print("\n✅ Model saved as model.pkl")
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv('metrics.csv', index=False)
    print("✅ Metrics saved as metrics.csv")
    
    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'model_type': 'RandomForestRegressor',
        'n_estimators': 100,
        'features': features,
        'metrics': metrics,
        'avg_mae': avg_mae,
        'data_shape': len(df),
        'forecast': {
            'open': forecast[0],
            'high': forecast[1],
            'low': forecast[2],
            'close': forecast[3]
        }
    }
    
    with open('metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("✅ Metadata saved as metadata.json")
    
    print("\n" + "="*50)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*50)
    
    return model, metrics, forecast

if __name__ == "__main__":
    try:
        model, metrics, forecast = main()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
