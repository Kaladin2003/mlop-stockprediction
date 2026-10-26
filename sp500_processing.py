import pandas as pd
import os
import time

from features import add_features, FEATURES, SMA_50_PERIOD

def process_single_ticker(filepath):
    ticker_name = os.path.basename(filepath).replace("_10y.csv", "")
    
    try:
        df = pd.read_csv(filepath)
        if 'Date' not in df.columns:
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'Date'}, inplace=True)
            
        df['Date'] = pd.to_datetime(df['Date'])
        # Χρονική ταξινόμηση για τους κινητούς μέσους και το shift
        df = df.sort_values('Date').reset_index(drop=True)
       
        if len(df) < SMA_50_PERIOD + 1:
            return None
            
        df = add_features(df)
        
        # Στόχος: 1 αν η αυριανή τιμή είναι μεγαλύτερη από τη σημερινή
        next_close = df['Close'].shift(-1)
        df['Target_Direction'] = (next_close > df['Close']).astype(int)
        
        # Η τελευταία γραμμή δεν έχει "αύριο"
        df = df[next_close.notna()]
        
        df = df.dropna(subset=FEATURES)
        
        needed_columns = ['Date', 'Close', 'SMA_10', 'SMA_50'] + FEATURES + ['Target_Direction']
        df = df[needed_columns].copy()
        df['Ticker_Name'] = ticker_name

        return df
    except Exception as e:
        print(f"Error processing {ticker_name}: {e}")
        return None

def merge_sp500_data(raw_folder="data_raw_sp500", output_file="data_processed_sp500/sp500_combined_10y.csv"):
    print("Processing and merging S&P 500 data. This will create a MASSIVE dataset...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    all_processed_dfs = []
    
    csv_files = [f for f in os.listdir(raw_folder) if f.endswith(".csv")]
    total_files = len(csv_files)
    
    for i, filename in enumerate(csv_files):
        print(f"[{i+1}/{total_files}] Processing: {filename}")
        filepath = os.path.join(raw_folder, filename)
        processed_df = process_single_ticker(filepath)
        
        if processed_df is not None:
            all_processed_dfs.append(processed_df)

    print("Merging all tickers into one giant DataFrame...")
    # Cross-sectional ένωση όλων των μετοχών
    combined_df = pd.concat(all_processed_dfs, ignore_index=True)
    
    print(f"Final Combined Dataset has {len(combined_df)} rows and {len(combined_df.columns)} columns.")
    print(f"Saving to {output_file}...")
    combined_df.to_csv(output_file, index=False)
    print("Merge Successful.")
    print("\nΔείγμα του τελικού συνόλου δεδομένων:")
    print(combined_df[['Ticker_Name', 'Date', 'Close', 'SMA_10', 'SMA_50', 'Target_Direction']].tail(5))

    return {"rows": len(combined_df), "tickers": combined_df['Ticker_Name'].nunique(),
            "last_date": str(combined_df['Date'].max().date())}

if __name__ == "__main__":
    start_time = time.time()
    merge_sp500_data()
    print(f"Συνολικός χρόνος επεξεργασίας: {round((time.time() - start_time)/60, 2)} λεπτά.")
