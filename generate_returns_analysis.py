#!/usr/bin/env python3
"""
Generate Returns Analysis - Standalone Script

Extracts mean daily returns for all trained symbols and generates returnsymbol.csv.
This script analyzes posterior trading results to determine the most profitable symbols.

Usage:
    python generate_returns_analysis.py

Output:
    returnsymbol.csv - Contains symbol and meanreturn columns
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib

# Set base path
basepath = '/Users/skumar81/Desktop/Personal/trading-final-stable'
sys.path.append(basepath)
os.chdir(basepath)

# Import parameters and utilities
from parameters import *
from common import generateposterior


def load_trained_data_and_models():
    """Load all trained data and models for return analysis"""
    print("Loading trained data and models...")
    
    # Initialize data structures
    rdflistp = {}
    qtnorm = {}
    lol = {}
    
    # Find all symbols with trained models
    trained_symbols = []
    models_dir = f"{basepath}/models/"
    
    if not os.path.exists(models_dir):
        print(f"Models directory not found: {models_dir}")
        return {}, {}, {}, []
    
    # Look for model files to determine trained symbols
    for filename in os.listdir(models_dir):
        if filename.endswith('localmodel.zip'):
            symbol = filename.replace('localmodel.zip', '')
            trained_symbols.append(symbol)
    
    print(f"Found trained models for {len(trained_symbols)} symbols: {trained_symbols}")
    
    # Load data and normalizers for each symbol
    for symbol in trained_symbols:
        print(f"Loading data for {symbol}...")
        
        # Load historical data
        for prefix in ["final"]:
            data_file = f"{basepath}/traindata/{prefix}mldf{symbol}.csv"
            if os.path.exists(data_file):
                try:
                    rdf = pd.read_csv(data_file)
                    # Convert currentt to datetime if it's a string
                    if 'currentt' in rdf.columns:
                        rdf['currentt'] = pd.to_datetime(rdf['currentt'])
                    rdflistp[symbol + prefix] = rdf
                    print(f"  Loaded {len(rdf)} rows for {symbol}{prefix}")
                except Exception as e:
                    print(f"  Error loading data for {symbol}: {e}")
                    continue
            else:
                print(f"  Data file not found: {data_file}")
                continue
        
        # Load quantile normalizer
        normalizer_file = f"{basepath}/models/{symbol}qt.joblib"
        if os.path.exists(normalizer_file):
            try:
                qtnorm[symbol] = joblib.load(normalizer_file)
                print(f"  Loaded normalizer for {symbol}")
            except Exception as e:
                print(f"  Error loading normalizer for {symbol}: {e}")
                continue
        else:
            print(f"  Normalizer not found: {normalizer_file}")
            continue
        
        # Load signals list
        signals_file = f"{basepath}/models/{symbol}signals.joblib"
        if os.path.exists(signals_file):
            try:
                lol[symbol] = joblib.load(signals_file)
                print(f"  Loaded {len(lol[symbol])} signals for {symbol}")
            except Exception as e:
                print(f"  Error loading signals for {symbol}: {e}")
                # Use default signal columns if signals file not found
                if symbol + "final" in rdflistp:
                    signal_cols = [col for col in rdflistp[symbol + "final"].columns 
                                 if col not in ['currentt', 'currento', 't', 'vwap2']]
                    lol[symbol] = signal_cols
                    print(f"  Using default signals: {len(signal_cols)} columns")
        else:
            print(f"  Signals file not found, using default columns")
            if symbol + "final" in rdflistp:
                signal_cols = [col for col in rdflistp[symbol + "final"].columns 
                             if col not in ['currentt', 'currento', 't', 'vwap2']]
                lol[symbol] = signal_cols
                print(f"  Using default signals: {len(signal_cols)} columns")
    
    # Filter to symbols that have all required data
    valid_symbols = []
    for symbol in trained_symbols:
        if (symbol + "final" in rdflistp and 
            symbol in qtnorm and 
            symbol in lol):
            valid_symbols.append(symbol)
        else:
            print(f"  Skipping {symbol} - missing required data")
    
    print(f"Ready to analyze {len(valid_symbols)} symbols: {valid_symbols}")
    return rdflistp, qtnorm, lol, valid_symbols


def generate_returns_analysis(rdflistp, qtnorm, lol, symbols):
    """Generate return analysis for all symbols - LAST 10 AND LAST 5 TRADING DAYS"""
    print("\nGenerating posterior analysis for return calculation (LAST 10 AND 5 TRADING DAYS)...")
    
    results = []
    
    for symbol in symbols:
        print(f"\nAnalyzing returns for {symbol}...")
        
        try:
            # Get the full data for this symbol
            full_data_key = f"{symbol}final"
            if full_data_key not in rdflistp:
                print(f"  No data found for {symbol}")
                results.append({
                    'symbol': symbol,
                    'meanreturn_10d': 0.0,
                    'meanreturn_5d': 0.0,
                    'mean_pnl_absolute': 0.0,
                    'status': 'no_data'
                })
                continue
                
            full_df = rdflistp[full_data_key]
            
            # Ensure currentdate column exists
            if 'currentdate' not in full_df.columns:
                if 'currentt' in full_df.columns:
                    full_df['currentt'] = pd.to_datetime(full_df['currentt'])
                    full_df['currentdate'] = full_df['currentt'].dt.date
                else:
                    print(f"  No date column found for {symbol}")
                    results.append({
                        'symbol': symbol,
                        'meanreturn_10d': 0.0,
                        'meanreturn_5d': 0.0,
                        'mean_pnl_absolute': 0.0,
                        'status': 'no_date'
                    })
                    continue
            
            # Get unique dates and split into train/test
            alldates = full_df['currentdate'].unique()
            traindates = alldates[:int(np.ceil(len(alldates)*TRAIN_MAX))]
            testdates = alldates[int(np.ceil(len(alldates)*TRAIN_MAX)):]
            
            # Calculate for both 10 days and 5 days
            returns_data = {}
            
            for num_days in [10, 5]:
                print(f"  Calculating {num_days}-day returns...")
                
                # Get the last N trading days of test data
                if len(testdates) > num_days:
                    last_n_dates = testdates[-num_days:]
                    print(f"    Using last {num_days} trading days: {last_n_dates[0]} to {last_n_dates[-1]}")
                else:
                    last_n_dates = testdates
                    print(f"    Using all {len(testdates)} test days: {last_n_dates[0]} to {last_n_dates[-1]}")
                
                # Filter the dataframe to only include training data + last N test days
                filtered_df = full_df[
                    (full_df['currentdate'].isin(traindates)) | 
                    (full_df['currentdate'].isin(last_n_dates))
                ].reset_index(drop=True)
                
                # Update the rdflistp with filtered data
                symbol_rdflistp = {full_data_key: filtered_df}
                symbol_qtnorm = {symbol: qtnorm[symbol]}
                symbol_lol = {symbol: lol[symbol]}
                
                # Generate trading actions using generateposterior
                print(f"    Generating trading actions...")
                df_test_actions_list = generateposterior(
                    symbol_rdflistp, symbol_qtnorm, [symbol], symbol_lol
                )
                
                # Calculate PnL using posteriorplots
                print(f"    Calculating PnL...")
                from common import posteriorplots
                
                symposterior = posteriorplots(df_test_actions_list, [symbol], 0)
                
                # Extract the mean PnL from symposterior
                mean_percentage_pnl = 0.0
                key = f"{symbol}final"
                
                if key in symposterior and symposterior[key] and len(symposterior[key]) > 0:
                    pnls = symposterior[key]
                    mean_pnl = np.mean(pnls)
                    mean_percentage_pnl = mean_pnl / INITIAL_ACCOUNT_BALANCE * 100
                    print(f"    {symbol}: {mean_percentage_pnl:.4f}% daily return (last {num_days} days)")
                else:
                    print(f"    No PnL data found for {symbol}")
                    mean_percentage_pnl = 0.0
                    mean_pnl = 0.0
                
                # Store results for this period
                returns_data[f'{num_days}d'] = round(mean_percentage_pnl, 4)
                if num_days == 10:
                    returns_data['mean_pnl_absolute'] = mean_pnl
            
            results.append({
                'symbol': symbol,
                'meanreturn_10d': returns_data['10d'],
                'meanreturn_5d': returns_data['5d'],
                'mean_pnl_absolute': returns_data.get('mean_pnl_absolute', 0.0),
                'status': 'success' if returns_data['10d'] != 0 or returns_data['5d'] != 0 else 'no_trades'
            })
                
        except Exception as e:
            print(f"  Error analyzing {symbol}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'symbol': symbol,
                'meanreturn_10d': 0.0,
                'meanreturn_5d': 0.0,
                'mean_pnl_absolute': 0.0,
                'status': 'error'
            })
    
    return results


def save_returns_to_csv(results, filename='returnsymbol.csv'):
    """Save return analysis results to CSV file"""
    print(f"\nSaving results to {filename}...")
    
    # Create DataFrame from results
    df = pd.DataFrame(results)
    
    # Sort by 10-day return descending (best performers first)
    df = df.sort_values('meanreturn_10d', ascending=False)
    
    # Save to CSV with symbol, meanreturn (10d), and meanreturn_5d columns
    output_df = df[['symbol', 'meanreturn_10d', 'meanreturn_5d']].copy()
    output_df.columns = ['symbol', 'meanreturn', 'meanreturn_5d']  # Rename for backward compatibility
    output_df.to_csv(filename, index=False)
    
    print(f"Saved {len(output_df)} symbols to {filename}")
    
    # Print summary
    print("\nReturn Analysis Summary:")
    print("=" * 70)
    print(f"{'Symbol':<12} {'10-Day Return %':<18} {'5-Day Return %':<18} {'Status':<10}")
    print("-" * 70)
    
    for _, row in df.iterrows():
        print(f"{row['symbol']:<12} {row['meanreturn_10d']:<18.4f} {row['meanreturn_5d']:<18.4f} {row['status']:<10}")
    
    print("-" * 70)
    
    successful_symbols = df[df['status'] == 'success']
    if len(successful_symbols) > 0:
        avg_return_10d = successful_symbols['meanreturn_10d'].mean()
        avg_return_5d = successful_symbols['meanreturn_5d'].mean()
        best_symbol_10d = successful_symbols.iloc[0]
        worst_symbol_10d = successful_symbols.iloc[-1]
        
        # Also find best/worst for 5-day returns
        df_5d_sorted = successful_symbols.sort_values('meanreturn_5d', ascending=False)
        best_symbol_5d = df_5d_sorted.iloc[0] if len(df_5d_sorted) > 0 else None
        worst_symbol_5d = df_5d_sorted.iloc[-1] if len(df_5d_sorted) > 0 else None
        
        print(f"\n10-Day Statistics:")
        print(f"  Average return: {avg_return_10d:.4f}%")
        print(f"  Best performer: {best_symbol_10d['symbol']} ({best_symbol_10d['meanreturn_10d']:.4f}%)")
        print(f"  Worst performer: {worst_symbol_10d['symbol']} ({worst_symbol_10d['meanreturn_10d']:.4f}%)")
        
        print(f"\n5-Day Statistics:")
        print(f"  Average return: {avg_return_5d:.4f}%")
        if best_symbol_5d is not None:
            print(f"  Best performer: {best_symbol_5d['symbol']} ({best_symbol_5d['meanreturn_5d']:.4f}%)")
            print(f"  Worst performer: {worst_symbol_5d['symbol']} ({worst_symbol_5d['meanreturn_5d']:.4f}%)")
    
    print(f"\nResults saved to: {os.path.abspath(filename)}")
    return output_df


def main():
    """Main function to run the return analysis"""
    print("=" * 60)
    print("TRADING RETURNS ANALYSIS")
    print("=" * 60)
    print("Analyzing mean daily returns for all trained symbols...")
    
    try:
        # Load data and models
        rdflistp, qtnorm, lol, symbols = load_trained_data_and_models()
        
        if not symbols:
            print("No trained symbols found. Please run the training pipeline first.")
            return
        
        # Generate returns analysis
        results = generate_returns_analysis(rdflistp, qtnorm, lol, symbols)
        
        # Save to CSV
        output_df = save_returns_to_csv(results)
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Use returnsymbol.csv to select the most profitable symbols for trading.")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()