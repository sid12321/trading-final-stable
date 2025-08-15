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
    """Generate return analysis for all symbols - LAST 10 TRADING DAYS ONLY"""
    print("\nGenerating posterior analysis for return calculation (LAST 10 TRADING DAYS)...")
    
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
                    'meanreturn': 0.0,
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
                        'meanreturn': 0.0,
                        'mean_pnl_absolute': 0.0,
                        'status': 'no_date'
                    })
                    continue
            
            # Get unique dates and split into train/test
            alldates = full_df['currentdate'].unique()
            traindates = alldates[:int(np.ceil(len(alldates)*TRAIN_MAX))]
            testdates = alldates[int(np.ceil(len(alldates)*TRAIN_MAX)):]
            
            # Get only the LAST 10 trading days of test data
            if len(testdates) > 10:
                last_10_dates = testdates[-10:]
                print(f"  Using last 10 trading days: {last_10_dates[0]} to {last_10_dates[-1]}")
            else:
                last_10_dates = testdates
                print(f"  Using all {len(testdates)} test days: {last_10_dates[0]} to {last_10_dates[-1]}")
            
            # Filter the dataframe to only include training data + last 10 test days
            # We need training data for the model to have context
            filtered_df = full_df[
                (full_df['currentdate'].isin(traindates)) | 
                (full_df['currentdate'].isin(last_10_dates))
            ].reset_index(drop=True)
            
            # Update the rdflistp with filtered data
            symbol_rdflistp = {full_data_key: filtered_df}
            symbol_qtnorm = {symbol: qtnorm[symbol]}
            symbol_lol = {symbol: lol[symbol]}
            
            # Step 1: Generate trading actions using generateposterior
            print(f"  Generating trading actions for {symbol} (last 10 days)...")
            df_test_actions_list = generateposterior(
                symbol_rdflistp, symbol_qtnorm, [symbol], symbol_lol
            )
            
            # Step 2: Calculate PnL using posteriorplots (same logic as model_trainer)
            print(f"  Calculating PnL for {symbol} (last 10 days)...")
            from common import posteriorplots
            
            # posteriorplots expects data for all iterations, so we'll run it with index 0
            symposterior = posteriorplots(df_test_actions_list, [symbol], 0)
            
            # Extract the mean PnL from symposterior (same logic as model_trainer)
            mean_percentage_pnl = 0.0
            key = f"{symbol}final"
            
            if key in symposterior and symposterior[key] and len(symposterior[key]) > 0:
                pnls = symposterior[key]
                mean_pnl = np.mean(pnls)
                mean_percentage_pnl = mean_pnl / INITIAL_ACCOUNT_BALANCE * 100
                print(f"  {symbol}: {mean_percentage_pnl:.4f}% daily return (last 10 days)")
            else:
                print(f"  No PnL data found for {symbol}")
                mean_percentage_pnl = 0.0
            
            results.append({
                'symbol': symbol,
                'meanreturn': round(mean_percentage_pnl, 4),
                'mean_pnl_absolute': mean_pnl if key in symposterior else 0.0,
                'status': 'success' if mean_percentage_pnl != 0 else 'no_trades'
            })
                
        except Exception as e:
            print(f"  Error analyzing {symbol}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'symbol': symbol,
                'meanreturn': 0.0,
                'mean_pnl_absolute': 0.0,
                'status': 'error'
            })
    
    return results


def save_returns_to_csv(results, filename='returnsymbol.csv'):
    """Save return analysis results to CSV file"""
    print(f"\nSaving results to {filename}...")
    
    # Create DataFrame from results
    df = pd.DataFrame(results)
    
    # Sort by meanreturn descending (best performers first)
    df = df.sort_values('meanreturn', ascending=False)
    
    # Save to CSV with just symbol and meanreturn columns (as requested)
    output_df = df[['symbol', 'meanreturn']].copy()
    output_df.to_csv(filename, index=False)
    
    print(f"Saved {len(output_df)} symbols to {filename}")
    
    # Print summary
    print("\nReturn Analysis Summary:")
    print("=" * 50)
    print(f"{'Symbol':<12} {'Mean Return %':<15} {'Status':<10}")
    print("-" * 50)
    
    for _, row in df.iterrows():
        print(f"{row['symbol']:<12} {row['meanreturn']:<15.4f} {row['status']:<10}")
    
    print("-" * 50)
    
    successful_symbols = df[df['status'] == 'success']
    if len(successful_symbols) > 0:
        avg_return = successful_symbols['meanreturn'].mean()
        best_symbol = successful_symbols.iloc[0]
        worst_symbol = successful_symbols.iloc[-1]
        
        print(f"Average return: {avg_return:.4f}%")
        print(f"Best performer: {best_symbol['symbol']} ({best_symbol['meanreturn']:.4f}%)")
        print(f"Worst performer: {worst_symbol['symbol']} ({worst_symbol['meanreturn']:.4f}%)")
    
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