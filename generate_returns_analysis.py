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
    """Generate return analysis for all symbols"""
    print("\nGenerating posterior analysis for return calculation...")
    
    results = []
    
    for symbol in symbols:
        print(f"\nAnalyzing returns for {symbol}...")
        
        try:
            # Run posterior analysis for this symbol only
            symbol_rdflistp = {k: v for k, v in rdflistp.items() if k.startswith(symbol)}
            symbol_qtnorm = {symbol: qtnorm[symbol]}
            symbol_lol = {symbol: lol[symbol]}
            
            # Capture the output from generateposterior
            # The function prints "Mean daily PnL: {mean_pnl} or {mean_percentage_pnl}% for {SYM}{prefix}"
            # We need to run it and capture the mean_pnl value
            
            from io import StringIO
            import contextlib
            
            # Redirect stdout to capture print statements
            captured_output = StringIO()
            
            with contextlib.redirect_stdout(captured_output):
                df_test_actions_list = generateposterior(
                    symbol_rdflistp, symbol_qtnorm, [symbol], symbol_lol
                )
            
            # Parse the captured output to extract mean PnL
            output_lines = captured_output.getvalue().split('\n')
            mean_pnl = None
            mean_percentage_pnl = None
            
            for line in output_lines:
                if f"Mean daily PnL:" in line and f"for {symbol}final" in line:
                    # Extract the numbers from the line
                    # Format: "Mean daily PnL: 594.4472222221941 or 0.59% for ITCfinal"
                    parts = line.split("Mean daily PnL: ")[1]
                    pnl_part = parts.split(" or ")[0]
                    percentage_part = parts.split(" or ")[1].split("%")[0]
                    
                    mean_pnl = float(pnl_part)
                    mean_percentage_pnl = float(percentage_part)
                    break
            
            if mean_pnl is not None:
                results.append({
                    'symbol': symbol,
                    'meanreturn': mean_percentage_pnl,  # Use percentage return
                    'mean_pnl_absolute': mean_pnl,      # Also include absolute PnL
                    'status': 'success'
                })
                print(f"  {symbol}: {mean_percentage_pnl}% daily return")
            else:
                print(f"  Could not extract return for {symbol}")
                results.append({
                    'symbol': symbol,
                    'meanreturn': 0.0,
                    'mean_pnl_absolute': 0.0,
                    'status': 'failed'
                })
                
        except Exception as e:
            print(f"  Error analyzing {symbol}: {e}")
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