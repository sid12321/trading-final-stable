#!/usr/bin/env python3
"""
Reformat reward_components_analysis.csv to have actual and scaled columns paired together
Makes it easier to compare actual vs scaled values side by side
"""

import pandas as pd

def reformat_csv():
    """Reformat the CSV with paired actual/scaled columns"""
    
    # Read the original CSV
    df = pd.read_csv("reward_components_analysis.csv")
    
    # Define the reward components
    components = ['profit', 'step', 'action', 'position', 'risk', 'activity', 'sharpe']
    
    # Create new column order
    new_columns = ['sample_id']
    
    # Add paired actual/scaled columns
    for comp in components:
        new_columns.extend([f'actual_{comp}', f'scaled_{comp}'])
    
    # Add remaining columns
    remaining_cols = [col for col in df.columns if col not in new_columns]
    new_columns.extend(remaining_cols)
    
    # Reorder the DataFrame
    df_reformatted = df[new_columns]
    
    # Save the reformatted CSV
    output_file = "reward_components_analysis_paired.csv"
    df_reformatted.to_csv(output_file, index=False)
    
    print(f"✅ Reformatted CSV saved to: {output_file}")
    print(f"📊 {len(df)} samples with paired actual/scaled columns")
    
    # Show first few rows to verify the format
    print("\nFirst 3 rows of reformatted data:")
    print("="*80)
    
    # Display with better formatting
    for i in range(min(3, len(df_reformatted))):
        print(f"\nSample {df_reformatted.iloc[i]['sample_id']}:")
        for comp in components:
            actual_val = df_reformatted.iloc[i][f'actual_{comp}']
            scaled_val = df_reformatted.iloc[i][f'scaled_{comp}']
            print(f"  {comp:8s}: {actual_val:12.6f} → {scaled_val:12.6f}")

if __name__ == "__main__":
    reformat_csv()