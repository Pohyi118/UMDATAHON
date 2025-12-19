import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import timedelta

# --- CONFIGURATION ---
INPUT_FILE = 'AstraZeneca_Cleaned_Processed_Data.csv'
OUTPUT_FILE = 'Forecast_Output_For_Dashboard.csv'
FORECAST_WEEKS = 24  # 6 months

def run_quant_analysis():
    print(">>> READING DATA...")
    df = pd.read_csv(INPUT_FILE)
    
    # Ensure 'week' is datetime
    df['week'] = pd.to_datetime(df['week'])
    
    # ---------------------------------------------------------
    # PART 1: PREPARE TIME SERIES (INFLOW vs OUTFLOW)
    # ---------------------------------------------------------
    print(">>> PREPARING TIME SERIES...")
    
    # Group by week and direction to get total Inflow and Outflow per week
    # We use 'Amount in USD'. Outflows are usually negative, we make them positive for the model
    # then flip them back later.
    weekly_flow = df.groupby(['week', 'cash_flow_direction'])['Amount in USD'].sum().unstack(fill_value=0)
    
    # Ensure we have columns for Inflow and Outflow
    if 'Inflow' not in weekly_flow.columns: weekly_flow['Inflow'] = 0.0
    if 'Outflow' not in weekly_flow.columns: weekly_flow['Outflow'] = 0.0
    
    # Resample to ensure we have every week (fill missing weeks with 0)
    weekly_flow = weekly_flow.resample('W-MON').sum().fillna(0)
    
    # ---------------------------------------------------------
    # PART 2: RUN FORECASTING MODEL (Holt-Winters)
    # ---------------------------------------------------------
    print(">>> RUNNING FORECAST MODELS...")
    
    future_dates = [weekly_flow.index[-1] + timedelta(weeks=x) for x in range(1, FORECAST_WEEKS + 1)]
    
    # Model 1: Inflow
    # We use additive because cash flow doesn't strictly grow exponentially like a startup
    model_in = ExponentialSmoothing(weekly_flow['Inflow'], trend='add', seasonal=None).fit()
    forecast_in = model_in.forecast(FORECAST_WEEKS)
    
    # Model 2: Outflow (Model takes positive numbers better, so we flip absolute value if needed)
    # But here we just model the raw numbers. If Outflow is negative, the model predicts negative.
    model_out = ExponentialSmoothing(weekly_flow['Outflow'], trend='add', seasonal=None).fit()
    forecast_out = model_out.forecast(FORECAST_WEEKS)
    
    # Compile Forecast Data
    forecast_df = pd.DataFrame({
        'week': future_dates,
        'Inflow': forecast_in.values,
        'Outflow': forecast_out.values,
        'Type': 'Forecast'
    })
    
    # Compile Historical Data
    history_df = pd.DataFrame({
        'week': weekly_flow.index,
        'Inflow': weekly_flow['Inflow'].values,
        'Outflow': weekly_flow['Outflow'].values,
        'Type': 'History'
    })
    
    # Combine
    full_df = pd.concat([history_df, forecast_df])
    full_df['Net_Cash_Flow'] = full_df['Inflow'] + full_df['Outflow']
    
    # Calculate Running Balance (Assuming starting 0 for relative change)
    full_df['Cumulative_Balance_Change'] = full_df['Net_Cash_Flow'].cumsum()
    
    # ---------------------------------------------------------
    # PART 3: ANOMALY DETECTION (Z-Score)
    # ---------------------------------------------------------
    print(">>> DETECTING ANOMALIES...")
    
    # Calculate rolling statistics on HISTORY only
    hist_mask = full_df['Type'] == 'History'
    
    # 4-week Rolling Mean and Std Dev
    rolling_mean = full_df.loc[hist_mask, 'Net_Cash_Flow'].rolling(window=4).mean()
    rolling_std = full_df.loc[hist_mask, 'Net_Cash_Flow'].rolling(window=4).std()
    
    # Calculate Z-Score
    # (Current Value - Average) / Standard Deviation
    full_df.loc[hist_mask, 'Z_Score'] = (full_df.loc[hist_mask, 'Net_Cash_Flow'] - rolling_mean) / rolling_std
    
    # Flag Anomalies (Z-Score > 2 or < -2)
    full_df['Anomaly_Flag'] = np.where(abs(full_df['Z_Score']) > 2, True, False)
    
    # ---------------------------------------------------------
    # PART 4: SCENARIO PLANNING (The "What-If")
    # ---------------------------------------------------------
    print(">>> RUNNING 'OCT 27 PAYMENT DELAY' SCENARIO...")
    
    # 1. Identify the target range (Week of Oct 27, 2025)
    # Note: Adjust year if your data is 2024 or 2025. The code looks for month 10, day 27 roughly.
    target_date = pd.to_datetime("2025-10-27") 
    
    # Create a copy for the scenario
    df_scenario = df.copy()
    
    # Find the specific transactions: Category 'AP' or 'Netting' around that date
    # We look for large outflows in that specific week
    mask_payment = (
        (df_scenario['week'] == target_date) & 
        (df_scenario['cash_flow_direction'] == 'Outflow')
    )
    
    # Calculate the amount we are "moving"
    amount_to_move = df_scenario.loc[mask_payment, 'Amount in USD'].sum()
    print(f"   Identify Amount to move from Oct 27: ${amount_to_move:,.2f}")
    
    # Logic: Remove from Oct 27, Add to Nov 3
    # We create a 'Scenario_Net_Flow' column in our main full_df
    full_df['Scenario_Net_Flow'] = full_df['Net_Cash_Flow']
    
    # Adjust Oct 27 (Remove the outflow, which means SUBTRACTING a negative number -> Adding positive)
    # Wait! 'amount_to_move' is negative. To 'remove' it, we subtract it from the total? 
    # No, if Net Flow was -100, and we remove -30 payment, new Net Flow is -70. 
    # So we MINUS the (negative) amount_to_move.
    
    row_oct_27 = full_df['week'] == target_date
    full_df.loc[row_oct_27, 'Scenario_Net_Flow'] -= amount_to_move 
    
    # Adjust Nov 3 (Add the outflow back)
    next_week = target_date + timedelta(weeks=1)
    row_nov_03 = full_df['week'] == next_week
    full_df.loc[row_nov_03, 'Scenario_Net_Flow'] += amount_to_move
    
    # Calculate Scenario Cumulative Balance
    full_df['Scenario_Balance_Change'] = full_df['Scenario_Net_Flow'].cumsum()
    
    # Compare Minimums
    min_orig = full_df[full_df['Type'] == 'History']['Cumulative_Balance_Change'].min()
    min_scen = full_df[full_df['Type'] == 'History']['Scenario_Balance_Change'].min()
    
    print(f"   Original Low Point: ${min_orig:,.2f}")
    print(f"   Scenario Low Point: ${min_scen:,.2f}")
    print(f"   IMPROVEMENT: ${min_scen - min_orig:,.2f}")
    
    # ---------------------------------------------------------
    # PART 5: EXPORT
    # ---------------------------------------------------------
    full_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n>>> SUCCESS! Output saved to {OUTPUT_FILE}")
    print(">>> Hand this file to Person 3 (The Storyteller).")

if __name__ == "__main__":
    run_quant_analysis()


    