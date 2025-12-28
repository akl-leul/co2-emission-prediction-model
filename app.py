import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.integrate import quad
from sklearn.linear_model import LinearRegression

# --------------------------------------------------
# EMISSIONS MODEL WITH AUTO-CALCULATED EMISSION_FACTOR
# --------------------------------------------------
BAG_WEIGHT_GRAMS = 350

def prepare_data(df, emission_factor):
    """Enhanced with dynamic emission factor"""
    for col in ["Avg_Charcoal_kg", "Households"]:
        if col not in df.columns:
            raise ValueError(f"Required column missing: {col}")
    df["Total_Charcoal_kg"] = df["Avg_Charcoal_kg"] * df["Households"]
    df["CO2_kg"] = df["Total_Charcoal_kg"] * emission_factor
    return df

def fit_consumption_model(df):
    X = df["Week"].values.reshape(-1, 1)
    y = df["Total_Charcoal_kg"].values
    model = LinearRegression()
    model.fit(X, y)
    return model.coef_[0], model.intercept_, model

def emission_rate(t, a, b, emission_factor):
    return emission_factor * (a * t + b)

def total_emissions(T, a, b, emission_factor):
    result, _ = quad(emission_rate, 0, T, args=(a, b, emission_factor))
    return result

def convert_to_json_serializable(obj):
    """Convert pandas Timestamp and other non-JSON types to strings"""
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if pd.isna(obj):
        return None
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# 🧠 ENHANCED SELF-LEARNING SYSTEM - SAVES FULL TABLES
LEARNING_FILE = "ai_learning_history.json"
FULL_DATA_FILE = "ai_full_data_history.json"

class LearningAI:
    def __init__(self):
        self.history = []
        self.full_data_history = []
        self.load_history()
        self.load_full_data()

    def load_history(self):
        if os.path.exists(LEARNING_FILE):
            try:
                with open(LEARNING_FILE, "r") as f:
                    self.history = json.load(f)
            except:
                self.history = []

    def load_full_data(self):
        if os.path.exists(FULL_DATA_FILE):
            try:
                with open(FULL_DATA_FILE, "r") as f:
                    self.full_data_history = json.load(f)
            except:
                self.full_data_history = []

    def calculate_optimal_emission_factor(self):
        """
        Calculate EMISSION_FACTOR based on Ground Truth data if available.
        Fallback to historical average or default.
        """
        # 1. Try to find Ground Truth data
        ground_truth_factors = []
        
        for data_entry in self.full_data_history:
             # Check if this dataset was marked as Ground Truth
            if data_entry.get('is_ground_truth', False) and 'processed_data_table' in data_entry:
                for row in data_entry['processed_data_table']:
                    charcoal = row.get('Total_Charcoal_kg', 0)
                    co2 = row.get('CO2_kg', 0)
                    if charcoal > 0 and co2 > 0:
                        ground_truth_factors.append(co2 / charcoal)

        if ground_truth_factors:
            # If we have real data, use it!
            return float(np.mean(ground_truth_factors))

        # 2. Fallback: Use all history (circular but better than nothing if no GT)
        # Or better: default to variable based on theoretical charcoal carbon content (~80%)
        # C + O2 -> CO2. 12g C -> 44g CO2. Ratio = 3.66.
        # Charcoal is ~75-85% Carbon. 0.8 * 3.66 = ~2.93.
        # Let's check history just in case the user manually adjusted it in previous versions (if we supported that)
        
        if not self.full_data_history:
            return 2.93  # Theoretical average for good charcoal

        all_charcoal = []
        all_co2 = []
        
        for data_entry in self.full_data_history:
            if 'processed_data_table' in data_entry:
                for row in data_entry['processed_data_table']:
                    if 'Total_Charcoal_kg' in row and 'CO2_kg' in row:
                        charcoal = row['Total_Charcoal_kg']
                        co2 = row['CO2_kg']
                        if charcoal > 0:
                            all_charcoal.append(charcoal)
                            all_co2.append(co2)
        
        if len(all_charcoal) > 0:
            optimal_factor = np.mean(all_co2) / np.mean(all_charcoal)
            return max(1.0, min(4.0, optimal_factor))
            
        return 2.93

    def save_full_dataset(self, raw_df, processed_df, a, b, r2, predictions, emission_factor, is_ground_truth=False):
        """🆕 Save ALL table rows + complete data - JSON SAFE"""
        
        # Convert dataframes to JSON-serializable format
        raw_serializable = raw_df.reset_index(drop=True).applymap(convert_to_json_serializable).to_dict('records')
        processed_serializable = processed_df.reset_index(drop=True).applymap(convert_to_json_serializable).to_dict('records')
        
        # Summary entry
        summary_entry = {
            "timestamp": datetime.now().isoformat(),
            "n_samples": len(processed_df),
            "n_raw_rows": len(raw_df),
            "slope_a": float(a),
            "intercept_b": float(b),
            "r2_score": float(r2),
            "auto_emission_factor": float(emission_factor),
            "emissions_1yr": float(total_emissions(52, a, b, emission_factor)),
            "emissions_2yr": float(total_emissions(104, a, b, emission_factor)),
        }
        self.history.append(summary_entry)
        
        # 🆕 FULL DATA: ALL RAW ROWS + ALL PROCESSED ROWS + PREDICTIONS
        full_entry = {
            "timestamp": datetime.now().isoformat(),
            "source_file": "Excel Upload",
            "n_raw_rows": len(raw_df),
            "n_processed_rows": len(processed_df),
            "model_params": {
                "slope_a": float(a),
                "intercept_b": float(b),
                "r2_score": float(r2),
                "emission_factor": float(emission_factor)
            },
            "is_ground_truth": is_ground_truth,
            "raw_data_table": raw_serializable,  # ALL RAW ROWS - JSON SAFE
            "processed_data_table": processed_serializable,  # ALL PROCESSED - JSON SAFE
            "predictions_list": predictions.tolist(),  # ALL PREDICTIONS
            "forecasts": {
                "1yr": float(total_emissions(52, a, b, emission_factor)),
                "2yr": float(total_emissions(104, a, b, emission_factor)),
                "3yr": float(total_emissions(156, a, b, emission_factor)),
                "5yr": float(total_emissions(260, a, b, emission_factor))
            }
        }
        self.full_data_history.append(full_entry)
        
        # Save files - KEEP LAST 20 FULL DATASETS
        with open(LEARNING_FILE, "w") as f:
            json.dump(self.history[-100:], f, indent=2)
        
        with open(FULL_DATA_FILE, "w") as f:
            json.dump(self.full_data_history[-20:], f, indent=2)

ai_brain = LearningAI()

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(page_title="Charcoal CO₂ Emissions Predictor", layout="wide")

st.title("🧠 AI Learns RAW Data + Auto-Calculates Emission Factor")
st.markdown("**Saves every row + predictions + learns optimal CO₂ factor**")

# Sidebar - Enhanced
with st.sidebar:
    st.header("🧠 AI Learning")
    st.metric("📚 Datasets Learned", len(ai_brain.history))
    st.metric("📋 Full Tables Saved", len(ai_brain.full_data_history))
    if ai_brain.history:
        current_factor = ai_brain.calculate_optimal_emission_factor()
        st.metric("🔬 CO₂ Factor", f"{current_factor:.3f} kg/kg")

# 🆕 TABS
tab1, tab2 = st.tabs(["📊 Instant Analysis", "🔮 CO₂ Prediction"])

with tab1:
    st.subheader("📁 Upload Excel File")
    uploaded_file1 = st.file_uploader("Choose Excel (.xlsx)", type=["xlsx"], key="analysis")

    if uploaded_file1:
        try:
            df_raw = pd.read_excel(uploaded_file1)
            df_raw.columns = df_raw.columns.str.strip()
            st.success(f"✅ Loaded **{len(df_raw)} rows × {len(df_raw.columns)} columns**")

            # 🆕 PIE CHARTS FOR ALL COLUMNS
            st.subheader("🥧 **Instant Analysis - All Columns**")
            for col_name in df_raw.columns:
                with st.container():
                    col1, col2 = st.columns([1, 4])
                    with col2:
                        value_counts = df_raw[col_name].value_counts().head(8)
                        if len(value_counts) > 0:
                            fig, ax = plt.subplots(figsize=(6, 5))
                            colors = plt.cm.Set3(np.linspace(0, 1, len(value_counts)))
                            ax.pie(value_counts.values, labels=value_counts.index, 
                                   autopct='%1.1f%%', startangle=90, colors=colors)
                            ax.set_title(f"📊 {col_name}", fontweight='bold')
                            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")

with tab2:
    st.subheader("📁 Upload Excel File")
    uploaded_file2 = st.file_uploader("Choose Excel (.xlsx)", type=["xlsx"], key="predict")

    if uploaded_file2:
        try:
            df_raw = pd.read_excel(uploaded_file2)
            df_raw.columns = df_raw.columns.str.strip()
            st.success(f"✅ Loaded **{len(df_raw)} rows × {len(df_raw.columns)} columns**")

            # Column mapping
            st.subheader("🔧 **CO₂ Model Setup**")
            col_options = ["None"] + list(df_raw.columns)
            col1, col2, col3 = st.columns(3)
            with col1: charcoal_col = st.selectbox("🥫 Charcoal", col_options)
            with col2: household_col = st.selectbox("👨‍👩‍👧‍👦 Households", col_options)
            with col3: freq_col = st.selectbox("📅 Frequency", col_options)
            
            # Additional optional column for Ground Truth
            co2_gt_col = st.selectbox("🧪 CO₂ Ground Truth (Optional)", ["None"] + list(df_raw.columns), 
                                      help="If you have actual CO₂ measurements, select this to calibrate the AI.")

            charcoal_unit = st.radio("📏 Unit", ["bags (350g)", "grams"], index=0)

            ai_factor = ai_brain.calculate_optimal_emission_factor()
            st.info(f"🤖 **AI Learned Factor: {ai_factor:.3f} kg CO₂/kg charcoal**")

            if all(c != "None" for c in [charcoal_col, household_col, freq_col]):
                if st.button("🚀 **TRAIN AI + SAVE FULL DATA**", type="primary", use_container_width=True):
                    
                    # Process data
                    df_work = df_raw.copy()
                    df_work = df_work.rename(columns={
                        charcoal_col: "Charcoal_per_use",
                        household_col: "Household_Size", 
                        freq_col: "Frequency"
                    })

                    def parse_charcoal(val):
                        try:
                            return float(str(val).replace("bags", "").replace("bag", "").replace(",", ".").strip())
                        except: return 0.0

                    df_work["Charcoal_per_use"] = df_work["Charcoal_per_use"].apply(parse_charcoal)
                    
                    if charcoal_unit == "bags (350g)":
                        df_work["Charcoal_per_use_kg"] = df_work["Charcoal_per_use"] * 0.35
                    else:
                        df_work["Charcoal_per_use_kg"] = df_work["Charcoal_per_use"] / 1000

                    freq_map = {"daily": 7, "once a week": 1, "twice a week": 2, "occasionally": 0.5, "never": 0}
                    df_work["Frequency_per_week"] = df_work["Frequency"].astype(str).str.lower().map(freq_map).fillna(1)
                    
                    df_work["Avg_Charcoal_kg"] = df_work["Charcoal_per_use_kg"] * df_work["Frequency_per_week"]
                    df_work["Households"] = pd.to_numeric(df_work["Household_Size"], errors='coerce').fillna(1)
                    df_work["Total_Charcoal_kg"] = df_work["Avg_Charcoal_kg"] * df_work["Households"]
                    df_work["Week"] = np.arange(1, len(df_work) + 1)
                    
                    # -----------------------------------------------
                    # SMARTER FACTOR CALCULATION
                    # -----------------------------------------------
                    is_ground_truth = False
                    
                    if co2_gt_col != "None":
                        # CASE A: GROUND TRUTH EXISTS
                        try:
                            # Use df_raw to ensure we get the column by its original name (df_work has renames)
                            df_work["CO2_kg"] = pd.to_numeric(df_raw[co2_gt_col], errors='coerce').fillna(0)
                            # Calculate factor from this specific dataset
                            valid_rows = df_work[ (df_work["Total_Charcoal_kg"] > 0) & (df_work["CO2_kg"] > 0) ]
                            if not valid_rows.empty:
                                current_interaction_factor = valid_rows["CO2_kg"].sum() / valid_rows["Total_Charcoal_kg"].sum()
                                optimal_factor = current_interaction_factor # Use THIS dataset's factor
                                is_ground_truth = True
                                st.success(f"🧪 **Ground Truth Detected!** Calibrated Factor: {optimal_factor:.3f}")
                            else:
                                st.warning("⚠️ CO₂ column selected but contained no valid data. Using AI prediction.")
                                optimal_factor = ai_brain.calculate_optimal_emission_factor()    
                        except Exception as e:
                            st.error(f"Error reading CO₂ column: {e}")
                            optimal_factor = ai_brain.calculate_optimal_emission_factor()
                    else:
                        # CASE B: PREDICTION (No Ground Truth)
                        optimal_factor = ai_brain.calculate_optimal_emission_factor()
                        # Only apply factor if we check "Theoretical" but here we just apply it
                        # to generate the "predictions"
                    
                    # Prepare dataframe (calculates 'CO2_kg' using factor if not present, or overwrites?)
                    # If we have Ground Truth, we trust valid rows, but 'prepare_data' currently OVERWRITES CO2_kg
                    # Let's modify logic: ONE source of truth.
                    
                    # If NOT ground truth, we calculate CO2
                    if not is_ground_truth:
                        df_processed = prepare_data(df_work, optimal_factor)
                    else:
                        df_processed = df_work.copy()
                        # Ensure all columns exist for consistency
                        if "CO2_kg" not in df_processed.columns:
                             df_processed["CO2_kg"] = df_processed["Total_Charcoal_kg"] * optimal_factor

                    # Standard model fitting
                    a, b, model = fit_consumption_model(df_processed)
                    r2 = model.score(df_processed["Week"].values.reshape(-1, 1), df_processed["Total_Charcoal_kg"])
                    
                    predictions = model.predict(df_processed["Week"].values.reshape(-1, 1))
                    df_processed['Predicted_Charcoal'] = predictions
                    df_processed['Prediction_Error'] = df_processed['Total_Charcoal_kg'] - predictions

                    emissions_1yr = total_emissions(52, a, b, optimal_factor)
                    emissions_2yr = total_emissions(104, a, b, optimal_factor)
                    emissions_5yr = total_emissions(260, a, b, optimal_factor)

                    # 🆕 SAVE ALL TABLES + LISTS - JSON SAFE
                    ai_brain.save_full_dataset(df_raw, df_processed, a, b, r2, predictions, optimal_factor, is_ground_truth=is_ground_truth)
                    
                    st.success(f"""
                        🎓 **AI FULLY TRAINED!** 
                        ✅ Saved {len(df_raw)} RAW table rows
                        ✅ Saved {len(df_processed)} processed table rows  
                        ✅ Saved {len(predictions)} predictions
                        ✅ Updated CO₂ factor: {optimal_factor:.3f}
                        ✅ Total full datasets: {len(ai_brain.full_data_history)}
                    """)

                    col1, col2, col3, col4 = st.columns(4)
                    with col1: st.metric("1Y CO₂", f"{emissions_1yr:,.0f} kg")
                    with col2: st.metric("2Y CO₂", f"{emissions_2yr:,.0f} kg")
                    with col3: st.metric("R² Fit", f"{r2:.3f}")
                    with col4: st.metric("CO₂ Factor", f"{optimal_factor:.3f}")

                    st.subheader("📊 Sample Results (First 10 rows)")
                    st.dataframe(df_processed[['Week', 'Total_Charcoal_kg', 'Predicted_Charcoal', 'Prediction_Error', 'CO2_kg']].head(10).round(2))

                    st.subheader("🌍 5-Year CO₂ Forecast")
                    fig, ax = plt.subplots(figsize=(12, 8))
                    max_weeks = 260
                    weeks_long = np.linspace(0, max_weeks, 200)
                    cum_long = np.array([total_emissions(t, a, b, optimal_factor) for t in weeks_long])
                    
                    ax.plot(weeks_long/52, cum_long/1000, linewidth=4, color='darkgreen', label='Total CO₂')
                    ax.fill_between(weeks_long/52, cum_long/1000, alpha=0.3, color='green')
                    
                    years = [1, 2, 3, 4, 5]
                    for year in years:
                        emissions_tons = total_emissions(year*52, a, b, optimal_factor) / 1000
                        ax.axvline(x=year, linestyle='--', alpha=0.7, linewidth=2, color='red')
                        ax.plot(year, emissions_tons, 'o', markersize=12, color='red')
                        ax.annotate(f'{year}Y\n{emissions_tons:.0f}t', (year, emissions_tons), 
                                   xytext=(5, 5), textcoords='offset points')
                    
                    ax.set_xlabel("Years"); ax.set_ylabel("CO₂ (metric tons)")
                    ax.set_title(f"5-Year Forecast (CO₂ Factor: {optimal_factor:.3f})")
                    ax.legend(); ax.grid(True, alpha=0.3)
                    st.pyplot(fig)

        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")
            st.exception(e)

    else:
        st.info("👆 **Upload → AI saves ALL table data + learns optimal CO₂ factor**")

st.markdown("---")
st.markdown("<p style='text-align:center;'>🧠 Saves Full Tables • All Rows • Complete Predictions</p>", unsafe_allow_html=True)
