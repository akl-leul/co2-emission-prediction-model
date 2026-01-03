import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.integrate import quad
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from supabase import create_client, Client
from emission_model import fit_consumption_model, emission_rate, total_emissions, prepare_data

# 🧠 ENHANCED SELF-LEARNING SYSTEM - SAVES FULL TABLES
LEARNING_FILE = "ai_learning_history.json"
FULL_DATA_FILE = "ai_full_data_history.json"

# 🗄️ SUPABASE CONFIGURATION
SUPABASE_URL = st.secrets.supabase.url
SUPABASE_KEY = st.secrets.supabase.key

def get_supabase_client() -> Client:
    """Initialize and return Supabase client"""
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        st.error(f"Failed to connect to Supabase: {e}")
        return None

def generate_frequency_distribution_table(df):
    """Generate frequency distribution table for key variables"""
    freq_tables = {}
    
    # Frequency distribution for key numeric columns - flexible column detection
    numeric_mappings = {
        'Total_Charcoal_kg': ['total_charcoal_kg', 'charcoal_kg', 'charcoal', 'charcoal amount', 'how much charcoal'],
        'CO2_kg': ['co2_kg', 'co2', 'carbon dioxide', 'emissions'],
        'Avg_Charcoal_kg': ['avg_charcoal_kg', 'average_charcoal', 'charcoal_per_use_kg', 'charcoal per use'],
        'Frequency_per_week': ['frequency_per_week', 'frequency', 'how often', 'usage frequency']
    }
    
    for target_col, possible_names in numeric_mappings.items():
        # Find matching column (case-insensitive)
        matching_col = None
        for col in df.columns:
            col_lower = col.lower().strip()
            if any(name in col_lower for name in possible_names):
                matching_col = col
                break
        
        if matching_col and matching_col in df.columns:
            # Create bins for better distribution analysis
            data = pd.to_numeric(df[matching_col], errors='coerce').dropna()
            if len(data) > 0:
                # Calculate appropriate bin size
                q1, q3 = data.quantile([0.25, 0.75])
                iqr = q3 - q1
                bin_width = 2 * iqr / (len(data) ** (1/3)) if iqr > 0 else 1
                
                # Create bins
                min_val = data.min()
                max_val = data.max()
                n_bins = min(10, max(5, int((max_val - min_val) / bin_width) + 1))
                
                # Create frequency table
                freq_table = pd.cut(data, bins=n_bins, include_lowest=True).value_counts().sort_index()
                freq_tables[target_col] = freq_table
    
    # Frequency distribution for categorical columns - flexible column detection
    categorical_mappings = {
        'Household_Size': ['household_size', 'household size', 'household', 'family size', 'family'],
        'Charcoal_per_use_kg': ['charcoal_per_use_kg', 'charcoal per use', 'charcoal amount', 'how much charcoal']
    }
    
    for target_col, possible_names in categorical_mappings.items():
        # Find matching column (case-insensitive)
        matching_col = None
        for col in df.columns:
            col_lower = col.lower().strip()
            if any(name in col_lower for name in possible_names):
                matching_col = col
                break
        
        if matching_col and matching_col in df.columns:
            data = df[matching_col].dropna()
            if len(data) > 0:
                # For numeric categorical data, create ranges
                try:
                    numeric_data = pd.to_numeric(data, errors='coerce').dropna()
                    if len(numeric_data) > 0:
                        data = numeric_data
                        is_numeric = True
                    else:
                        is_numeric = False
                except:
                    is_numeric = False
                
                if is_numeric:
                    if target_col == 'Household_Size':
                        # Household size typically 1-10
                        bins = [0, 1, 2, 3, 4, 5, 7, 10, float('inf')]
                        labels = ['1', '2', '3', '4', '5', '6-7', '8-10', '10+']
                    elif target_col == 'Charcoal_per_use_kg':
                        # Charcoal per use in kg
                        bins = [0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, float('inf')]
                        labels = ['<0.1kg', '0.1-0.2kg', '0.2-0.35kg', '0.35-0.5kg', '0.5-0.75kg', '0.75-1kg', '>1kg']
                    else:
                        bins = 5
                        labels = None
                    
                    freq_table = pd.cut(data, bins=bins, labels=labels, include_lowest=True).value_counts().sort_index()
                else:
                    freq_table = data.value_counts().sort_index()
                
                freq_tables[target_col] = freq_table
    
    return freq_tables

# --------------------------------------------------
# EMISSIONS MODEL - USING POLYNOMIAL REGRESSION ONLY
# --------------------------------------------------
# All model functions are now imported from emission_model.py

def convert_to_json_serializable(obj):
    """Convert pandas Timestamp and other non-JSON types to strings"""
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        # Handle NaN values
        if np.isnan(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.floating, np.float64)):
        # Handle NaN values
        if np.isnan(obj):
            return None
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float):
        # Handle NaN values for regular Python floats
        if np.isnan(obj):
            return None
        return obj
    return obj

def parse_charcoal(val):
    try:
        return float(str(val).replace("bags", "").replace("bag", "").replace(",", ".").strip())
    except: 
        return 0.0

class LearningAI:
    def __init__(self):
        self.history = []
        self.full_data_history = []
        self.supabase = get_supabase_client()
        self.load_history()
        self.load_full_data()

    def load_history(self):
        """Load history from Supabase, fallback to local file if needed"""
        if self.supabase:
            try:
                response = self.supabase.table("emission_summaries").select("*").order("timestamp", desc=True).execute()
                if response.data:
                    self.history = response.data
                else:
                    self.history = []
            except Exception as e:
                st.warning(f"Failed to load history from Supabase: {e}")
                self.history = []
        else:
            # Fallback to local file if Supabase is not available
            if os.path.exists(LEARNING_FILE):
                try:
                    with open(LEARNING_FILE, "r") as f:
                        self.history = json.load(f)
                except:
                    self.history = []

    def load_full_data(self):
        """Load full data from Supabase, fallback to local file if needed"""
        if self.supabase:
            try:
                response = self.supabase.table("emission_datasets").select("*").order("timestamp", desc=True).execute()
                if response.data:
                    self.full_data_history = response.data
                else:
                    self.full_data_history = []
            except Exception as e:
                st.warning(f"Failed to load full data from Supabase: {e}")
                self.full_data_history = []
        else:
            # Fallback to local file if Supabase is not available
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

    def save_full_dataset(self, raw_df, processed_df, coefficients, intercept, r2, predictions, emission_factor, is_ground_truth=False):
        """🆕 Save ALL table rows + complete data to Supabase and local files"""
        
        # Convert dataframes to JSON-serializable format
        raw_serializable = raw_df.reset_index(drop=True).map(convert_to_json_serializable).to_dict('records')
        processed_serializable = processed_df.reset_index(drop=True).map(convert_to_json_serializable).to_dict('records')
        
        # Summary entry for local history
        summary_entry = {
            "timestamp": datetime.now().isoformat(),
            "n_samples": len(processed_df),
            "n_raw_rows": len(raw_df),
            "slope_a": float(coefficients[0]) if len(coefficients) > 0 else 0.0,
            "intercept_b": float(intercept),
            "r2_score": float(r2),
            "auto_emission_factor": float(emission_factor),
            "emissions_1yr": float(total_emissions(52, coefficients, intercept, emission_factor)),
            "emissions_2yr": float(total_emissions(104, coefficients, intercept, emission_factor)),
        }
        self.history.append(summary_entry)
        
        # 🆕 FULL DATA: ALL RAW ROWS + ALL PROCESSED ROWS + PREDICTIONS
        full_entry = {
            "timestamp": datetime.now().isoformat(),
            "source_file": "Multiple Files Upload" if len(processed_df) > 0 else "Excel Upload",
            "n_raw_rows": len(raw_df),
            "n_processed_rows": len(processed_df),
            "model_params": {
                "slope_a": float(coefficients[0]) if len(coefficients) > 0 else 0.0,
                "intercept_b": float(intercept),
                "r2_score": float(r2),
                "emission_factor": float(emission_factor)
            },
            "is_ground_truth": is_ground_truth,
            "raw_data_table": raw_serializable,  # ALL RAW ROWS - JSON SAFE
            "processed_data_table": processed_serializable,  # ALL PROCESSED - JSON SAFE
            "predictions_list": predictions.tolist(),  # ALL PREDICTIONS
            "forecasts": {
                "1yr": float(total_emissions(52, coefficients, intercept, emission_factor)),
                "2yr": float(total_emissions(104, coefficients, intercept, emission_factor)),
                "3yr": float(total_emissions(156, coefficients, intercept, emission_factor)),
                "5yr": float(total_emissions(260, coefficients, intercept, emission_factor))
            }
        }
        self.full_data_history.append(full_entry)
        
        # 🗄️ SAVE TO SUPABASE
        if self.supabase:
            try:
                # Save summary to emission_summaries table
                summary_data = {
                    "timestamp": datetime.now().isoformat(),
                    "n_samples": len(processed_df),
                    "n_raw_rows": len(raw_df),
                    "slope_a": float(coefficients[0]) if len(coefficients) > 0 else 0.0,
                    "intercept_b": float(intercept),
                    "r2_score": float(r2),
                    "auto_emission_factor": float(emission_factor),
                    "emissions_1yr": float(total_emissions(52, coefficients, intercept, emission_factor)),
                    "emissions_2yr": float(total_emissions(104, coefficients, intercept, emission_factor))
                }
                self.supabase.table("emission_summaries").insert(summary_data).execute()
                
                # Save full dataset to emission_datasets table
                dataset_data = {
                    "timestamp": datetime.now().isoformat(),
                    "source_file": full_entry["source_file"],
                    "n_raw_rows": len(raw_df),
                    "n_processed_rows": len(processed_df),
                    "model_params": full_entry["model_params"],
                    "is_ground_truth": is_ground_truth,
                    "raw_data_table": raw_serializable,
                    "processed_data_table": processed_serializable,
                    "predictions_list": predictions.tolist(),
                    "forecasts": full_entry["forecasts"]
                }
                dataset_response = self.supabase.table("emission_datasets").insert(dataset_data).execute()
                
                # Save individual records to emission_records table
                if dataset_response.data:
                    dataset_id = dataset_response.data[0]["id"]
                    records = []
                    for idx, row in processed_df.iterrows():
                        def safe_float(value):
                            """Convert to float, handling NaN values"""
                            try:
                                val = float(value)
                                return None if np.isnan(val) else val
                            except (ValueError, TypeError):
                                return 0.0
                        
                        record = {
                            "dataset_id": dataset_id,
                            "week": int(row.get("Week", idx + 1)),
                            "total_charcoal_kg": safe_float(row.get("Total_Charcoal_kg", 0)),
                            "predicted_charcoal": safe_float(row.get("Predicted_Charcoal", 0)),
                            "prediction_error": safe_float(row.get("Prediction_Error", 0)),
                            "co2_kg": safe_float(row.get("CO2_kg", 0)),
                            "households": int(row.get("Households", 1)),
                            "avg_charcoal_kg": safe_float(row.get("Avg_Charcoal_kg", 0)),
                            "frequency_per_week": safe_float(row.get("Frequency_per_week", 1)),
                            "charcoal_per_use_kg": safe_float(row.get("Charcoal_per_use_kg", 0)),
                            "household_size": safe_float(row.get("Household_Size", 1))
                        }
                        records.append(record)
                    
                    # Insert records in batches to avoid size limits
                    batch_size = 100
                    for i in range(0, len(records), batch_size):
                        batch = records[i:i + batch_size]
                        self.supabase.table("emission_records").insert(batch).execute()
                
                st.success("✅ Data successfully saved to Supabase!")
                
            except Exception as e:
                st.error(f"❌ Failed to save to Supabase: {e}")
                st.info("💾 Data saved locally as fallback")
        
        # Save local files - KEEP LAST 20 FULL DATASETS
        with open(LEARNING_FILE, "w") as f:
            json.dump(self.history[-100:], f, indent=2)
        
        with open(FULL_DATA_FILE, "w") as f:
            json.dump(self.full_data_history[-20:], f, indent=2)

ai_brain = LearningAI()

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="CO2 Emission from charcoal in Addis Ababa",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# � CUSTOM CSS - PREMIUM ECO THEME
st.markdown("""
    <style>
    /* Import Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main Background & Gradient Header */
    .stApp {
        background-color: #0E1117;
    }
    
    h1, h2, h3 {
        color: #FFFFFF !important;
        font-weight: 800 !important;
    }
    
    /* Metrics Cards */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        color: #00CC96 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        color: #A0AEC0 !important;
    }
    div[data-testid="stMetric"] {
        background-color: #1A202C;
        border: 1px solid #2D3748;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: #00CC96;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        width: 100%;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1A202C;
        border-radius: 8px 8px 0 0;
        color: #CBD5E0;
        padding-top: 10px;
        padding-bottom: 10px;
        flex-grow: 1; /* Force tabs to fill width */
        text-align: center; /* Center text */
        justify-content: center;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00CC96 !important;
        color: #FFFFFF !important;
    }

    /* Buttons */
    div.stButton > button {
        background: linear-gradient(90deg, #00CC96 0%, #2E8B57 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%; /* Full width buttons */
    }
    div.stButton > button:hover {
        box-shadow: 0 10px 15px -3px rgba(46, 139, 87, 0.5);
        color: #FFFFFF;
        transform: scale(1.02);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #2D3748;
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        width: 100%;
    }
    [data-testid="stFileUploader"] section {
        padding: 3rem !important;
        background-color: #1A202C !important;
        border: 2px dashed #00CC96 !important;
        border-radius: 12px !important;
        min-height: 200px !important; /* Taller drop area */
        display: flex;
        align-items: center;
        justify-content: center;
    }
    [data-testid="stFileUploader"] section > input {
        min-height: 200px !important;
    }
    /* Hide the 'Limit 200MB' text if needed, or style it */
    [data-testid="stFileUploader"] .stMarkdown small {
        color: #A0AEC0 !important;
        font-size: 0.9em;
    }
    </style>
""", unsafe_allow_html=True)

# 🏞️ HERO SECTION
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🌿 CO2 Emission from charcoal in Addis Ababa")
    st.markdown("### Intelligent Emissions Modeling for Ethiopia")
    st.markdown("""
    This system uses **AI** to learn consumption patterns in **Addis Ababa** households.
    It self-calibrates using local *Ground Truth* data to quantify environmental impact.
    """)
with col2:
    # Just a placeholder visual metric or logo if needed
    st.metric("Addis Avg Factor", f"2.93", "kgCO₂/kg")

st.markdown("---")

# Sidebar - Enhanced Profile
with st.sidebar:
    st.markdown("## 🧠 Neural Engine")
    
    # Styled Container for metrics
    with st.container():
        st.markdown("### Model Status")
        st.metric("📚 Datasets Learned", len(ai_brain.history))
        st.metric("💾 Full Tables DB", len(ai_brain.full_data_history))
        
        if ai_brain.history:
            current_factor = ai_brain.calculate_optimal_emission_factor()
            delta_color = "normal"
            if current_factor > 3.0: delta_color = "inverse"
            st.metric("🔥 Current Factor", f"{current_factor:.3f}", "kg CO₂/kg", delta_color=delta_color)

    st.markdown("---")
    st.info("💡 **Tip:** Upload files with a 'CO₂' column to auto-calibrate the physics engine.")

# 🆕 TABS
tab1, tab2 = st.tabs(["📊 Instant Analysis", "🔮 CO₂ Prediction"])

with tab1:
    st.subheader("📁 Upload Excel File(s)")
    uploaded_files1 = st.file_uploader("Choose Excel (.xlsx)", type=["xlsx"], key="analysis", accept_multiple_files=True)

    if uploaded_files1:
        try:
            # Combine all files
            df_list = []
            for file in uploaded_files1:
                df_temp = pd.read_excel(file)
                df_list.append(df_temp)
            
            df_raw = pd.concat(df_list, ignore_index=True)
            df_raw.columns = df_raw.columns.str.strip()
            
            file_count = len(uploaded_files1)
            st.success(f"✅ Loaded **{len(df_raw)} rows** from **{file_count} file{'s' if file_count > 1 else ''}** (Columns: {len(df_raw.columns)})")

            # 🆕 PIE CHARTS FOR ALL COLUMNS
            st.markdown("### 🥧 column distribution")
            
            # Use 3 columns layout for charts
            chart_cols = st.columns(3)
            
            for i, col_name in enumerate(df_raw.columns):
                with chart_cols[i % 3]:
                    value_counts = df_raw[col_name].value_counts().head(8)
                    if len(value_counts) > 0:
                        # Dark Theme Plot
                        with plt.style.context("dark_background"):
                            fig, ax = plt.subplots(figsize=(5, 5))
                            # Premium Colors (Greens/Teals)
                            colors = plt.cm.summer(np.linspace(0.2, 0.8, len(value_counts)))
                            
                            wedges, texts, autotexts = ax.pie(
                                value_counts.values, 
                                labels=value_counts.index, 
                                autopct='%1.1f%%', 
                                startangle=90, 
                                colors=colors,
                                wedgeprops=dict(width=0.5, edgecolor='#1A202C'), # Donut chart style
                                textprops={'color': "white", 'fontsize': 9}
                            )
                            plt.setp(autotexts, size=8, weight="bold")
                            ax.set_title(f"{col_name}", fontweight='bold', color='#00CC96', pad=20)
                            
                            # Transparent background
                            fig.patch.set_facecolor('none')
                            ax.patch.set_facecolor('none')
                            
                            st.pyplot(fig, width="stretch")
            
            # 🆕 FREQUENCY DISTRIBUTION ANALYSIS FOR INSTANT ANALYSIS
            st.markdown("### 📈 Frequency Distribution Analysis")
            
            try:
                # Create frequency distribution tables for the raw data
                freq_tables = generate_frequency_distribution_table(df_raw)
                
                if freq_tables:
                    # Create tabs for different frequency tables
                    freq_tab_names = [f"{col.replace('_', ' ').title()}" for col in freq_tables.keys()]
                    freq_tabs = st.tabs(freq_tab_names)
                    
                    for i, (col_name, freq_table) in enumerate(freq_tables.items()):
                        with freq_tabs[i]:
                            # Convert to display format
                            display_df = freq_table.reset_index()
                            display_df.columns = ['Range/Value', 'Frequency']
                            display_df['Percentage'] = (display_df['Frequency'] / display_df['Frequency'].sum() * 100).round(1)
                            
                            # Add statistics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Count", display_df['Frequency'].sum())
                            with col2:
                                st.metric("Most Frequent", f"{display_df.iloc[display_df['Frequency'].idxmax(), 0]}")
                            with col3:
                                st.metric("Categories", len(display_df))
                            
                            # Display the table
                            st.dataframe(display_df, use_container_width=True)
                            
                            # Simple bar chart
                            with plt.style.context("dark_background"):
                                fig, ax = plt.subplots(figsize=(10, 4))
                                fig.patch.set_facecolor('#0E1117')
                                ax.patch.set_facecolor('#0E1117')
                                
                                # Create bar chart
                                bars = ax.bar(range(len(display_df)), display_df['Frequency'], 
                                            color='#00CC96', alpha=0.8)
                                
                                # Customize
                                ax.set_xlabel(col_name.replace('_', ' ').title(), color='gray')
                                ax.set_ylabel('Frequency', color='gray')
                                ax.set_title(f'Distribution of {col_name.replace("_", " ").title()}', 
                                           color='#00CC96', fontweight='bold')
                                
                                # Set x-axis labels
                                if len(display_df) <= 10:
                                    ax.set_xticks(range(len(display_df)))
                                    ax.set_xticklabels(display_df['Range/Value'], rotation=45, ha='right')
                                else:
                                    ax.set_xticks([])
                                
                                # Grid and styling
                                ax.grid(color='#2D3748', linestyle='--', linewidth=0.5, alpha=0.5)
                                ax.spines['top'].set_visible(False)
                                ax.spines['right'].set_visible(False)
                                ax.spines['left'].set_color('#2D3748')
                                ax.spines['bottom'].set_color('#2D3748')
                                
                                st.pyplot(fig, width="stretch")
                else:
                    st.info("No frequency distribution data available")
                    
            except Exception as freq_error:
                st.warning(f"⚠️ Frequency analysis unavailable: {str(freq_error)}")
            
        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")

with tab2:
    st.subheader(" Upload Excel File(s)")
    uploaded_files2 = st.file_uploader("Choose Excel (.xlsx)", type=["xlsx"], key="predict", accept_multiple_files=True)

    if uploaded_files2:
        try:
            # First, let's get the column mappings
            st.subheader("")
            # Read the first file and strip whitespace from column names
            df_sample = pd.read_excel(uploaded_files2[0])
            df_sample.columns = df_sample.columns.str.strip()
            col_options = ["None"] + list(df_sample.columns)
            
            # Smart defaults based on new survey questions
            default_charcoal = "None"
            default_household = "None"
            default_freq = "None"
            
            for col in col_options[1:]:  # Skip "None" option
                c_lower = col.lower().strip()
                if "how much charcoal" in c_lower: default_charcoal = col
                elif "charcoal" in c_lower and default_charcoal == "None": default_charcoal = col
                
                if "households" in c_lower: default_household = col
                elif "household" in c_lower and default_household == "None": default_household = col
                
                if "often do" in c_lower and "use charcoal" in c_lower: default_freq = col
                elif "frequency" in c_lower and default_freq == "None": default_freq = col
            
            # Get column selections
            col1, col2, col3 = st.columns(3)
            with col1: 
                charcoal_index = col_options.index(default_charcoal)
                charcoal_col = st.selectbox(" Charcoal Amount", col_options, index=charcoal_index)
            with col2: 
                household_index = col_options.index(default_household)
                household_col = st.selectbox(" Households", col_options, index=household_index)
            with col3: 
                freq_index = col_options.index(default_freq)
                freq_col = st.selectbox(" Frequency / Usage", col_options, index=freq_index)
            
            # Additional optional column for Ground Truth
            co2_gt_col = st.selectbox(" CO₂ Ground Truth (Optional)", ["None"] + col_options[1:], 
                                      help="If you have actual CO₂ measurements, select this to calibrate the AI.")
            
            # Process each file with its own unit selection
            processed_dfs = []
            
            for file in uploaded_files2:
                with st.expander(f" {file.name}"):
                    # Read the file
                    df_temp = pd.read_excel(file)
                    df_temp.columns = df_temp.columns.str.strip()
                    
                    # Let user select unit for this file
                    file_charcoal_unit = st.radio(
                        " Unit for this file",
                        ["bags (350g)", "grams"],
                        key=f"unit_{file.name}",
                        index=0
                    )
                    
                    # Store the processed dataframe with its unit
                    processed_dfs.append({
                        'df': df_temp,
                        'unit': file_charcoal_unit,
                        'name': file.name
                    })
            
            # Now process each file with its selected unit
            final_dfs = []
            for item in processed_dfs:
                df_temp = item['df']
                file_charcoal_unit = item['unit']
                
                # Process the dataframe with its specific unit
                df_work = df_temp.copy()
                # Clean column names by stripping whitespace
                df_work.columns = df_work.columns.str.strip()
                # First create the column with the original data
                df_work["Charcoal_per_use"] = df_work[charcoal_col.strip()].copy()
                df_work["Household_Size"] = df_work[household_col.strip()].copy()
                df_work["Frequency"] = df_work[freq_col.strip()].copy()
                
                # Then apply the parsing function
                df_work["Charcoal_per_use"] = df_work["Charcoal_per_use"].apply(parse_charcoal)
                
                if file_charcoal_unit == "bags (350g)":
                    df_work["Charcoal_per_use_kg"] = df_work["Charcoal_per_use"] * 0.35
                else:
                    df_work["Charcoal_per_use_kg"] = df_work["Charcoal_per_use"] / 1000
                
                final_dfs.append(df_work)
            
            # Combine all processed dataframes
            if final_dfs:
                df_raw = pd.concat(final_dfs, ignore_index=True)
            else:
                st.error("No valid data to process")
                st.stop()
            
            file_count = len(uploaded_files2)
            st.success(f" Loaded **{len(df_raw)} rows** from **{file_count} file{'s' if file_count > 1 else ''}** (Columns: {len(df_raw.columns)})")

            # Column mapping already done before file processing

            ai_factor = ai_brain.calculate_optimal_emission_factor()
            st.info(f"🤖 **AI Learned Factor: {ai_factor:.3f} kg CO₂/kg charcoal**")

            if all(c != "None" for c in [charcoal_col, household_col, freq_col]):
                if st.button("🚀 **TRAIN AI + SAVE FULL DATA**", type="primary", width="stretch"):
                    
                    # parse_charcoal function is now defined at the top level
                    # Process data (now handled in the file upload section)
                    df_work = df_raw.copy()  # Already processed with individual units

                    # -----------------------------------------------
                    # SMART FREQUENCY PARSING
                    # -----------------------------------------------
                    def parse_frequency(val, col_name):
                        val_str = str(val).lower()
                        
                        # Logic for "How many hours per day..."
                        if "hours per day" in col_name.lower():
                            try:
                                hours = float(val)
                                if hours > 0: return 7  # Daily use if > 0 hours
                                return 0
                            except:
                                return 0
                                
                        # Standard logic
                        freq_map = {"daily": 7, "once a week": 1, "twice a week": 2, "occasionally": 0.5, "never": 0}
                        return freq_map.get(val_str, 1) # Default to 1 if unknown text

                    df_work["Frequency_per_week"] = df_work.apply(lambda x: parse_frequency(x["Frequency"], freq_col), axis=1)
                    
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

                    # Polynomial model fitting
                    coefficients, intercept, model = fit_consumption_model(df_processed)
                    r2 = model.score(df_processed["Week"].values.reshape(-1, 1), df_processed["Total_Charcoal_kg"])
                    
                    predictions = model.predict(df_processed["Week"].values.reshape(-1, 1))
                    df_processed['Predicted_Charcoal'] = predictions
                    df_processed['Prediction_Error'] = df_processed['Total_Charcoal_kg'] - predictions

                    emissions_1yr = total_emissions(52, coefficients, intercept, optimal_factor)
                    emissions_2yr = total_emissions(104, coefficients, intercept, optimal_factor)
                    emissions_5yr = total_emissions(260, coefficients, intercept, optimal_factor)

                    # 🆕 SAVE ALL TABLES + LISTS - JSON SAFE
                    ai_brain.save_full_dataset(df_raw, df_processed, coefficients, intercept, r2, predictions, optimal_factor, is_ground_truth=is_ground_truth)
                    
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

                    st.subheader("📊 All Data Results")
                    st.dataframe(df_processed[['Week', 'Total_Charcoal_kg', 'Predicted_Charcoal', 'Prediction_Error', 'CO2_kg']].round(2))

                    st.subheader("🌍 5-Year CO₂ Forecast")
                    
                    with plt.style.context("dark_background"):
                        fig, ax = plt.subplots(figsize=(12, 6))
                        # Dark canvas
                        fig.patch.set_facecolor('#0E1117')
                        ax.set_facecolor('#0E1117')
                        
                        max_weeks = 260
                        weeks_long = np.linspace(0, max_weeks, 200)
                        cum_long = np.array([total_emissions(t, coefficients, intercept, optimal_factor) for t in weeks_long])
                        
                        # Plot Line with Neon Glow effect
                        ax.plot(weeks_long/52, cum_long/1000, linewidth=3, color='#00CC96', label='Cumulative CO₂')
                        ax.fill_between(weeks_long/52, cum_long/1000, alpha=0.1, color='#00CC96')
                        
                        # Milestones
                        years = [1, 2, 3, 4, 5]
                        for year in years:
                            emissions_tons = total_emissions(year*52, coefficients, intercept, optimal_factor) / 1000
                            # Vertical dashed lines
                            ax.axvline(x=year, linestyle=':', alpha=0.5, linewidth=1, color='gray')
                            # Points
                            ax.plot(year, emissions_tons, 'o', markersize=8, color='#FFFFFF', markeredgecolor='#00CC96', markeredgewidth=2)
                            # Annotations
                            ax.annotate(f'{emissions_tons:.1f}t', (year, emissions_tons), 
                                       xytext=(0, 10), textcoords='offset points', 
                                       ha='center', color='white', fontweight='bold', fontsize=9)
                        
                        # Styling
                        ax.set_xlabel("Years into Future", color='gray')
                        ax.set_ylabel("Metric Tons CO₂", color='gray')
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['left'].set_color('#2D3748')
                        ax.spines['bottom'].set_color('#2D3748')
                        ax.grid(color='#2D3748', linestyle='--', linewidth=0.5, alpha=0.5)
                        
                        st.pyplot(fig, width="stretch")

        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")
            st.exception(e)

    else:
        st.info("👆 **Upload → AI saves ALL table data + learns optimal CO₂ factor**")

st.markdown("---")
st.markdown("<p style='text-align:center;'>🧠 Saves Full Tables • All Rows • Complete Predictions</p>", unsafe_allow_html=True)
