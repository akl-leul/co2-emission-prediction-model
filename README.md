# Charcoal CO₂ Emissions Predictor - Comprehensive System Documentation

## 1. Project Overview

**The Problem:** Charcoal consumption in households is often irregular and difficult to track. Traditional methods use simple averages which fail to capture trends (e.g., a family slowly reducing usage). Furthermore, the conversion from *Charcoal Mass* to *CO₂ Emissions* (the Emission Factor) varies widely based on charcoal quality and stove efficiency.

**The Solution:** This application is an **AI-Powered Modeling System** that:
1.  **Ingests Raw Survey Data**: Accepts Excel files with diverse column structures.
2.  **Learns Consumption Patterns**: Uses Linear Regression to model how consumption changes over time ($C(t)$).
3.  **Self-Calibrates CO₂ Factors**: Uses "Ground Truth" data (actual measured emissions) to learn the specific Emission Factor for a context, rather than relying on global constants.
4.  **Forecasts Future Emissions**: Uses Calculus (Integration) to accurately sum up total emissions over 1, 2, or 5 years.

---

## 2. Technical Architecture (`app.py`)

The application is built on **Streamlit** for the frontend and **Pandas/Scikit-Learn/SciPy** for the backend logic.

### 2.1. Library Imports & Dependencies

```python
import json
import os
import numpy as np
import pandas as pd
import streamlit as st
from scipy.integrate import quad
from sklearn.linear_model import LinearRegression
```

-   **`json/os`**: Essential for the "AI Brain" persistence layer. We save state to disk so the AI "remembers" across sessions.
-   **`numpy/pandas`**: The workhorses of data manipulation.
-   **`scikit-learn`**: Used for the `LinearRegression` model. We chose linear regression over complex neural networks because consumption trends over short periods (weeks/months) are typically linear or stable, and we need *interpretability* (Slope = rate of change).
-   **`scipy.integrate.quad`**: This is a critical component. We don't just multiply `Avg * 52 weeks`. We integrate the continuous rate function over time. This accounts for the *Slope* of change. If a family reduces usage by 1% per week, `quad` captures the exact curve, whereas simple multiplication would overestimate.

### 2.2. Physics & Mathematics Models

#### The Consumption function $C(t)$
We model charcoal consumption as a linear function of time:
$$ C(t) = a \cdot t + b $$
Where:
-   $t$ is time in weeks.
-   $a$ is the **Rate of Change** (Slope).Negative $a$ means usage is decreasing.
-   $b$ is the **Initial Consumption** (Intercept).

Implemented in `fit_consumption_model` (Lines 25-30):
```python
model = LinearRegression()
model.fit(X, y) # X=Week, y=Kg
```

#### The Emission Rate $E(t)$
Emissions are proportional to consumption:
$$ E(t) = \text{Factor} \cdot C(t) = \text{Factor} \cdot (a \cdot t + b) $$

#### Total Emissions Integral
To get total emissions over a duration $T$, we calculate the area under the curve:
$$ \text{Total}(T) = \int_{0}^{T} E(t) \, dt = \int_{0}^{T} \text{Factor} \cdot (at+b) \, dt $$

Implemented in `total_emissions` (Lines 35-37):
```python
def total_emissions(T, a, b, emission_factor):
    result, _ = quad(emission_rate, 0, T, args=(a, b, emission_factor))
    return result
```
*Why Integration?* It gives mathematical precision. For a 5-year forecast ($T=260$ weeks), slight errors in simple multiplication compound significantly. Integration is exact for the modeled linear trend.

---

## 3. The "AI Brain" (Self-Learning System)

The core innovation of this app is the `LearningAI` class (Lines 57-182). It separates this tool from a simple Calculator.

### 3.1. Persistence Layer
The AI maintains two files:
1.  **`ai_learning_history.json`**: A lightweight summary of every training run (Timestamp, R2 score, Calculated Factor). Used for quick metrics.
2.  **`ai_full_data_history.json`**: A **Deep Storage** formatted file.
    -   It saves the **ENTIRE MAPPED DATAFRAME** for every upload.
    -   It saves **EVERY PREDICTION**.
    -   This allows the AI to re-analyze past data in future updates (e.g., if we upgrade the model to Polynomial Regression, we can re-train on old data).

### 3.2. Smart Calibration Logic (`calculate_optimal_emission_factor`)

This method determines the "Truth" about how much CO₂ 1kg of charcoal produces.

**The Logic Flow:**
1.  **Ground Truth Scan**:
    -   The code iterates through `self.full_data_history`.
    -   It checks for the flag `is_ground_truth=True`.
    -   This flag is set ONLY when a user explicitly uploads a file containing measured `CO2_kg` data.
2.  **Evidence Extraction**:
    -   If Ground Truth datasets are found, it extracts the ratio $Ratio = \frac{\sum CO_2}{\sum Charcoal}$ for those specific datasets.
    -   It computes the **mean** of these real-world ratios.
    -   *Result*: The AI ignores theoretical defaults and uses the Empirical Average of user data.
3.  **Fallback (Prediction Mode)**:
    -   If NO Ground Truth data exists in history, it falls back to a scientific default or historical average.
    -   Default: **2.93 kg CO₂ / kg Charcoal**.
    -   *Derivation*: Charcoal is ~80% Carbon. $C + O_2 \to CO_2$. Molar mass ratio $44/12 \approx 3.66$. $0.80 \times 3.66 = 2.93$.

### 3.3. Serialization (`save_full_dataset`)
Python objects (like pandas DataFrames and NumPy arrays) cannot be saved to JSON directly.
The method `convert_to_json_serializable` (Lines 39-51) handles this mapping:
-   `pd.Timestamp` $\to$ ISO Format String
-   `np.int64` $\to$ Python `int`
-   `np.float64` $\to$ Python `float`
-   `np.ndarray` $\to$ Python `list`

---

## 4. User Interface Workflow

### 4.1. "Instant Analysis" Tab
**Goal**: Immediate data intuition.
-   Uses `st.file_uploader` to ingest Excel.
-   **Loop through columns**:
    ```python
    for col_name in df_raw.columns:
        value_counts = df_raw[col_name].value_counts()
        # Plot Pie Chart
    ```
-   This provides a rapid visual check: "Is my data balanced? Do I have missing values?"

### 4.2. "CO₂ Prediction" Tab (The Main Engine)

#### Step 1: Mapping
The user must map their Excel columns to the Logic columns.
-   **Charcoal**: The raw amount (e.g., "Charcoal Amount").
-   **Households**: The multiplier (e.g., "Number of People").
-   **Frequency**: How often they buy (e.g., "Purchase Freq").

#### Step 2: Ground Truth Selection (NEW)
-   Dropdown: **"🧪 CO₂ Ground Truth (Optional)"**.
-   This is the critical input for the Learning System.
-   If selected, the app stops *predicting* CO₂ for this file and starts *reading* it to teach the AI.

#### Step 3: Data Processing
Inside the "TRAIN AI" button:
1.  **Unit Conversion**: Converts "bags" (350g) to "kg".
2.  **Frequency Normalization**: Converts "Daily", "Weekly", "Occasional" text to multipliers (7, 1, 0.5).
3.  **Physics Check**:
    -   If `Ground Truth` detected $\to$ Calculate Factor from Data $\to$ `is_ground_truth=True`.
    -   If `Ground Truth` NOT detected $\to$ `Factor = ai_brain.calculate_optimal_emission_factor()`.

#### Step 4: Forecasting
The app generates a matplotlib plot showing the 5-Year forecast.
-   **Green Zone**: Represents confidence/accumulated emissions.
-   **Red Markers**: Specific milestones (1yr, 2yr, 5yr totals).

---

## 5. Data Structures Explained

### 5.1. `ai_full_data_history.json` Structure
This file grows significantly over time. Periodic cleaning may be required for production systems.

```json
[
  {
    "timestamp": "2023-10-27T10:00:00",
    "source_file": "Excel Upload",
    "model_params": {
      "slope_a": -0.05,        // Usage decreasing by 0.05kg/week
      "intercept_b": 5.0,      // Started at 5.0kg
      "r2_score": 0.85,        // Good fit
      "emission_factor": 3.1   // Learned or Predicted Factor
    },
    "is_ground_truth": true,   // CRITICAL FLAG
    "raw_data_table": [ ... ], // Complete copy of uploaded Excel
    "processed_data_table": [ ... ], // Data used for training
    "forecasts": {
        "1yr": 1200.50,
        "5yr": 5400.20
    }
  }
]
```

---

## 6. How to Extend/Contribute

### Adding New Model Types
Currently, we only use `LinearRegression`. To add a **Polynomial Model**:
1.  Import `PolynomialFeatures` from sklearn.
2.  Modify `fit_consumption_model`:
    ```python
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model.fit(X_poly, y)
    ```
3.  Update the `total_emissions` integral function to integrate $at^2 + bt + c$.

### Adding New Data Formats
To support CSV or JSON upload:
1.  In `app.py`, under `st.file_uploader`, add `type=["csv", "json"]`.
2.  Add conditional loading:
    ```python
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    ```

---

## 7. Troubleshooting

**Error: "Required column missing"**
-   The user did not select a valid column in the dropdowns. Ensure "None" is not selected for Charcoal/Households.

**Error: "KeyError" on CO₂ Calculation**
-   This happens if the "Ground Truth" column name in the Excel file has leading/trailing spaces.
-   *Fix*: The code now includes `df.columns = df.columns.str.strip()` to auto-fix this.

**Issue: "Factor didn't change after upload"**
-   Did you select a **CO₂ Ground Truth** column?
-   If you didn't, the AI treated the file as "Test Data" and just applied its existing knowledge. It only learns (updates the Factor) from "Training Data" (files with Ground Truth).

---

## 8. License & Usage
This code is open-source.
-   **License**: MIT
-   **Author**: Antigravity AI
-   **Date**: December 2025
