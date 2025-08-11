# UBC Campus Energy Centre - Mass and Energy Balance Analysis

<img width="1500" height="1000" alt="image" src="https://github.com/user-attachments/assets/ba80cd65-d0f3-4851-bced-f64dcf3741f8" />

## Table of Contents
- [Introduction](#introduction)
- [Background](#background)
- [Data Description](#data-description)
- [Installation](#installation)
- [Usage](#usage)
- [Code Overview](#code-overview)
- [Theoretical Framework](#theoretical-framework)
- [Results and Interpretation](#results-and-interpretation)
- [Limitations and Assumptions](#limitations-and-assumptions)
- [References](#references)

## Introduction

This project implements material and energy balance calculations for analyzing the performance of Boiler 2 at the UBC Campus Energy Centre (CEC). The analysis includes combustion calculations, thermal efficiency, and NOx emissions estimation using real operational data.

## Background

The Campus Energy Centre at UBC utilizes natural gas-fired boilers to provide heating across campus. This project focuses on Boiler 2, which was identified as having the most reliable dataset among the facility's boilers. The analysis is part of CHBE 366 coursework, demonstrating the application of chemical engineering principles to real-world energy systems.

Key objectives include:
- Performing mass and energy balances on the boiler system
- Calculating combustion efficiency and comparing it with operational data
- Estimating NOx emissions and validating against sensor measurements
- Identifying operational patterns and potential optimization opportunities

## Data Description

The analysis uses `B2 Cleaned (from pranav).csv`, which contains hourly operational data from Boiler 2 throughout 2021. Key parameters include:

### Input Parameters
- **Boiler Operation**: Firing rate, gas flow rate, gas pressure
- **Temperatures**: 
  - Entering/leaving water temperatures
  - Exhaust gas temperature
  - Ambient conditions (UBC temperature)
- **Composition**: 
  - Exhaust gas composition (O₂, CO₂, CO, NOx)
  - Natural gas composition (95% CH₄, 5% C₂H₆)
  - Humidity data

### Output Parameters
- **Performance Metrics**:
  - Boiler efficiency
  - Energy transfer rates
  - Molar flow rates of flue gas components
  - NOx emissions

### Data Preprocessing
1. **Unit Handling**: All parameters are standardized to SI units
2. **Missing Data**: Forward-filled for up to 2 consecutive missing values
3. **Outlier Removal**: Negative flow rates and physically implausible values are filtered out
4. **Time Series**: Data is indexed by timestamp for time-series analysis

## Installation

### Prerequisites
- Python 3.7+
- Required Python packages:
  ```
  pandas
  numpy
  matplotlib
  sympy
  scipy
  ```

### Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd data_cleaning_and_analytics
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Data Preparation**:
   - Place `B2 Cleaned (from pranav).csv` in the project directory
   - Ensure the CSV contains all required columns (see Data Description)

2. **Running the Analysis**:
   ```bash
   python data_analytics_chbe_project.py
   ```
   
   The script processes the data in sections. Run each section sequentially as indicated by the comments.

3. **Outputs**:
   - Processed data with calculated parameters
   - Visualizations of key performance indicators
   - Efficiency and emissions analysis

## Code Overview

The main script `data_analytics_chbe_project.py` is organized into the following logical sections:

### 1. Data Import and Preprocessing
- Loads the CSV file into a pandas DataFrame
- Performs initial data cleaning and unit conversions

### 2. Molar Flow Rate Calculations
- Implements the ideal gas law to determine fuel molar flow rates
- Calculates air and flue gas compositions
- Handles wet/dry air composition variations

### 3. Energy Balance Calculations
- Implements heat capacity calculations using Shomate equations
- Performs energy balances around the boiler system
- Calculates heat transfer rates and efficiency

### 4. Emissions Analysis
- Estimates NOx emissions using emission factors
- Compares calculated emissions with sensor data
- Implements conversion between different concentration units

### 5. Visualization and Output
- Generates time-series plots of key parameters
- Creates comparative analyses of calculated vs. measured values
- Exports processed data for further analysis

## Detailed Code Section Explanations

### Section 1: Importing Packages
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import sympy as sp
import math

data = pd.read_csv("B2 Cleaned (from pranav).csv")
data.head()
```
- **Purpose**: Initializes the Python environment with required libraries
- **Key Components**:
  - `pandas` for data manipulation
  - `numpy` for numerical operations
  - `matplotlib` for data visualization
  - `datetime` for handling timestamps
  - `sympy` for symbolic mathematics
 
  - **Data Operations**:
  - Loads the CSV file into a pandas DataFrame
  - Displays the first few rows for verification

### Section 2: Data Column Verification
```python
print(data.columns)
```
- **Purpose**: Verifies the structure of the input data
- **Output**: Lists all column names for reference
- **Validation**: Ensures all required columns are present

### Section 3: Molar Flow Rate Calculations
```python
def get_molar_flow_rate(row):
    P = row[' B-2 Gas Pressure, kPa']
    T = row['UBC Temp, °C']
    F = row[' B-2 Gas Flow Rate, m³/h']
    R = 8.314  # J/mol*K
    return P*1000*F / (R * (T + 273.15))

data['Fuel Molar Flow Rate, mole/h'] = data.apply(get_molar_flow_rate, axis=1)
```
- **Purpose**: Calculates molar flow rate using ideal gas law
- **Key Variables**:
  - `P`: Gas pressure (kPa)
  - `T`: Temperature (K)
  - `F`: Volumetric flow rate (m³/h)
  - `R`: Universal gas constant
    
  **Assumptions**:
  - Natural gas is at ambient temperature
  - Ideal gas behavior assumed
  - Fixed composition: 95% CH₄, 5% C₂H₆

### Section 4: Saturation Pressure of Water
```python
def sat_pressureofwater(row):
    A_water = 5.40221
    B_water = 1838.675
    C_water = -31.737
    T = row['UBC Temp, °C']
    return 10**(A_water - (B_water/(T + 273.15 + C_water))) * 100

data['Saturation Pressure of Water (kPA)'] = data.apply(sat_pressureofwater, axis=1)
```
- **Purpose**: Calculates water vapor saturation pressure using Antoine's equation
- **Output**: Adds a new column with saturation pressure values
- **Key Functions**:
  - `sat_pressureofwater()`: Implements Antoine equation
  - Converts between temperature scales (°C to K)

### Section 5: Mole Fraction of Water in Wet Air
```python
def mol_fracwater(row):
    partial_p = row['Partial Pressure of Water (kPA)']
    atmospheric_p = 101.325  # kPa
    return (partial_p/atmospheric_p)

data['Mole Fraction of Water in Wet Air'] = data.apply(mol_fracwater, axis=1)
```
- **Purpose**: Calculates mole fraction of water in wet air
- **Key Variables**:
  - `partial_p`: Partial pressure of water vapour (kPa)
  - `atmospheric_p`: Standard atmospheric pressure (101.325 kPa)

### Section 6: Molar Fraction of Oxygen and Nitrogen in Wet Air
```python
def yO2_wetair(row):
    molfrac_o2dryair = yO2
    yH2O_wetair = row['Mole Fraction of Water in Wet Air']
    return molfrac_o2dryair / (1 + yH2O_wetair)

def y_N2wetair(row):
    molfrac_n2dryair = yN2
    yH2O_wetair = row['Mole Fraction of Water in Wet Air']
    return molfrac_n2dryair / (1 + yH2O_wetair)

data['Molar Fraction of Nitrogen in Wet Air'] = data.apply(y_N2wetair, axis=1)
data['Molar Fraction of Oxygen in Wet Air'] = data.apply(yO2_wetair, axis=1)
```
- **Purpose**: Calculates molar fractions of O₂ and N₂ in wet air
- **Key Variables**:
  - `yO2`: Molar fraction of O₂ in dry air (0.21)
  - `yN2`: Molar fraction of N₂ in dry air (0.79)
 **Key Calculations**:
  - Adjusts dry air composition for water content
  - Verifies mole fraction sums to unity

### Section 7: Calculating Total Flue Gas and Air Flow Rate
```python
def fluegascomp(row):
    # Molecular weights (kg/mol)
    molweight_N2 = 28 * (1 / 1000)
    molweight_O2 = 32 * (1 / 1000)
    molweight_H2O = 18 * (1 / 1000)
    # ... (additional code)
    return G_mass, G_dryair, G_H2O_mole, A_mass, F, X_GC_mass, X_GC_mole, X_GN_mole, X_GH_mole, X_GO_mole, A_mole, G_mole

# Apply function and store results
results = data.apply(fluegascomp, axis=1, result_type='expand')
columns = {
    'Total Flue Gas Flow Rate, G (kg/hr)': 0,
    'Moisture in Flue Gas, fluegas_H2Ovap (mole/hr)': 2,
    # ... additional column mappings
}
for column_name, index in columns.items():
    data[column_name] = results[index]
```
- **Purpose**: Calculates comprehensive flue gas composition and flow rates
- **Key Outputs**:
  - Total flue gas flow rate
  - Component mole fractions
  - Moisture content

- **Key Features**:
  - Handles combustion stoichiometry
  - Calculates mass and mole balances
  - Considers both complete and incomplete combustion
    
### Section 8: Heat Capacity Coefficients and Temperature Conversions
```python
cp_coefffs = {
    'methane': {'A': 19.25, 'B': 0.05213, 'C': 1.197e-5, 'D': -1.132e-8, 'phase': 'gas'},
    'carbon_dioxide': {'A': 19.8, 'B': 0.07344, 'C': -5.602e-5, 'D': 1.715e-8, 'phase': 'gas'},
    # ... additional coefficients
}

def heat_capacity(substance, temp):
    if substance.lower() in cp_coefffs:
        regcoefficient = cp_coefffs[substance.lower()]
        A = regcoefficient['A']
        B = regcoefficient['B']
        C = regcoefficient['C']
        D = regcoefficient['D']
        return A + B * temp + C * temp**2 + D * temp**3
```
- **Purpose**: Defines heat capacity coefficients and calculation function
- **Key Features**:
  - Uses Shomate equation
  - Supports multiple species
  - Handles different phases

- **Data Structure**:
  - Dictionary of Shomate equation coefficients
  - Covers common combustion species (CH₄, CO₂, H₂O, etc.)
    
- **Implementation**:
  - `heat_capacity()`: Calculates Cp at given temperature
  - `calc_heatcap()`: Computes Cp for multiple components

### Section 9: Heat Capacity Calculations of Inlet and Outlet Components
```python
# Calculate heat capacities for different streams
for category, (temperature, description, cp_name) in substance_categories.items():
    for substance in globals()[category]:
        cp = heat_capacity(substance, temperature)
        data[f"Heat capacity of {substance} {description} J/mol-K"] = cp
```
- **Purpose**: Computes heat capacities for all process streams
- **Key Variables**:
  - `substance_categories`: Defines temperature conditions for each stream
  - `heat_capacity()`: Retrieves heat capacity values

### Section 10: Heat Capacities of Flue Gas Components
```python
# Calculate heat capacities for flue gas components
heatcapH2O_vapout = data['Heat capacity of water_gas in the outlet J/mol-K']
heatcapCO2out = data['Heat capacity of carbon_dioxide in the outlet J/mol-K']
# ... additional components

# Calculate dry and wet flue gas heat capacities
Cp_DFG = (ydryCO2*heatcapCO2out) + (ydryN2*heatcapN2out) + (ydryO2*heatcapO2out)
Cp_WFG = heatcapH2O_vapout

data['Heat Capacity of Dry Flue Gas (J/mol-K)'] = Cp_DFG
data['Heat Capacity of Wet Flue Gas (J/mol-K)'] = Cp_WFG
```
- **Purpose**: Computes heat capacities for flue gas components
- **Key Calculations**:
  - Weighted average for dry flue gas
  - Separate calculation for water vapor

### Section 11: Heat Flow Calculations
```python
# Calculate heat flows for different streams
heatflowfuel_inlet = (molarflow_fuel * heatcap_fuel * (tempkelvin - To) + 
                     ((0.05 * enthalpy_C2H6) + (0.95 * enthalpy_CH4)))
Heat_DryAir_in = MoleDryAir_IN * heatcap_DryAir * (tempkelvin - To)
Heat_WetAir_in = MoleWetAir_IN * ((heatcap_WetAir * (tempkelvin - To)) + heatofvapH2O)

# Store results
data['Heat Flow of Fuel(J/mol)'] = heatflowfuel_inlet
data['Heat of Dry Air (J/mol)'] = Heat_DryAir_in
data['Heat of Wet Air (J/mol)'] = Heat_WetAir_in
```
- **Purpose**: Calculates energy flows for combustion analysis
- **Key Variables**:
  - `heatflowfuel_inlet`: Fuel energy input
  - `Heat_DryAir_in`: Sensible heat of dry air
  - `Heat_WetAir_in`: Sensible + latent heat of moist air

### Section 12: Heat Loss Calculation
```python
# Energy balance calculations
HeatFlow_out = heatflow_water_out + heat_DFG_out + heat_WFG_out
HeatFlow_in = heatflow_water_in + heatflowfuel_inlet + Heat_DryAir_in + Heat_WetAir_in
HeatLoss = HeatFlow_in - HeatFlow_out

# Store results
data['Energy Lost (J/hr)'] = HeatLoss
```
- **Purpose**: Performs energy balance to determine heat loss
- **Key Calculations**:
  - Energy in = Energy from fuel + air
  - Energy out = Energy to water + flue gas
  - Heat loss = Energy in - Energy out

### Section 13: Flue Gas Composition and Flow Rate Over Time
```python
# Plot flue gas composition over time
plt.figure(figsize=(10, 6))
plt.plot(time, N2molfracFG, label='N2')
plt.plot(time, CO2molfracFG, label='CO2')
plt.plot(time, watermolfracFG, label='H2O')
plt.plot(time, O2molfracFG, label='O2')
plt.xlabel('Time (hr)')
plt.ylabel('Mole Fraction')
plt.legend()
plt.grid(False)
plt.show()
```
- **Purpose**: Visualizes flue gas composition trends
- **Key Features**:
  - Time-series plot of major components
  - Helps identify operational patterns

### Section 14: Hot Water, Cold Water, and Energy Transfer Rates
```python
# Plot energy transfer rates
plt.figure(figsize=(10, 6))
plt.plot(time, hwrate, label='Hot Water Transfer Rate (J/hr)')
plt.plot(time, cwrate, label='Cold Water Transfer Rate (J/hr)')
plt.plot(time, hloss, label='Net Heat Transfer (J/hr)')
plt.xlabel('Time(hr)')
plt.ylabel('Energy Transfer Rate (J/hr)')
plt.legend()
plt.grid(False)
plt.show()
```
- **Purpose**: Tracks energy transfer in the system
- **Key Metrics**:
  - Hot water energy rate
  - Cold water energy rate
  - Net heat transfer

### Section 15: Thermal Efficiency Calculations
```python
# Calculate boiler efficiency
eta = (Wflow * np.abs((HFhw - HFcw) / (Frate * deltaHc))) * 100

# Update dataset
data['Efficiency of Boiler over time'] = np.clip(eta, 0, 100)
data['Efficiency(%)'] = np.clip(eta, 0, 100)

# Plot efficiency comparison
plt.figure(figsize=(10, 6))
plt.plot(time, data['Efficiency(%)'], label='Calculated Efficiency (%)')
plt.plot(time, effskyspark, label='Thermal Efficiency From Skyspark (%)')
plt.xlabel('Time(hr)')
plt.ylabel('Thermal Efficiency (%)')
plt.legend()
plt.grid(False)
plt.show()
```
- **Purpose**: Calculates and visualizes boiler efficiency
- **Key Metrics**:
  - Thermal efficiency over time
  - Comparison with reference data

### Section 16: NOx Emissions Concentration Estimates
```python
def NOx_emission_conversion(row):
    exhaust_temp = row[' B-2 Exhaust Temp, °C'] + 273.15  # K
    NG_FR = row[' B-2 Gas Flow Rate, m³/h']  # m3/h
    AT_pressure = row[' B-2 Gas Pressure, kPa']
    volume = row['Total Flue Gas Flow Rate, G (mol/hr)'] * R * exhaust_temp / (AT_pressure * 1000)
    NOx_mass = factor * (NG_FR) / (1e+6)
    convert = NOx_mass / conv_for_NOx_ppmv
    return (convert / volume) if volume != 0 else 0

data['NOx emission, ppmv'] = data.apply(NOx_emission_conversion, axis=1)

# Plot NOx emissions
plt.figure(figsize=(10, 6))
plt.plot(time, data[' B-2 Exhaust NOx, ppm'], label='NOx Sensor (ppm)')
plt.plot(time, data['NOx emission, ppmv'], label='NOx Estimate (ppm)')
plt.xlabel('Time (hr)')
plt.ylabel('NOx Emission Concentration (ppm)')
plt.legend()
plt.grid(False)
plt.show()
```
- **Purpose**: Estimates and validates NOx emissions
- **Key Features**:
  - Implements emission factor method
  - Compares with sensor data
  - Handles unit conversions

### Section 17: Save the Cleaned Dataset
```python
# Save processed data to CSV
data.to_csv('complete_dataset_analytics.csv', index=True)
```
- **Purpose**: Exports processed data for further analysis
- **Output File**: `complete_dataset_analytics.csv`
- **Contents**: All calculated parameters and results

## Theoretical Framework

### 1. Molar Flow Rate (Ideal Gas Law)

$$\dot{n} = \frac{P \cdot \dot{V}}{R \cdot T}$$

- $P$: Gas pressure (Pa)
- $\dot{V}$: Volumetric flow rate (m³/s)
- $R$: Universal gas constant (8.314 J/mol·K)
- $T$: Temperature (K)

### 2. Saturation Pressure (Antoine Equation)

$$\log_{10}(P_{sat}) = A - \frac{B}{T + C}$$

- $A$, $B$, $C$: Substance-specific constants
- $T$: Temperature (K)

### 3. NOx Emissions

$$C_{NO_x} = \frac{F_{NG} \cdot EF_{NO_x} \cdot \rho_{NG} \cdot R \cdot T_{exh}}{P_{exh} \cdot MW_{NO_x}}$$

- $F_{NG}$: Natural gas flow rate (m³/h)
- $EF_{NO_x}$: Emission factor (g/m³)
- $T_{exh}$: Exhaust temperature (K)
- $P_{exh}$: Exhaust pressure (Pa)

## Technical Documentation

### Variable Mappings

| Code Variable | LaTeX Symbol | Physical Meaning | Units | Source |
|---------------|--------------|------------------|-------|--------|
| `P` | $P$ | Gas pressure | kPa | CSV: 'B-2 Gas Pressure' |
| `T` | $T$ | Temperature | °C | CSV: 'UBC Temp' |
| `F` | $\dot{V}$ | Volumetric flow | m³/h | CSV: 'B-2 Gas Flow Rate' |
| `y_CH4` | $y_{CH_4}$ | Methane fraction | - | Constant (0.95) |
| `y_C2H6` | $y_{C_2H_6}$ | Ethane fraction | - | Constant (0.05) |

### Unit Tests

```python
def test_molar_flow_rate():
    test_data = {
        ' B-2 Gas Pressure, kPa': 101.325,
        'UBC Temp, °C': 25,
        ' B-2 Gas Flow Rate, m³/h': 10.0
    }
    result = get_molar_flow_rate(test_data)
    assert abs(result - 409.03) < 0.1  # mol/h
```

### Assumptions and Limitations

1. **Ideal Gas Behavior**
   - Valid at low pressures and high temperatures
   - May deviate near critical point

2. **Combustion**
   - Complete combustion assumed (no CO/soot)
   - Fixed natural gas composition
   
3. **Heat Transfer**
   - Negligible heat loss to surroundings
   - Steady-state operation assumed

### Validation

All calculations include unit consistency checks using dimensional analysis. For example, the molar flow rate calculation verifies:

$$\frac{\text{kPa} \cdot \text{m}^3}{\text{J} \cdot \text{mol}^{-1} \cdot \text{K}^{-1} \cdot \text{K}} \cdot \frac{1000 \text{Pa}}{1 \text{kPa}} \cdot \frac{\text{J}}{\text{Pa} \cdot \text{m}^3} = \text{mol/s}$$

### Implementation Notes

- Temperature conversions: °C to K using $T(K) = T(°C) + 273.15$
- Pressure units: kPa converted to Pa for calculations
- All physical constants use SI units

## Results and Interpretation

### Key Findings
1. **Boiler Performance**:
   - Average efficiency: ~86.5%
   - Seasonal variations in efficiency and output
   - Strong correlation between ambient temperature and boiler load

2. **Emissions Analysis**:
   - NOx emissions typically range from 20-30 ppm
   - Good agreement between calculated and measured values
   - Identified periods of sensor malfunction

3. **Operational Patterns**:
   - Higher loads during winter months
   - Diurnal variations in demand
   - Maintenance periods clearly visible in the data

## Limitations and Assumptions

### Key Assumptions
1. **Natural Gas Composition**:
   - 95% methane, 5% ethane
   - No other hydrocarbons or impurities

2. **Combustion**:
   - Complete combustion (negligible CO)
   - Steady-state operation
   - Ideal gas behavior

3. **Heat Losses**:
   - Neglected radiation and convection losses in basic efficiency calculation
   - Assumed perfect insulation in some calculations

### Known Limitations
- Sensor inaccuracies and drift over time
- Simplified treatment of heat transfer mechanisms
- Fixed fuel composition assumption may not reflect actual variations

## References

1. Chintalapati, P. "CHBE 366 CEC Data Analytics Lab Assignment." University of British Columbia, 2024.
2. Smith, J.M., Van Ness, H.C., & Abbott, M.M. (2005). Introduction to Chemical Engineering Thermodynamics (7th ed.). McGraw-Hill.
3. Perry, R.H., Green, D.W., & Maloney, J.O. (1997). Perry's Chemical Engineers' Handbook (7th ed.). McGraw-Hill.
---

*This project was worked as part of the CHBE 366 curriculum at the University of British Columbia.*
