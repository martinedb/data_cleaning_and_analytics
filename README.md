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

The main script `data_analytics_chbe_project.py` is organized into logical sections:

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

## Theoretical Framework

### Mass Balance
- **Ideal Gas Law**: Used to relate pressure, volume, temperature, and moles of gas
- **Stoichiometry**: Balances chemical equations for complete combustion
- **Humidity Calculations**: Accounts for water vapor in combustion air

### Energy Balance
- **First Law of Thermodynamics**: Energy conservation in the boiler system
- **Heat Capacity**: Temperature-dependent heat capacities for all components
- **Enthalpy Calculations**: For water/steam and combustion products

### Efficiency Calculation
- **Thermal Efficiency**: Ratio of energy absorbed by water to energy input from fuel
- **Emission Factors**: Standard factors for NOx emissions from natural gas combustion

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
4. UBC Campus Energy Centre Operational Manual, 2021.

---

*This project was developed as part of CHBE 366 at the University of British Columbia.*
