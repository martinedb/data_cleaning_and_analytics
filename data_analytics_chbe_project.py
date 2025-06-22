## Run each of these sections of code one at a time. Running them all at once can lead to a slower script loading.

# ==========================================
# SECTION 1: IMPORTING PACKAGES
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import sympy as sp
import math

data = pd.read_csv("B2 Cleaned (from pranav).csv")
data.head()


# ==========================================
# SECTION 2: PRINTING DATA COLUMNS TO VERIFY COLUMN NAMES
# ==========================================
print(data.columns)


# ==========================================
# SECTION 3: MOLAR FLOW RATE CALCULATIONS
# ==========================================
### Determine the molar flow rate of fuel at operating conditions. ###
# Assumption:
#Natural gas is at ambient air temperature
#Natural gas uses ideal gas law
#PV = nRT, N = PV/RT

#Parameters
# P = Pressure in kPa
# T = temperature in deg C
# F = flow rate in m^3/h
# R = gas constant in J/mol*K

#Defining constants
# Constants
molweight_air = 28.97  # Molecular weight of air (g/mol)
molweight_fuel = 16.04  # Molecular weight of natural gas (g/mol)

# Fuel/Natural Gas composition (mole fractions)
y_CH4 = 0.95 #Mol
y_C2H6 = 0.05

# Air composition (mole fractions)
yO2 = 0.21 #Molar fraction of oxygen in dry air
yN2 = 0.79 #Molar fraction of nitrogen in dry air

def get_molar_flow_rate(row):
  P = row[' B-2 Gas Pressure, kPa'] # + 101.325 # kPa, absolute pressure ???
  T = row['UBC Temp, °C']
  F = row[' B-2 Gas Flow Rate, m³/h']
  R = 8.314 #J/mol*K
  return P*1000*F / (R * (T +273.15))


# Apply the function to each row of the dataframe
data['Fuel Molar Flow Rate, mole/h'] = data.apply(get_molar_flow_rate,  axis=1)
data.head()


# ==========================================
# SECTION 4: ADDING SATURATION PRESSURE OF WATER AND PARTIAL PRESSURE WATER DATA COLUMNS
# ==========================================

#Create column for saturation pressure of water based on Antoine's Equation
#Create column for partial pressure of water based on relative humidity formula

#For a temperature range of 0 degrees Celsius to 30 deg C

def sat_pressureofwater(row):
  A_water = 5.40221
  B_water = 1838.675
  C_water = -31.737
  T = row['UBC Temp, °C'] #Add 273.15 to convert to kelvin
  return 10**(A_water - (B_water/(T + 273.15 + C_water))) * 100

data['Saturation Pressure of Water (kPA)'] = data.apply(sat_pressureofwater,  axis=1)
data.head()
#1 bar = 100 kPA

#Generate column for partial pressure of water based on saturation pressure times relative humidity using columns in a certain dataset
#Assume total pressure is atmospheric pressure and include the assumptions into your report.

data['Partial Pressure of Water (kPA)'] =(data['Saturation Pressure of Water (kPA)']*data['UBC Humidity, %RH'])/(100)
data.head()

# ==========================================
# SECTION 5: FINDING MOLAR FRACTION OF WATER IN WET AIR USING RAOULT'S LAW
# ==========================================
#Find the molar fraction of water in wet air using Raoult's Law
#Use constant atmospheric pressure

#Formula: mol frac of water = (partial pressure)/(atmospheric pressure)
def mol_fracwater(row):
  partial_p = row['Partial Pressure of Water (kPA)']
  atmospheric_p = 101.325 #kPa
  return (partial_p/atmospheric_p)

data['Mole Fraction of Water in Wet Air'] = data.apply(mol_fracwater,  axis=1)
mol_frach2o = data['Mole Fraction of Water in Wet Air']
data.head()

# ==========================================
# SECTION 6: FINDING MOLAR FRACTION OF OXYGEN AND NITROGEN IN WET AIR USING RAOULT'S LAW
# ==========================================

# Molar fractions calculations for all components in wet air
# Assuming you already have the molar fraction of water in wet air calculated
yH2O_wetair = data['Mole Fraction of Water in Wet Air']  # Assuming you already have this calculated
molfrac_n2dryair = yN2 = 0.79 # Nitrogen and oxygen mole fractions in dry air
molfrac_o2dryair = yO2 = 0.21
y1 = yO2 + yN2 + yH2O_wetair #Overall molar fraction (wet air + dry air composition)

def yO2_wetair(row):
    molfrac_o2dryair = yO2
    yH2O_wetair = row['Mole Fraction of Water in Wet Air']
    return molfrac_o2dryair / (1 + yH2O_wetair)

def y_N2wetair(row):
    molfrac_n2dryair = yN2
    yH2O_wetair = row['Mole Fraction of Water in Wet Air']
    return molfrac_n2dryair / (1 + yH2O_wetair)

# Verify the sum of molar fractions of nitrogen and oxygen equals 1 minus the molar fraction of water in wet air
    yN2_wetaircalc + yO2_wetaircalc == 1 - yH2O_wetair

# Add new columns for the nitrogen and oxygen mole fractions in wet air
data['Molar Fraction of Nitrogen in Wet Air'] = data.apply(y_N2wetair,  axis=1)
data['Molar Fraction of Oxygen in Wet Air'] = data.apply(yO2_wetair,  axis=1)

# Display the modified DataFrame
data.head()


# ==========================================
# SECTION 7: FINDING MOLAR FRACTION OF OXYGEN AND NITROGEN IN WET AIR USING RAOULT'S LAW
# ==========================================
