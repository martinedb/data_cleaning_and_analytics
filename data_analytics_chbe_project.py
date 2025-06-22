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
# SECTION 7: CALCULATING TOTAL FLUE GAS AND AIR FLOW RATE
# ==========================================
# Changed

# Define the fluegascomp function
def fluegascomp(row):
    # Molecular weights of substances in Natural Gas and Flue Gas
    molweight_N2 = 28 * (1 / 1000)
    molweight_O2 = 32 * (1 / 1000)
    molweight_H2O = 18 * (1 / 1000)
    molweight_CO2 = 44 * (1 / 1000)
    molweight_C = 12 * (1 / 1000)
    molweight_CH4 = 16 * (1 / 1000)
    molweight_C2H6 = 30 * (1 / 1000)

    ## Dry Air composition
    yO2_dryair = 0.21
    yN2_dryair = 0.79

  # Combustion reaction constants for CH4 and C2H6
    # MR stands for mole ratio between two substances in the chemical reaction
    MR_O2byCH4 = 2
    MR_O2byC2H6 = 3.5

    #Natural gas composition constants
    yCH4_NG = 0.95
    yC2H6_NG = 0.05

    FuelFlow = row['Fuel Molar Flow Rate, mole/h']
    O2_exhaust = row[' B-2 Exhaust O2, %'] / 100

    # Natural gas mol flow rate
    NG_molflow = FuelFlow

#Find molar flow component rates
    O2moles = NG_molflow * (yCH4_NG * MR_O2byCH4 + yC2H6_NG * MR_O2byC2H6)
    molesair = O2moles / yO2_dryair

    #Compute number of moles of water in air
    mole_fraction_H2O_air = row['Mole Fraction of Water in Wet Air']
    moles_H2O_in_air = molesair * mole_fraction_H2O_air

    # Moisture (water vapour) mol flow rate due to combustion
    moles_H2O_from_combustion = NG_molflow * (yCH4_NG * 2 + yC2H6_NG * 3)

    A_dry = molesair * (yO2_dryair * molweight_O2 + (yN2_dryair) * molweight_N2)
    A_H2O_vap = moles_H2O_in_air * molweight_H2O
    A_mass = A_dry + A_H2O_vap  # kg/hr
    F = NG_molflow * (yCH4_NG * molweight_CH4 + yC2H6_NG * molweight_C2H6)
    G_mass = (A_mass + F) / (1 - O2_exhaust * (yO2_dryair * molweight_O2 + yN2_dryair * molweight_N2))

    # Calculation moisture in flue gas mass flow rate
    fluegas_H2Ovap = (A_H2O_vap + moles_H2O_from_combustion * molweight_H2O)
    G_dryair = G_mass - fluegas_H2Ovap
    molratioCtoCH4 = 1
    molratioCtoC2H6 = 2

    X_FC_mass = NG_molflow * molweight_C * (yCH4_NG * molratioCtoCH4 + yC2H6_NG * molratioCtoC2H6) / (F + 0.0001)

    if G_mass != 0:
        X_GC_mass = X_FC_mass * (F) / (G_dryair)
    else:
        X_GC_mass = 0
    massfracCO2 = row[' B-2 Exhaust CO2, %'] / 100

    if G_mass != 0:
        massfracH2O = fluegas_H2Ovap / (G_mass)
    else:
        massfracH2O = 0

    denominator = (row[' B-2 Exhaust O2, %'] / molweight_O2 + massfracCO2 / molweight_CO2 + massfracH2O / molweight_H2O +
                   (1 - row[' B-2 Exhaust O2, %'] / 100 - massfracCO2 - massfracH2O) / molweight_N2)
    if denominator != 0:
        molweight_G = 1 / denominator
    else:
        molweight_G = 0

    X_GC_mole = massfracCO2 / molweight_CO2 * molweight_G

    if G_mass != 0:
        X_GH_mole = fluegas_H2Ovap / (G_mass) / molweight_H2O * molweight_G
    else:
        X_GH_mole = 0

    X_GO_mole = O2_exhaust / molweight_O2 * molweight_G
    A_mole = A_mass * (yO2_dryair / molweight_O2 + (yN2_dryair) / molweight_N2)
    moles_N2_in_air = (A_mole) * yN2_dryair
    G_mole = G_mass / molweight_G
    G_H2O_mole = fluegas_H2Ovap / molweight_H2O

    if G_mass != 0:
        X_GN_mole = moles_N2_in_air / (G_mass) * molweight_G
    else:
        X_GN_mole = 0

    return G_mass, G_dryair, G_H2O_mole, A_mass, F, X_GC_mass, X_GC_mole, X_GN_mole, X_GH_mole, X_GO_mole, A_mole, G_mole

# Apply function
results = data.apply(fluegascomp, axis=1, result_type='expand')
# Column names and corresponding indices
columns = {
    'Total Flue Gas Flow Rate, G (kg/hr)': 0,
    'Moisture in Flue Gas, fluegas_H2Ovap (mole/hr)': 2,
    'Total Air Flow Rate, A (kg/hr)': 3,
    'Total Gas Flowrate, F (kg/hr)': 4,
    'Mole fraction of CO2 In G': 6,
    'Mole fraction of Nitrogen In G': 7,
    'Mole fraction of Water In G': 8,
    'Mole fraction of Oxygen In G': 9,
    'Total Air Flow Rate, A (mol/hr)': 10,
    'Total Flue Gas Flow Rate, G (mol/hr)': 11
}

# Assign results to corresponding columns using a loop
for column_name, index in columns.items():
    data[column_name] = results[index]

data.head()

# ==========================================
# SECTION 8: DEFINING HEAT CAPACITY COEFFICIENTS AND TEMPERATURE CONVERSIONS
# ==========================================

# Heat Capacity Coefficients From Thermo Textbook
cp_coefffs = {
    'methane': {'A': 19.25, 'B': 0.05213, 'C': 1.197e-5, 'D': -1.132e-8, 'phase': 'gas'},
    'carbon_dioxide': {'A': 19.8, 'B': 0.07344, 'C': -5.602e-5, 'D': 1.715e-8, 'phase': 'gas'},
    'carbon_monoxide': {'A': 30.87, 'B': -0.01285, 'C':2.789e-5, 'D': -1.272e-8, 'phase': 'gas'},
    'ethane': {'A': 5.409, 'B': 0.1781, 'C':-6.938e-5, 'D': 8.713e-9, 'phase': 'gas'},
    'nitrogen': {'A': 31.15, 'B': -0.01357, 'C': 2.680e-5, 'D': -1.168e-8, 'phase': 'gas'},
    'oxygen': {'A': 28.11, 'B': -3.7e-6, 'C': 1.746e-5, 'D': -1.065e-8, 'phase': 'gas'},
    'water': {'A': 7.243E+1, 'B': 1.039e-2, 'C': -1.497e-6, 'D': 0, 'phase': 'liquid'},
    'water_gas': {'A': 32.24, 'B': 0.001924, 'C': 1.055e-5, 'D': -3.569e-9, 'phase': 'gas'},
}

def heat_capacity(substance, temp): #in J/mol-K
    # If statement to verify chemical is defined in dictionary
    if substance.lower() in cp_coefffs:
        regcoefficient = cp_coefffs[substance.lower()]
        A = regcoefficient['A']
        B = regcoefficient['B']
        C = regcoefficient['C']
        D = regcoefficient['D']
        HeatCapacity = A + B * temp + C * temp**2 + D * temp**3
        return HeatCapacity

def calc_heatcap(components, temp):
    heatcap_values = {}
    for substance in components:
        heatcap_values[substance] = heat_capacity(substance, temp)
    return heatcap_values

ingascomp = ['water_gas', 'methane', 'ethane', 'nitrogen', 'oxygen']
outgascomp = ['water_gas', 'carbon_dioxide', 'carbon_monoxide', 'nitrogen', 'oxygen']
inwatercomp=['water']
outwatercomp=['water']

#Temperature Values and Conversions
tempinC = data['UBC Temp, °C']
tempoutC = data[' B-2 Exhaust Temp, °C']
tempKgas_in = tempinC + 273.15
tempoutKgas = tempoutC + 273.15
temp_water_in=data[' B-2 Entering Water Temp, °C']
temp_water_out=data[' B-2 Leaving Water Temp, °C']
temp_water_in_K=temp_water_in+273.25
temp_water_out_K=temp_water_out+273.15

# Define a dictionary to store substance categories and corresponding temperature values
substance_categories = {
    "inwatercomp": (temp_water_in_K, "for the water coming in", "inH2O_cp"),
    "outwatercomp": (temp_water_out_K, "for the water coming out", "outH2O_cp"),
    "ingascomp": (tempKgas_in, "in the inlet", "Cpin"),
    "outgascomp": (tempoutKgas, "in the outlet", "Cpout")
}

# Iterate over substance categories
for category, (temperature, description, cp_name) in substance_categories.items():
    # Iterate over substances in the current category
    for substance in globals()[category]:
        # Calculate heat capacity
        cp = heat_capacity(substance, temperature)
        # Update data dictionary with heat capacity information using the respective cp_name
        data[f"Heat capacity of {substance} {description} J/mol-K"] = cp

data.head()

# ==========================================
# SECTION 9: HEAT CAPACITY CALCULATIONS OF INLET AND OUTLET COMPONENTS
# ==========================================
yH2Owetair=data['Mole Fraction of Water in Wet Air']
molfrac_dryAir=1-yH2Owetair

#Cp of components of Air
HeatCapN2in=data['Heat capacity of nitrogen in the inlet J/mol-K']
HeatCapO2in=data['Heat capacity of oxygen in the inlet J/mol-K']
HeatCapWatervapin=data['Heat capacity of water_gas in the inlet J/mol-K']
heatcap_DryAir=((0.79*HeatCapN2in)+(0.21*HeatCapO2in))
heatcap_WetAir=HeatCapWatervapin

#Heat Capacities of Inlet Methane and Ethane
##Inlet and Outlet Water Heat Capacities

heatcapCH4in=data['Heat capacity of methane in the inlet J/mol-K']
heatcapethane_in=data['Heat capacity of ethane in the inlet J/mol-K']
heatcap_fuel=(0.05*heatcapethane_in)+(0.95*heatcapCH4in)
heatcapliqH2O=data['Heat capacity of water for the water coming in J/mol-K']
heatcapliqH2O_out=data['Heat capacity of water for the water coming out J/mol-K']

#Insert data into dataset
data['Cp of Dry Air (J/mol*K)']=heatcap_DryAir
data['Cp of Wet Air (J/mol*K)']=heatcap_WetAir
data['Heat Capacity of Fuel in (J/mol-K)']=heatcap_fuel
data['Heat Capacity of Water Flowing in (J/mol-K)']=heatcapliqH2O
data['Heat Capacity of Water Flowing out (J/mol-K)']=heatcapliqH2O_out

data.head()

# ==========================================
# SECTION 10: HEAT CAPACITIES OF FLUE GAS COMPONENTS FOR DRY AND WET FLUE GAS
# ==========================================
#Heat Capacities for Flue Gas Components
heatcapH2O_vapout=data['Heat capacity of water_gas in the outlet J/mol-K']
heatcapCO2out=data['Heat capacity of carbon_dioxide in the outlet J/mol-K']
heatcapCOout=data['Heat capacity of carbon_monoxide in the outlet J/mol-K']
heatcapN2out=data['Heat capacity of nitrogen in the outlet J/mol-K']
heatcapO2out=data['Heat capacity of oxygen in the outlet J/mol-K']

#Flue Gas Mole Fractions
molfracN2_G=data['Mole fraction of Nitrogen In G']
molfracCO2_G=data['Mole fraction of CO2 In G']
molfracwater_G=data['Mole fraction of Water In G']
molfracoxygen_G=data['Mole fraction of Oxygen In G']

ydryN2 = (molfracN2_G) /(1 - molfracwater_G)
ydryCO2 = (molfracCO2_G) /(1 -molfracwater_G)
ydryO2 = (molfracoxygen_G) / (1- molfracwater_G)

#Heat Capacities of Dry Flue Gas
Cp_DFG=(ydryCO2*heatcapCO2out)+(ydryN2*heatcapN2out)+(ydryO2*heatcapO2out)
Cp_WFG=heatcapH2O_vapout

#Print out data
data['Heat Capacity of Dry Flue Gas (J/mol-K)']=Cp_DFG
data['Heat Capacity of Wet Flue Gas (J/mol-K)']=Cp_WFG

data.head()

# ==========================================
# SECTION 11: HEAT FLOW CALCULATIONS FOR DRY AND WET FLUE GAS 
# ==========================================
To= data[' B-2 Leaving Water Temp Setpoint, °C'] + 273.15
tempkelvin = data['UBC Temp, °C'] +273.15

#Define constants
enthalpy_CH4=890.15 #kJ/mol
enthalpy_C2H6=1560.7 #kJ/mol
Patm=1 #atm
R=8.205745e-2 #L*atm/(K*mol)
heatofvapH2O=45.054 #kJ/mol

#Unit Conversions for Temp
tempKgas_in = tempinC + 273.15
tempoutKgas = tempoutC + 273.15
H2Oflowrate=data[' B-2 Water Flow Rate, L/s']

#mol flow in & Heat of Water in
H2O_mol_in=((Patm*H2Oflowrate)/(temp_water_in_K*R))*(3600) #L/s to mol/hr
H2O_mol_out=((Patm*H2Oflowrate)/(temp_water_out_K*R))*(3600)
heatflow_water_in=(H2O_mol_in*heatcapliqH2O*(temp_water_in_K-To))
data['Q Water in (J/mol)']=heatflow_water_in
molarflow_fuel = data['Fuel Molar Flow Rate, mole/h']

# Oxygen Flow Calculations
CH4inlet = 0.95 * molarflow_fuel #Constant
C2H6inlet = 0.05 * molarflow_fuel #Constant
O2toCH4 = CH4inlet
O2toC2H6 = 3.5 * C2H6inlet
oxygeninlet = O2toCH4 + O2toC2H6
nitrogeninlet = (0.79 / 0.21) * oxygeninlet  # Nitrogen Flow Calculation

# Dry and Wet Air Flow Calculations
MoleDryAir_IN = nitrogeninlet + oxygeninlet
MolFrac_WetAirH2O = data['Mole Fraction of Water in Wet Air']
mol_frac_Dry_Air = 1 - MolFrac_WetAirH2O
MoleWetAir_IN = (MolFrac_WetAirH2O / mol_frac_Dry_Air) * MoleDryAir_IN
TOTALAir_in = MoleDryAir_IN + MoleWetAir_IN


heatflowfuel_inlet = (molarflow_fuel * heatcap_fuel * (tempkelvin - To) + ((0.05 * enthalpy_C2H6) + (0.95 * enthalpy_CH4)))
Heat_DryAir_in = MoleDryAir_IN * heatcap_DryAir * (tempkelvin - To)
Heat_WetAir_in = MoleWetAir_IN * ((heatcap_WetAir * (tempkelvin - To)) + heatofvapH2O)

# Flue Gas Out Calculation
FG_outmole = TOTALAir_in + molarflow_fuel
heatflow_water_out = (H2O_mol_out * heatcapliqH2O_out * (temp_water_out_K - To))

#Define data to insert into dataset
data['Heat Flow of Fuel(J/mol)'] = heatflowfuel_inlet
data['Heat of Dry Air (J/mol)'] = Heat_DryAir_in
data['Heat of Wet Air (J/mol)'] = Heat_WetAir_in
data['Heat of Water Out (J/mol)'] = heatflow_water_out

# Heat Calculation for Dry and Wet Flue Gas Out
#WFG = wet flue gas
#DFG = dry flue gas
WFG_frac = data['Mole fraction of Water In G']
DFG_frac = 1 - WFG_frac
heat_DFG_out = (Cp_DFG * (FG_outmole * DFG_frac) * (tempoutKgas - To))
heat_WFG_out = (FG_outmole * WFG_frac) * ((Cp_WFG * (tempoutKgas - To)) + heatofvapH2O)

data['Heat from Dry Flue Gas (J/mol)'] = heat_DFG_out
data['Heat from Wet Flue Gas (J/mol)'] = heat_WFG_out

data.head()  # Displaying the first few rows of the DataFrame

# ==========================================
# SECTION 12: HEAT LOSS CALCULATION BASED ON ENERGY BALANCE 
# ==========================================
#Energy Balances Based on the Assignment Document
HeatFlow_out=heatflow_water_out+heat_DFG_out+heat_WFG_out
HeatFlow_in=heatflow_water_in+heatflowfuel_inlet+Heat_DryAir_in+Heat_WetAir_in
HeatLoss=HeatFlow_in-HeatFlow_out
FG_flowrate = FG_outmole

data['Energy Lost (J/hr)']=HeatLoss
data.head()


# ==========================================
# SECTION 13: FLUE GAS COMPOSITION AND FLUE GAS FLOW RATE OVER TIME
# ==========================================
# Flue Gas Composition
N2molfracFG = data['Mole fraction of Nitrogen In G']
CO2molfracFG = data['Mole fraction of CO2 In G']
watermolfracFG = data['Mole fraction of Water In G']
O2molfracFG = data['Mole fraction of Oxygen In G']

#Time series for data
time = np.arange(0, 8628, 1)

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

# Graph of flue gas flow rate over time
plt.figure(figsize=(10, 6))
plt.plot(time, FG_flowrate, label='Flue Gas Flow Rate (mol/hr)')
plt.xlabel('Time (hr)')
plt.ylabel('Flue Gas Flow Rate (mol/hr)')
plt.legend()
plt.grid(False)
plt.show()

# ==========================================
# SECTION 14: HOT WATER, COLD WATER, AND OVERALL ENERGY TRANSFER RATE OVER TIME
# ==========================================
hwrate = np.abs(heatflow_water_out)
cwrate = np.abs(heatflow_water_in)
hloss = HeatLoss

# Plot the hot water energy transfer rate
plt.figure(figsize=(10, 6))
plt.plot(time, hwrate, label='Hot Water Transfer Rate (J/hr)')
plt.xlabel('Time (hr)')
plt.ylabel('Hot Water Energy Transfer Rate (J/hr)')
plt.legend()
plt.grid(False)
plt.show()

# Plot the cold water energy transfer rate
plt.figure(figsize=(10, 6))
plt.plot(time, cwrate, label='Cold Water Transfer Rate (J/hr)')
plt.xlabel('Time (hr)')
plt.ylabel('Cold Water Energy Transfer Rate (J/hr)')
plt.legend()
plt.grid(False)
plt.show()

# Plotting the overall energy transfer rate
plt.figure(figsize=(10, 6))
plt.plot(time, hloss, label='Net Heat Transfer (J/hr)')
plt.xlabel('Time(hr)')
plt.ylabel('Overall Energy Transfer Rate (J/hr)')
plt.legend()
plt.grid(False)
plt.show()

# ==========================================
# SECTION 15: THERMAL EFFICIENCY CALCULATIONS
# ==========================================
## Efficiency Calculations
enthalpy_CH4 = -890.15  # kJ/mol
enthalpy_C2H6 = -1560.7  # kJ/mol
molweight_H2O = 0.018015 #kg/mol
cw_density = 1 #1 kilogram per liter

# Enthalpy of Combustion
deltaHc =(enthalpy_C2H6 * 0.05) + (enthalpy_CH4 * 0.95)

# Extracting enthalpy data
HFhw = data['Q Water in (J/mol)'] / 1000  # Energy out
HFcw = data['Heat of Water Out (J/mol)'] / 1000  # Energy in

# Calculate Water and Fuel Flow Rates
Wflow = (data[' B-2 Water Flow Rate, L/s'] * cw_density) / molweight_H2O
Frate = data['Fuel Molar Flow Rate, mole/h'] /3600

#Calculate Boiler Efficiency #eta is greek letter for efficiency
eta = (Wflow * np.abs((HFhw - HFcw) / (Frate * deltaHc))) * 100

# Updating data with efficiency values
data['Efficiency of Boiler over time'] = np.clip(eta, 0, 100)
data['Efficiency(%)'] = np.clip(eta, 0, 100)

# Plotting
effskyspark = data[' B-2 Efficiency, %']

plt.figure(figsize=(10, 6))
plt.plot(time, data['Efficiency(%)'], label='Calculated Efficiency (%)')
plt.plot(time, effskyspark, label='Thermal Efficiency From Skyspark (%)')
plt.xlabel('Time(hr)')
plt.ylabel('Thermal Efficiency (%)')
plt.legend()
plt.grid(False)
plt.show()

data.head()

# ==========================================
# SECTION 16: NOX EMISSIONS CONCENTRATION ESTIMATES
# ==========================================
# NOx Emissions Calculations
# Using the IG Law: PV = nRT V: V = nRT/P

# Define constants
NOx_emission_fact = 50  # lb/million.ft3
lb_to_g_conversion = 453.59 / 1  # g/lb (Pounds to Grams Conversion)
ft3_m3_conversion = 0.028317 / 1  # m3/ft3 (Cubic Meters to Cubic Feet Conversion)
factor = NOx_emission_fact * lb_to_g_conversion / ft3_m3_conversion  # g/million.m3 = (g/m3)/million
R = 8.314  # J/(mol*K) (Gas Constant)
conv_for_NOx_ppmv = 1.88e-3   # (g/m3)/ppmv

# Function for NOx emission conversion
def NOx_emission_conversion(row):
    exhaust_temp = row[' B-2 Exhaust Temp, °C'] + 273.15  # K
    NG_FR = row[' B-2 Gas Flow Rate, m³/h']  # m3/h
    AT_pressure = row[' B-2 Gas Pressure, kPa']
    volume = row['Total Flue Gas Flow Rate, G (mol/hr)'] * R * exhaust_temp / (AT_pressure * 1000)  # m3/h
    NOx_mass = factor * (NG_FR) / (1e+6)  # (g/m3)/1e+6 m3/h = g/h
    convert = NOx_mass / conv_for_NOx_ppmv  # ppmv.m3/h

    if volume == 0:
        return 0
    else:
        return (convert / volume)  # ppmv

data['NOx emission, ppmv'] = data.apply(NOx_emission_conversion, axis=1)
NOx_ppm_sensor = data[' B-2 Exhaust NOx, ppm']
NOx_emission_concentration = data['NOx Calculated (ppm)'] = data.apply(NOx_emission_conversion, axis=1)

# Plotting the estimated NOx concentration and the real-time sensor data
plt.figure(figsize=(10, 6))
plt.plot(time, NOx_ppm_sensor, label='NOx Sensor (ppm)')
plt.plot(time, NOx_emission_concentration, label='NOx Estimate (ppm)')
plt.xlabel('Time (hr)')
plt.ylabel('NOx Emission Concentration (ppm)')
plt.legend()
plt.grid(False)
plt.show()

data.head()



# ==========================================
# SECTION 17: SAVE THE CLEANED DATASET
# =========================================
# Save new data set for boiler 2
data.to_csv('complete_dataset_analytics.csv', index=True)









