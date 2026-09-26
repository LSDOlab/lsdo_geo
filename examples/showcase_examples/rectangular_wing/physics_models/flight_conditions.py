"""Tropospheric ISA atmosphere helper and flight conditions for BWB optimization.

Computes density, speed of sound, temperature, and dynamic viscosity from ISA
altitude in SI units without Python conditionals on CSDL variables.

Standard constants (ISA):
    T0 = 288.15 K
    p0 = 101325.0 Pa
    L  = 0.0065 K/m
    g0 = 9.80665 m/s^2
    R  = 287.05287 J/(kg K)
    gamma = 1.4

Dynamic viscosity (Sutherland's law diagnostic):
    mu_0 = 1.716e-5 Pa*s
    T_s  = 273.15 K
    S    = 110.4 K
    mu(T) = mu_0 * (T / T_s)**(3/2) * (T_s + S) / (T + S)
"""

from __future__ import annotations
from typing import Union
import numpy as np
import csdl_alpha as csdl

# Physical constants
T0 = 288.15        # Sea-level standard temperature [K]
P0 = 101325.0      # Sea-level standard pressure [Pa]
L = 0.0065         # Troposphere temperature lapse rate [K/m]
G0 = 9.80665       # Standard gravitational acceleration [m/s^2]
R_GAS = 287.05287  # Specific gas constant for dry air [J/(kg K)]
GAMMA = 1.4        # Ratio of specific heats for dry air

# Sutherland's law constants for air
MU_0 = 1.716e-5    # Reference viscosity [Pa*s]
T_S = 273.15       # Reference temperature [K]
S_SUTHERLAND = 110.4  # Sutherland temperature constant [K]

# Mission & sizing condition specifications
CRUISE_ALTITUDE_M = 30000.0 * 0.3048  # 9,144 m (30,000 ft)
CRUISE_MACH = 0.70
SIZING_EQUIVALENT_SPEED_FACTOR = 1.25


def compute_isa_troposphere(altitude_m: Union[csdl.Variable, float, np.ndarray]):
    """Compute ISA troposphere atmospheric properties in SI units.
    
    Valid for altitudes h <= 11,000 m (troposphere).
    Does not use Python conditionals on the input.

    Parameters
    ----------
    altitude_m : csdl.Variable, float, or np.ndarray
        Geometric altitude above mean sea level [m].

    Returns
    -------
    dict
        Dictionary containing CSDL variables or floats:
        - 'temperature' [K]
        - 'pressure' [Pa]
        - 'density' [kg/m^3]
        - 'speed_of_sound' [m/s]
        - 'viscosity' [Pa*s]
    """
    exponent = G0 / (R_GAS * L)
    
    if isinstance(altitude_m, (int, float, np.ndarray)):
        temp = T0 - L * altitude_m
        pressure = P0 * (temp / T0)**exponent
        density = pressure / (R_GAS * temp)
        sos = np.sqrt(GAMMA * R_GAS * temp)
        viscosity = MU_0 * (temp / T_S)**1.5 * (T_S + S_SUTHERLAND) / (temp + S_SUTHERLAND)
        return {
            'temperature': temp,
            'pressure': pressure,
            'density': density,
            'speed_of_sound': sos,
            'viscosity': viscosity,
        }
    
    temp = T0 - L * altitude_m
    pressure = P0 * ((temp / T0)**exponent)
    density = pressure / (R_GAS * temp)
    sos = csdl.sqrt(GAMMA * R_GAS * temp)
    viscosity = MU_0 * ((temp / T_S)**1.5) * (T_S + S_SUTHERLAND) / (temp + S_SUTHERLAND)
    return {
        'temperature': temp,
        'pressure': pressure,
        'density': density,
        'speed_of_sound': sos,
        'viscosity': viscosity,
    }


def get_nominal_flight_conditions():
    """Return exact nominal condition values for cruise, sizing, and node table."""
    cruise_atm = compute_isa_troposphere(CRUISE_ALTITUDE_M)
    sea_level_atm = compute_isa_troposphere(0.0)

    v_cruise = CRUISE_MACH * float(cruise_atm['speed_of_sound'])
    q_cruise = 0.5 * float(cruise_atm['density']) * (v_cruise**2)
    q_sizing = (SIZING_EQUIVALENT_SPEED_FACTOR**2) * q_cruise
    v_sizing = np.sqrt(2.0 * q_sizing / float(sea_level_atm['density']))
    m_sizing = v_sizing / float(sea_level_atm['speed_of_sound'])

    return {
        'cruise': {
            'altitude_m': CRUISE_ALTITUDE_M,
            'temperature_K': float(cruise_atm['temperature']),
            'pressure_Pa': float(cruise_atm['pressure']),
            'density_kg_m3': float(cruise_atm['density']),
            'speed_of_sound_m_s': float(cruise_atm['speed_of_sound']),
            'viscosity_Pa_s': float(cruise_atm['viscosity']),
            'speed_m_s': v_cruise,
            'mach': CRUISE_MACH,
            'dynamic_pressure_Pa': q_cruise,
        },
        'sizing': {
            'altitude_m': 0.0,
            'temperature_K': float(sea_level_atm['temperature']),
            'pressure_Pa': float(sea_level_atm['pressure']),
            'density_kg_m3': float(sea_level_atm['density']),
            'speed_of_sound_m_s': float(sea_level_atm['speed_of_sound']),
            'viscosity_Pa_s': float(sea_level_atm['viscosity']),
            'speed_m_s': v_sizing,
            'mach': m_sizing,
            'dynamic_pressure_Pa': q_sizing,
            'equivalent_speed_factor': SIZING_EQUIVALENT_SPEED_FACTOR,
        },
    }

