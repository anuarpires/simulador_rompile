import streamlit as st
import pandas as pd
import numpy as np
import math
from scipy.optimize import linprog
import plotly.graph_objects as go

# ==========================================
# 1. CONFIGURAÇÃO DA PÁGINA E ESTILO
# ==========================================
st.set_page_config(page_title="Otimizador de Pilha ROM - Copelmi", layout="wide")
st.title("Otimizador Avançado de Blending - Pilha ROM")
st.markdown("Motor de otimização por Programação Linear para maximizar o fechamento da pilha respeitando restrições.")

# ==========================================
# 2. FUNÇÕES MATEMÁTICAS E GEOMÉTRICAS
# ==========================================
def build_linear_problem(df, specs, target_mass, volume_max_m3):
    n = len(df)
    vm = df['vm'].to_numpy(dtype=float)
    ts = df['ts'].to_numpy(dtype=float)
    cinza = df['cinza'].to_numpy(dtype=float)
    rho = df['densidade'].to_numpy(dtype=float)
    
    bounds = [(0.0, float(row['ton_report'])) for _, row in df.iterrows()]
    
    A_ub, b_ub = [], []
    A_eq, b_eq = [], []
    
    A_eq.append(np.ones(n))
    b_eq.append(float(target_mass))
    
    if specs.get('ts_max') is not None:
        A_ub.append(ts - float(specs['ts_max']))
        b_ub.append(0.0)
    if specs.get('cinza_max') is not None:
        A_ub.append(cinza - float(specs['cinza_max']))
        b_ub.append(0.0)
    if specs.get('vm_min') is not None:
        A_ub.append(float(specs['vm_min']) - vm)
        b_ub.append(0.0)
        
    if volume_max_m3 is not None:
        A_ub.append(1.0 / rho)
        b_ub.append(float(volume_max_m3))
        
    def zscore(x):
        s = x.std()
        return np.zeros_like(x) if s == 0 else (x - x.mean()) / s
        
    c = (-1.0 * zscore(vm) + 1.0 * zscore(ts) + 1.0 * zscore(cinza))
    
    return c, A_ub, b_ub, A_eq, b_eq, bounds

def longitudinal_trapezoid_volume(
