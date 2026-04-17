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

def longitudinal_trapezoid_volume(comp, larg, alt_max, angulo):
    tanv = math.tan(math.radians(angulo))
    retracao_lateral = alt_max / tanv
    largura_topo = max(0.0, larg - 2.0 * retracao_lateral)
    area_secao = alt_max * (larg + largura_topo) / 2.0
    return comp * area_secao

# ==========================================
# 3. INTERFACE DE USUÁRIO (Sidebar)
# ==========================================
st.sidebar.header("Parâmetros da Pilha")
alvo_massa = st.sidebar.number_input("Massa Alvo (t)", value=50000.0, step=1000.0)

st.sidebar.header("Geometria")
comp_base = st.sidebar.number_input("Comprimento Base (m)", value=120.0)
larg_base = st.sidebar.number_input("Largura Base (m)", value=70.0)
alt_max = st.sidebar.number_input("Altura Máxima (m)", value=5.0)
angulo_rep = st.sidebar.number_input("Ângulo Repouso (Graus)", value=37.0)

st.sidebar.header("Restrições da Usina")
vm_min = st.sidebar.number_input("VM Mínimo (%)", value=19.30)
ts_max = st.sidebar.number_input("TS Máximo (%)", value=2.20)
cinza_max = st.sidebar.number_input("Cinza/CBS Máximo", value=57.17)

# ==========================================
# 4. TABELA DE DADOS INTERATIVA
# ==========================================
st.subheader("Inventário de Frentes de Lavra (Editável)")
dados_iniciais = pd.DataFrame({
    "camada": ["S6", "S5", "S4", "S3", "S2", "CS", "CI"],
    "ton_report": [0.0, 4138.0, 7926.0, 16039.0, 26914.0, 36211.0, 55195.0],
    "vm": [20.00, 21.11, 19.10, 21.07, 22.37, 19.23, 17.81],
    "ts": [1.50, 0.70, 0.94, 2.09, 1.44, 1.43, 1.14],
    "cinza": [50.00, 51.31, 58.13, 47.63, 46.47, 51.86, 59.73],
    "densidade": [1.60, 1.70, 1.80, 1.60, 1.70, 1.70, 1.70]
})

df_editado = st.data_editor(dados_iniciais, num_rows="dynamic", use_container_width=True)
df_valido = df_editado[df_editado['ton_report'] > 0].copy()

# ==========================================
# 5. BOTÃO DE OTIMIZAÇÃO
# ==========================================
if st.button("Rodar Solver de Otimização", type="primary"):
    if df_valido.empty:
        st.error("Nenhuma camada com tonelagem válida para otimizar.")
    else:
        try:
            vol_max = longitudinal_trapezoid_volume(comp_base, larg_base, alt_max, angulo_rep)
            specs = {'vm_min': vm_min, 'ts_max': ts_max, 'cinza_max': cinza_max}
            c, A_ub, b_ub, A_eq, b_eq, bounds = build_linear_problem(df_valido, specs, alvo_massa, vol_max)
            
            res = linprog(c, A_ub=A_ub if A_ub else None, b_ub=b_ub if b_ub else None, 
                          A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            
            if not res.success:
                st.error(f"O Solver não encontrou uma solução viável matemática para essas restrições: {res.message}")
            else:
                st.success("Solução Otimizada Encontrada!")
                
                df_valido['ton_calculada'] = res.x
                df_res = df_valido[df_valido['ton_calculada'] > 1e-6].copy()
                
                massa_final = df_res['ton_calculada'].sum()
                vm_final = np.average(df_res['vm'], weights=df_res['ton_calculada'])
                ts_final = np.average(df_res['ts'], weights=df_res['ton_calculada'])
                cinza_final = np.average(df_res['cinza'], weights=df_res['ton_calculada'])
                
                area_base = comp_base * larg_base
                df_res['volume_m3'] = df_res['ton_calculada'] / df_res['densidade']
                df_res['espessura_m'] = df_res['volume_m3'] / area_base
                
                st.divider()
                st.subheader("Resultados de Qualidade")
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Massa Total", f"{massa_final:,.0f} t", delta=f"{massa_final - alvo_massa:,.0f} vs Alvo")
                cor_vm = "normal" if vm_final >= vm_min else "inverse"
                c2.metric("VM Final", f"{vm_final:.2f}%", delta=f"Mínimo: {vm_min}", delta_color=cor_vm)
                cor_ts = "normal" if ts_final <= ts_max else "inverse"
                c3.metric("TS Final", f"{ts_final:.2f}%", delta=f"Máximo: {ts_max}", delta_color=cor_ts)
                cor_cz = "normal" if cinza_final <= cinza_max else "inverse"
                c4.metric("Cinza/CBS Final", f"{cinza_final:.2f}", delta=f"Máximo: {cinza_max}", delta_color=cor_cz)
                
                st.divider()
                st.subheader("Composição e Geometria da Pilha")
                
                col_tabela, col_grafico = st.columns([1, 1.2])
                
                # --- TABELA DE ENGENHARIA ---
                with col_
