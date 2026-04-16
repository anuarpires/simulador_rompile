import streamlit as st
import pandas as pd

st.set_page_config(page_title="Simulador de Blending ROM", layout="wide")
st.title("⛏️ Painel de Controle de Blending - Pilha ROM (50.000t)")
st.markdown("Ajuste as frentes de lavra disponíveis para fechar a qualidade da pilha.")

# Dados Fixos
massa_S5, vm_S5, ts_S5, cbs_S5, den_S5 = 4138, 21.11, 0.70, 51.31, 1.70
massa_S4, vm_S4, ts_S4, cbs_S4, den_S4 = 7926, 19.10, 0.94, 58.13, 1.80
massa_S3, vm_S3, ts_S3, cbs_S3, den_S3 = 16039, 21.07, 2.09, 47.63, 1.60
massa_topo = massa_S5 + massa_S4 + massa_S3

# Controles Laterais
st.sidebar.header("Ajuste de Frentes (Fundo)")
corte_S2 = st.sidebar.slider("Tonelagem S2", 0, 26914, 2520)
corte_CS = st.sidebar.slider("Tonelagem CS", 0, 36211, 10362)
corte_CI = st.sidebar.slider("Tonelagem CI", 0, 55195, 9015)

# Cálculos
ton_total = massa_topo + corte_S2 + corte_CS + corte_CI

if ton_total > 0:
    vm_final = ((massa_S5*vm_S5) + (massa_S4*vm_S4) + (massa_S3*vm_S3) + (corte_S2*22.37) + (corte_CS*19.23) + (corte_CI*17.81)) / ton_total
    ts_final = ((massa_S5*ts_S5) + (massa_S4*ts_S4) + (massa_S3*ts_S3) + (corte_S2*1.44) + (corte_CS*1.43) + (corte_CI*1.14)) / ton_total
    cbs_final = ((massa_S5*cbs_S5) + (massa_S4*cbs_S4) + (massa_S3*cbs_S3) + (corte_S2*46.47) + (corte_CS*51.86) + (corte_CI*59.73)) / ton_total
    altura_pilha = (((massa_S5/den_S5) + (massa_S4/den_S4) + (massa_S3/den_S3)) + ((corte_S2+corte_CS+corte_CI)/1.7)) / 8400

# Resultados
st.markdown("### Resultados do Blending")
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Massa Total", f"{ton_total:,.0f} t")
col2.metric("VM Final", f"{vm_final:.2f}")
col3.metric("TS Final", f"{ts_final:.2f}")
col4.metric("CBS Final", f"{cbs_final:.2f}")
col5.metric("Altura", f"{altura_pilha:.2f} m")