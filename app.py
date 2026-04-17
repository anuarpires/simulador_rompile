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
st.title("Planejamento Operacional de Blending - Pilha ROM")
st.markdown("Integração entre motor de otimização matemática e planejamento executivo de pátio.")

# ==========================================
# 2. FUNÇÕES MATEMÁTICAS E GEOMÉTRICAS
# ==========================================
def build_linear_problem(df, specs, target_mass, volume_max_m3, modo_calculo):
    n = len(df)
    vm = df['vm'].to_numpy(dtype=float)
    ts = df['ts'].to_numpy(dtype=float)
    cinza = df['cinza'].to_numpy(dtype=float)
    rho = df['densidade'].to_numpy(dtype=float)
    
    bounds = []
    for _, row in df.iterrows():
        camada = row['camada']
        max_tons = float(row['ton_report'])
        
        if modo_calculo == "Receita Gerencial (Forcar S5, S4, S3)":
            if camada in ["S5", "S4", "S3"]:
                bounds.append((max_tons, max_tons)) 
            elif camada in ["CS", "CI"]:
                bounds.append((0.0, 0.0)) 
            else:
                bounds.append((0.0, max_tons))
        else:
            bounds.append((0.0, max_tons))
            
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
st.sidebar.header("Regra de Calculo (Pesos)")
modo_calculo = st.sidebar.radio("Como definir a mistura?", 
                        ["Receita Gerencial (Forcar S5, S4, S3)", "Otimizacao Matematica Livre"])

st.sidebar.header("Parametros da Pilha")
alvo_massa = st.sidebar.number_input("Massa Alvo (t)", value=50000.0, step=1000.0)

st.sidebar.header("Geometria")
comp_base = st.sidebar.number_input("Comprimento Base (m)", value=120.0)
larg_base = st.sidebar.number_input("Largura Base (m)", value=70.0)
alt_max = st.sidebar.number_input("Altura Maxima Permitida (m)", value=5.0)
angulo_rep = st.sidebar.number_input("Angulo Repouso (Graus)", value=37.0)

st.sidebar.header("Configuracao de Lifts (Opcao 02)")
num_lifts = st.sidebar.number_input("Quantidade de Lifts", min_value=1, max_value=20, value=5, step=1)

st.sidebar.header("Restricoes da Usina")
vm_min = st.sidebar.number_input("VM Minimo (%)", value=19.30)
ts_max = st.sidebar.number_input("TS Maximo (%)", value=2.20)
cinza_max = st.sidebar.number_input("Cinza/CBS Maximo", value=57.17)

# ==========================================
# 4. TABELA DE DADOS INTERATIVA
# ==========================================
st.subheader("Inventario de Frentes de Lavra (Editavel)")
dados_iniciais = pd.DataFrame({
    "camada": ["S6", "S5", "S4", "S3", "S2", "CS", "CI"],
    "ton_report": [0.0, 4138.0, 7926.0, 16039.0, 33165.0, 136318.0, 118595.0],
    "vm": [20.00, 21.11, 19.10, 21.07, 22.37, 19.23, 17.81],
    "ts": [1.50, 0.70, 0.94, 2.09, 1.44, 1.43, 1.14],
    "cinza": [50.00, 51.31, 58.13, 47.63, 46.47, 51.86, 59.73],
    "densidade": [1.60, 1.70, 1.80, 1.60, 1.70, 1.70, 1.70]
})

df_editado = st.data_editor(dados_iniciais, num_rows="dynamic", use_container_width=True)
df_valido = df_editado[df_editado['ton_report'] > 0].copy()

# ==========================================
# 5. BOTOES DE EXECUCAO E RESULTADOS
# ==========================================
st.markdown("---")
st.subheader("Simular Construcao da Pilha")
col_btn1, col_btn2 = st.columns(2)

btn_op1 = col_btn1.button("Opcao 01: Pilha de Camadas Homogeneas", use_container_width=True, type="primary")
btn_op2 = col_btn2.button("Opcao 02: Pilha Bed Lifting", use_container_width=True, type="primary")

if btn_op1 or btn_op2:
    modo_empilhamento = "Camadas" if btn_op1 else "Bed Lifting"
    
    if df_valido.empty:
        st.error("Nenhuma camada com tonelagem valida para otimizar.")
    else:
        try:
            vol_max = longitudinal_trapezoid_volume(comp_base, larg_base, alt_max, angulo_rep)
            specs = {'vm_min': vm_min, 'ts_max': ts_max, 'cinza_max': cinza_max}
            c, A_ub, b_ub, A_eq, b_eq, bounds = build_linear_problem(df_valido, specs, alvo_massa, vol_max, modo_calculo)
            
            res = linprog(c, A_ub=A_ub if A_ub else None, b_ub=b_ub if b_ub else None, 
                          A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            
            if not res.success:
                st.error(f"O Solver nao encontrou solucao para essas restricoes: {res.message}")
            else:
                st.success("Calculo Finalizado com Sucesso!")
                
                df_valido['ton_calculada'] = res.x
                df_res = df_valido[df_valido['ton_calculada'] > 1e-6].copy()
                
                massa_final = df_res['ton_calculada'].sum()
                vm_final = np.average(df_res['vm'], weights=df_res['ton_calculada'])
                ts_final = np.average(df_res['ts'], weights=df_res['ton_calculada'])
                cinza_final = np.average(df_res['cinza'], weights=df_res['ton_calculada'])
                
                area_base = comp_base * larg_base
                df_res['volume_m3'] = df_res['ton_calculada'] / df_res['densidade']
                df_res['espessura_m'] = df_res['volume_m3'] / area_base
                altura_total_calculada = df_res['espessura_m'].sum()
                
                # --- EXIBICAO DOS RESULTADOS ---
                st.divider()
                st.subheader("Resultados de Qualidade")
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Massa Total", f"{massa_final:,.0f} t")
                c2.metric("VM Final", f"{vm_final:.2f}%", delta=f"Minimo: {vm_min}", delta_color="normal" if vm_final >= vm_min else "inverse")
                c3.metric("TS Final", f"{ts_final:.2f}%", delta=f"Maximo: {ts_max}", delta_color="normal" if ts_final <= ts_max else "inverse")
                c4.metric("Cinza/CBS Final", f"{cinza_final:.2f}", delta=f"Maximo: {cinza_max}", delta_color="normal" if cinza_final <= cinza_max else "inverse")
                
                st.divider()
                col_tabela, col_grafico = st.columns([1, 1.2])
                
                # --- TABELA SE ADAPTA AO MODO ESCOLHIDO ---
                with col_tabela:
                    if modo_empilhamento == "Camadas":
                        st.subheader("Engenharia: Camadas Homogeneas")
                        df_view = df_res[['camada', 'ton_calculada', 'espessura_m']].copy()
                        df_view.columns = ['Frente de Lavra', 'Tonelagem (t)', 'Espessura (m)']
                        st.dataframe(df_view.style.format({'Tonelagem (t)': '{:,.0f}', 'Espessura (m)': '{:.2f}'}), use_container_width=True)
                        st.info(f"**Altura Total da Pilha:** {altura_total_calculada:.2f}m")
                        
                    else:
                        st.subheader("Engenharia: Bed Lifting")
                        df_res['participacao_pct'] = (df_res['ton_calculada'] / massa_final) * 100
                        df_view = df_res[['camada', 'ton_calculada', 'participacao_pct']].copy()
                        df_view.columns = ['Frente de Lavra', 'Tonelagem Total (t)', 'Participacao no Blend (%)']
                        st.dataframe(df_view.style.format({'Tonelagem Total (t)': '{:,.0f}', 'Participacao no Blend (%)': '{:.1f}%'}), use_container_width=True)
                        
                        espessura_por_lift = altura_total_calculada / num_lifts
                        st.info(f"**Operacao:** O Blend acima deve ser misturado e repetido ao longo de **{num_lifts} Lifts**.\n\n**Altura de cada Lift:** {espessura_por_lift:.2f}m \n\n**Altura Total:** {altura_total_calculada:.2f}m")

                # --- GRAFICO SE ADAPTA AO MODO ESCOLHIDO ---
                with col_grafico:
                    fig = go.Figure()
                    y_atual = 0.0
                    largura_atual = larg_base
                    centro_x = larg_base / 2.0  
                    tan_alpha = math.tan(math.radians(angulo_rep))
                    
                    if modo_empilhamento == "Camadas":
                        st.subheader("Visao Transversal: Camadas Homogeneas")
                        ordem_construcao = {'CI': 1, 'CS': 2, 'S2': 3, 'S3': 4, 'S4': 5, 'S5': 6, 'S6': 7}
                        df_res['ordem_plot'] = df_res['camada'].map(ordem_construcao).fillna(99)
                        df_plot = df_res.sort_values('ordem_plot')
                        cores = ['#2E4053', '#839192', '#E67E22', '#D4AC0D', '#27AE60', '#2980B9', '#8E44AD']

                        for i, row in df_plot.iterrows():
                            camada = row['camada']
                            espessura = row['espessura_m']
                            y_topo = y_atual + espessura
                            largura_topo = max(0.0, largura_atual - 2 * (espessura / tan_alpha))
                            
                            x_coords = [centro_x - largura_atual/2, centro_x + largura_atual/2, centro_x + largura_topo/2, centro_x - largura_topo/2, centro_x - largura_atual/2]
                            y_coords = [y_atual, y_atual, y_topo, y_topo, y_atual]
                            cor = cores[int(row['ordem_plot'])-1] if int(row['ordem_plot'])-1 < len(cores) else '#555555'
                            
                            fig.add_trace(go.Scatter(x=x_coords, y=y_coords, fill='toself', name=f"{camada}",
                                mode='lines', line=dict(color='white', width=1), fillcolor=cor,
                                text=f"Camada: {camada}<br>Espessura: {espessura:.2f}m", hoverinfo="text"))
                            
                            y_atual = y_topo
                            largura_atual = largura_topo

                    else:
                        st.subheader("Visao Transversal: Pilha em Lifts Misturados")
                        espessura_lift = altura_total_calculada / num_lifts
                        cores_lifts = ['#34495E', '#2C3E50'] 

                        for i in range(num_lifts):
                            y_topo = y_atual + espessura_lift
                            largura_topo = max(0.0, largura_atual - 2 * (espessura_lift / tan_alpha))
                            
                            x_coords = [centro_x - largura_atual/2, centro_x + largura_atual/2, centro_x + largura_topo/2, centro_x - largura_topo/2, centro_x - largura_atual/2]
                            y_coords = [y_atual, y_atual, y_topo, y_topo, y_atual]
                            cor = cores_lifts[i % 2] 
                            
                            fig.add_trace(go.Scatter(x=x_coords, y=y_coords, fill='toself', name=f"Lift {i+1}",
                                mode='lines', line=dict(color='white', width=1), fillcolor=cor,
                                text=f"<b>Lift {i+1}</b><br>Blend Homogeneizado<br>Espessura: {espessura_lift:.2f}m", hoverinfo="text"))
                            
                            y_atual = y_topo
                            largura_atual = largura_topo

                    fig.add_hline(y=alt_max, line_dash="dash", line_color="red", annotation_text="Limite Altura")
                    fig.update_layout(
                        xaxis_title="Largura da Pilha (m)", yaxis_title="Altura (m)",
                        yaxis=dict(range=[0, alt_max + 1]),
                        xaxis=dict(range=[-5, larg_base + 5], tick0=0, dtick=10),
                        margin=dict(l=20, r=20, t=40, b=20),
                        template="plotly_white", hovermode="x unified", showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Erro na execucao matematica: {e}")
