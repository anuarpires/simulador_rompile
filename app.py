import streamlit as st
import pandas as pd
import numpy as np
import math
from scipy.optimize import linprog
import plotly.graph_objects as go

# ==========================================
# 1. CONFIGURAÇÃO DA PÁGINA E ESTILO
# ==========================================
# Define o layout da página para ocupar toda a largura da tela do navegador
st.set_page_config(page_title="Otimizador de Pilha ROM - Copelmi", layout="wide")
st.title("Otimizador Avançado de Blending - Pilha ROM")
st.markdown("Motor de otimização por Programação Linear para maximizar o fechamento da pilha respeitando restrições.")

# ==========================================
# 2. FUNÇÕES MATEMÁTICAS E GEOMÉTRICAS
# ==========================================

def build_linear_problem(df, specs, target_mass, volume_max_m3):
    """
    Esta função traduz o problema de blending do mundo real para uma matriz
    matemática que o Solver (linprog) consegue resolver.
    """
    n = len(df) # Número de camadas disponíveis (ex: 7 camadas)
    
    # Extrai as colunas de qualidade como arrays matemáticos (vetores)
    vm = df['vm'].to_numpy(dtype=float)
    ts = df['ts'].to_numpy(dtype=float)
    cinza = df['cinza'].to_numpy(dtype=float)
    rho = df['densidade'].to_numpy(dtype=float)
    
    # BOUNDS (Limites individuais de cada variável):
    # Garante que o otimizador não vai sugerir uma tonelagem negativa,
    # nem vai sugerir usar mais toneladas do que o inventário possui.
    bounds = [(0.0, float(row['ton_report'])) for _, row in df.iterrows()]
    
    # Matrizes de inequação (Upper Bound: <=) e equação (Equal: ==)
    A_ub, b_ub = [], [] 
    A_eq, b_eq = [], []
    
    # RESTRIÇÃO 1: EQUAÇÃO DE MASSA (==)
    # A soma das toneladas de todas as camadas DEVE ser exatamente igual ao alvo (ex: 50.000t).
    A_eq.append(np.ones(n))
    b_eq.append(float(target_mass))
    
    # RESTRIÇÕES 2, 3 e 4: LIMITES DE QUALIDADE DA USINA (<=)
    # Para TS e Cinza, a média ponderada tem que ser menor ou igual ao limite.
    # A fórmula reorganizada fica: (TS_camada - TS_limite) * massa <= 0
    if specs.get('ts_max') is not None:
        A_ub.append(ts - float(specs['ts_max']))
        b_ub.append(0.0)
        
    if specs.get('cinza_max') is not None:
        A_ub.append(cinza - float(specs['cinza_max']))
        b_ub.append(0.0)
        
    # Para o VM, a usina exige um MÍNIMO. Multiplicamos por -1 para 
    # inverter o sinal matemático e encaixar na matriz de <= (menor igual).
    if specs.get('vm_min') is not None:
        A_ub.append(float(specs['vm_min']) - vm)
        b_ub.append(0.0)
        
    # RESTRIÇÃO 5: LIMITE GEOMÉTRICO DO PÁTIO (<=)
    # O somatório dos volumes (Massa / Densidade) não pode estourar o limite físico do pátio.
    if volume_max_m3 is not None:
        A_ub.append(1.0 / rho)
        b_ub.append(float(volume_max_m3))
        
    # FUNÇÃO OBJETIVO: O que o algoritmo deve tentar minimizar?
    def zscore(x):
        # A função Z-Score normaliza (padroniza) os valores. Como VM está na casa dos 20%
        # e o TS na casa dos 1%, normalizar garante que o algoritmo dê peso igual a ambos,
        # impedindo que o VM "engula" o TS na hora do cálculo matemático.
        s = x.std()
        return np.zeros_like(x) if s == 0 else (x - x.mean()) / s
        
    # A fórmula abaixo diz: "Busque a mistura que jogue o VM para cima (-1) e 
    # ao mesmo tempo puxe o TS e a Cinza para baixo (+1)".
    c = (-1.0 * zscore(vm) + 1.0 * zscore(ts) + 1.0 * zscore(cinza))
    
    return c, A_ub, b_ub, A_eq, b_eq, bounds

def longitudinal_trapezoid_volume(comp, larg, alt_max, angulo):
    """
    Calcula o volume máximo em metros cúbicos que o pátio suporta.
    Imagina a pilha como um prisma com seção transversal em formato de trapézio.
    """
    # Converte o ângulo de graus para radianos e calcula a tangente
    tanv = math.tan(math.radians(angulo))
    
    # Calcula quanto a pilha "encolhe" para dentro dos dois lados ao atingir a altura máxima
    retracao_lateral = alt_max / tanv
    
    # A largura lá no topo (crista) é a base menos o que encolheu dos dois lados
    largura_topo = max(0.0, larg - 2.0 * retracao_lateral)
    
    # Área do trapézio: (Base Maior + Base Menor) * Altura / 2
    area_secao = alt_max * (larg + largura_topo) / 2.0
    
    # Volume total = Área da face multiplicada pelo comprimento da pilha
    return comp * area_secao

# ==========================================
# 3. INTERFACE DE USUÁRIO (Sidebar)
# ==========================================
# Cria os campos de entrada numéricos na barra lateral esquerda
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

# Cria um DataFrame do Pandas simulando a leitura do arquivo CSV
dados_iniciais = pd.DataFrame({
    "camada": ["S6", "S5", "S4", "S3", "S2", "CS", "CI"],
    "ton_report": [0.0, 4138.0, 7926.0, 16039.0, 26914.0, 36211.0, 55195.0],
    "vm": [20.00, 21.11, 19.10, 21.07, 22.37, 19.23, 17.81],
    "ts": [1.50, 0.70, 0.94, 2.09, 1.44, 1.43, 1.14],
    "cinza": [50.00, 51.31, 58.13, 47.63, 46.47, 51.86, 59.73],
    "densidade": [1.60, 1.70, 1.80, 1.60, 1.70, 1.70, 1.70]
})

# st.data_editor transforma o DataFrame em uma planilha onde o usuário pode digitar
df_editado = st.data_editor(dados_iniciais, num_rows="dynamic", use_container_width=True)

# Remove as camadas que o usuário zerou a tonelagem para não sujar o cálculo
df_valido = df_editado[df_editado['ton_report'] > 0].copy()

# ==========================================
# 5. BOTÃO DE OTIMIZAÇÃO E RESULTADOS
# ==========================================
if st.button("Rodar Solver de Otimização", type="primary"):
    if df_valido.empty:
        st.error("Nenhuma camada com tonelagem válida para otimizar.")
    else:
        try:
            # Etapa 1: Preparar as restrições para a biblioteca SciPy
            vol_max = longitudinal_trapezoid_volume(comp_base, larg_base, alt_max, angulo_rep)
            specs = {'vm_min': vm_min, 'ts_max': ts_max, 'cinza_max': cinza_max}
            c, A_ub, b_ub, A_eq, b_eq, bounds = build_linear_problem(df_valido, specs, alvo_massa, vol_max)
            
            # Etapa 2: Executar o algoritmo Simplex (linprog)
            # O método 'highs' é um dos solvers lineares mais modernos e rápidos embutidos no SciPy
            res = linprog(c, A_ub=A_ub if A_ub else None, b_ub=b_ub if b_ub else None, 
                          A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            
            # Checa se o algoritmo encontrou o "ótimo" matemático
            if not res.success:
                st.error(f"O Solver não encontrou uma solução viável matemática para essas restrições: {res.message}")
            else:
                st.success("Solução Otimizada Encontrada!")
                
                # Etapa 3: Extrair a tonelagem ideal calculada pelo solver (res.x)
                df_valido['ton_calculada'] = res.x
                # Filtra apenas as camadas que o solver decidiu usar (massa > 0)
                df_res = df_valido[df_valido['ton_calculada'] > 1e-6].copy()
                
                # Calcula as qualidades finais usando média ponderada (peso da massa)
                massa_final = df_res['ton_calculada'].sum()
                vm_final = np.average(df_res['vm'], weights=df_res['ton_calculada'])
                ts_final = np.average(df_res['ts'], weights=df_res['ton_calculada'])
                cinza_final = np.average(df_res['cinza'], weights=df_res['ton_calculada'])
                
                # Etapa 4: Geometria retroativa (Massa -> Volume -> Altura na pilha)
                area_base = comp_base * larg_base
                df_res['volume_m3'] = df_res['ton_calculada'] / df_res['densidade']
                df_res['espessura_m'] = df_res['volume_m3'] / area_base
                
                # ----------------------------------------
                # EXIBIÇÃO VISUAL NA TELA
                # ----------------------------------------
                st.divider()
                st.subheader("Resultados de Qualidade")
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Massa Total", f"{massa_final:,.0f} t", delta=f"{massa_final - alvo_massa:,.0f} vs Alvo")
                
                # Lógica para cor da setinha (vermelho se estiver fora da especificação)
                cor_vm = "normal" if vm_final >= vm_min else "inverse"
                c2.metric("VM Final", f"{vm_final:.2f}%", delta=f"Mínimo: {vm_min}", delta_color=cor_vm)
                
                cor_ts = "normal" if ts_final <= ts_max else "inverse"
                c3.metric("TS Final", f"{ts_final:.2f}%", delta=f"Máximo: {ts_max}", delta_color=cor_ts)
                
                cor_cz = "normal" if cinza_final <= cinza_max else "inverse"
                c4.metric("Cinza/CBS Final", f"{cinza_final:.2f}", delta=f"Máximo: {cinza_max}", delta_color=cor_cz)
                
                st.divider()
                st.subheader("Composição e Geometria da Pilha")
                
                col_tabela, col_grafico = st.columns([1, 1.2])
                
                # TABELA DE ENGENHARIA (Esquerda)
                with col_tabela:
                    df_view = df_res[['camada', 'ton_calculada', 'volume_m3', 'espessura_m']].copy()
                    df_view.columns = ['Camada', 'Tonelagem (t)', 'Volume (m³)', 'Espessura (m)']
                    st.dataframe(df_view.style.format({'Tonelagem (t)': '{:,.0f}', 'Volume (m³)': '{:,.0f}', 'Espessura (m)': '{:.2f}'}), use_container_width=True)
                    st.info(f"**Geometria Projetada:** Base de {larg_base}m x {comp_base}m. \n\n**Altura Total Calculada:** {df_res['espessura_m'].sum():.2f}m")
                
                # GRÁFICO DA PILHA TRANSVERSAL (Direita)
                with col_grafico:
                    # Dicionário fixo para forçar a renderização de baixo para cima (do fundo ao topo)
                    ordem_construcao = {'CI': 1, 'CS': 2, 'S2': 3, 'S3': 4, 'S4': 5, 'S5': 6, 'S6': 7}
                    df_res['ordem_plot'] = df_res['camada'].map(ordem_construcao).fillna(99)
                    df_plot = df_res.sort_values('ordem_plot')

                    fig = go.Figure()
                    
                    # Variáveis de controle para desenhar um bloco em cima do outro
                    y_atual = 0.0
                    largura_atual = larg_base
                    centro_x = larg_base / 2.0  # Desloca o gráfico para começar em X = 0
                    tan_alpha = math.tan(math.radians(angulo_rep))
                    
                    # Paleta de cores para diferenciar visualmente as camadas
                    cores = ['#2E4053', '#839192', '#E67E22', '#D4AC0D', '#27AE60', '#2980B9', '#8E44AD']

                    # Loop que desenha cada camada como um polígono (trapézio)
                    for i, row in df_plot.iterrows():
                        camada = row['camada']
                        espessura = row['espessura_m']
                        
                        y_topo = y_atual + espessura
                        
                        # A largura do topo é menor que a da base por causa da inclinação do talude
                        largura_topo = max(0.0, largura_atual - 2 * (espessura / tan_alpha))
                        
                        # Coordenadas X e Y das 4 pontas do polígono (esq-base, dir-base, dir-topo, esq-topo, fecha)
                        x_coords = [
                            centro_x - largura_atual/2, 
                            centro_x + largura_atual/2, 
                            centro_x + largura_topo/2, 
                            centro_x - largura_topo/2, 
                            centro_x - largura_atual/2
                        ]
                        y_coords = [y_atual, y_atual, y_topo, y_topo, y_atual]
                        
                        cor = cores[int(row['ordem_plot'])-1] if int(row['ordem_plot'])-1 < len(cores) else '#555555'
                        
                        # Adiciona a forma no gráfico
                        fig.add_trace(go.Scatter(
                            x=x_coords, y=y_coords, fill='toself', name=f"{camada} ({espessura:.2f}m)",
                            mode='lines', line=dict(color='white', width=1), fillcolor=cor,
                            text=f"Camada: {camada}<br>Espessura: {espessura:.2f}m<br>Massa: {row['ton_calculada']:,.0f}t",
                            hoverinfo="text" # Tooltip interativo
                        ))
                        
                        # Atualiza as referências de altura e largura para a PRÓXIMA camada
                        y_atual = y_topo
                        largura_atual = largura_topo

                    # Linha de alerta pontilhada mostrando o limite seguro de empilhamento
                    fig.add_hline(y=alt_max, line_dash="dash", line_color="red", annotation_text="Altura Máx")

                    # Configurações visuais do gráfico (eixos fixos, margens, grid)
                    fig.update_layout(
                        title="Seção Transversal da Pilha Calculada",
                        xaxis_title="Largura da Pilha (m)",
                        yaxis_title="Altura (m)",
                        yaxis=dict(range=[0, alt_max + 1]),
                        xaxis=dict(range=[-5, larg_base + 5], tick0=0, dtick=10), # Força os marcadores a andar de 10 em 10m
                        margin=dict(l=20, r=20, t=40, b=20),
                        template="plotly_white",
                        hovermode="x unified"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Erro na execução matemática: {e}")
