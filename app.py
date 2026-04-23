import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import linprog

# ============================================================
# 1) CONFIGURAÇÃO GERAL DA APLICAÇÃO
# ============================================================
st.set_page_config(page_title="Otimizador de Pilha ROM - Copelmi", layout="wide")
st.title("Otimizador Avançado de Blending - Pilha ROM")

st.markdown(
    """
    Este aplicativo otimiza o **blend global** por programação linear e, em seguida,
    representa a pilha de duas formas distintas:

    - **Pilha A - Estratos por camada**: primeira camada esteirada como base e as demais camadas empilhadas por cima;
    - **Pilha B - Lift/Sublifting**: cada **camada geológica** é subdividida em **lifts** e cada lift pode ser subdividido em **sublifts**.

    A composição química média global do blend permanece a mesma; o que muda entre os casos
    é a **distribuição espacial da pilha**, a **tabela executiva** e a **visualização gráfica**.
    """
)

ORDEM_CONSTRUCAO = {"CI": 1, "CS": 2, "S2": 3, "S3": 4, "S4": 5, "S5": 6, "S6": 7}

CORES_CAMADAS = {
    "CI": "#2E4053",
    "CS": "#839192",
    "S2": "#E67E22",
    "S3": "#D4AC0D",
    "S4": "#27AE60",
    "S5": "#2980B9",
    "S6": "#8E44AD",
}

# ============================================================
# 2) FUNÇÕES MATEMÁTICAS DO BLEND
# ============================================================
"""
NOTA DE ENGENHARIA DE MINAS SOBRE O COMPORTAMENTO DO SOLVER:
O algoritmo de programação linear excluiu as camadas S4 e CI porque, estritamente sob a ótica 
matemática da função objetivo, elas representam o pior custo-benefício químico (maior teor de 
cinzas e menor matéria volátil) para fechar a meta de 50.000 toneladas. O otimizador atua de 
forma puramente analítica, priorizando a exaustão total das frentes de melhor qualidade (S2, S3 e S5) 
e utilizando apenas uma fração da camada CS para completar o pequeno saldo numérico restante, 
descartando sumariamente as opções inferiores. 

No entanto, embora entregue o blend teoricamente perfeito, essa solução esbarra na restrição 
geométrica da mineração: como a camada S4 está fisicamente sobreposta à S3 no pacote geológico, 
é operacionalmente impossível acessar e extrair a S3 sem antes decapear e destinar a S4, 
evidenciando que o código precisará de novas regras de precedência — amarrando a extração de 
uma camada inferior à obrigatoriedade de uso da camada superior — para que a matemática reflita 
com exatidão a realidade da operação.
"""
def zscore(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return np.zeros_like(x) if s == 0 else (x - x.mean()) / s

def build_linear_problem(df: pd.DataFrame, specs: dict, target_mass: float, volume_max_m3: float | None):
    n = len(df)
    vm = df["vm"].to_numpy(dtype=float)
    ts = df["ts"].to_numpy(dtype=float)
    cinza = df["cinza"].to_numpy(dtype=float)
    rho = df["densidade"].to_numpy(dtype=float)

    bounds = [(0.0, float(row["ton_report"])) for _, row in df.iterrows()]

    A_ub, b_ub = [], []
    A_eq, b_eq = [], []

    A_eq.append(np.ones(n))
    b_eq.append(float(target_mass))

    if specs.get("ts_max") is not None:
        A_ub.append(ts - float(specs["ts_max"]))
        b_ub.append(0.0)

    if specs.get("cinza_max") is not None:
        A_ub.append(cinza - float(specs["cinza_max"]))
        b_ub.append(0.0)

    if specs.get("vm_min") is not None:
        A_ub.append(float(specs["vm_min"]) - vm)
        b_ub.append(0.0)

    if volume_max_m3 is not None:
        A_ub.append(1.0 / rho)
        b_ub.append(float(volume_max_m3))

    c = (-1.0 * zscore(vm) + 1.0 * zscore(ts) + 1.0 * zscore(cinza))
    return c, A_ub, b_ub, A_eq, b_eq, bounds

def calculate_quality_metrics(df_res: pd.DataFrame) -> dict:
    massa_final = float(df_res["ton_calculada"].sum())
    volume_total = float((df_res["ton_calculada"] / df_res["densidade"]).sum())
    vm_final = float(np.average(df_res["vm"], weights=df_res["ton_calculada"]))
    ts_final = float(np.average(df_res["ts"], weights=df_res["ton_calculada"]))
    cinza_final = float(np.average(df_res["cinza"], weights=df_res["ton_calculada"]))
    rho_aparente = massa_final / volume_total if volume_total > 0 else np.nan

    return {
        "massa_final": massa_final,
        "volume_total": volume_total,
        "vm_final": vm_final,
        "ts_final": ts_final,
        "cinza_final": cinza_final,
        "rho_aparente": rho_aparente,
    }

def enrich_total_composition(df_res: pd.DataFrame, metrics: dict) -> pd.DataFrame:
    df_total = df_res.copy()
    df_total["volume_m3"] = df_total["ton_calculada"] / df_total["densidade"]
    df_total["frac_massa_%"] = 100.0 * df_total["ton_calculada"] / metrics["massa_final"]
    df_total["frac_volume_%"] = 100.0 * df_total["volume_m3"] / metrics["volume_total"]
    df_total["ordem_plot"] = df_total["camada"].map(ORDEM_CONSTRUCAO).fillna(99)
    return df_total.sort_values("ordem_plot").reset_index(drop=True)

# ============================================================
# 3) FUNÇÕES GEOMÉTRICAS DO INVÓLUCRO DA PILHA
# ============================================================
def width_at_height(y: float, larg_base: float, angulo: float) -> float:
    tanv = math.tan(math.radians(angulo))
    if tanv <= 0:
        return larg_base
    return max(0.0, larg_base - 2.0 * (y / tanv))

def cross_section_area_up_to(h: float, larg_base: float, angulo: float) -> float:
    h = max(0.0, h)
    largura_topo = width_at_height(h, larg_base, angulo)
    return h * (larg_base + largura_topo) / 2.0

def longitudinal_trapezoid_volume(comp: float, larg_base: float, alt_max: float, angulo: float) -> float:
    return comp * cross_section_area_up_to(alt_max, larg_base, angulo)

def volume_between_heights(y0: float, y1: float, comp: float, larg_base: float, angulo: float) -> float:
    y0 = max(0.0, y0)
    y1 = max(y0, y1)
    return comp * (cross_section_area_up_to(y1, larg_base, angulo) - cross_section_area_up_to(y0, larg_base, angulo))

def solve_height_for_volume(volume_target: float, comp: float, larg_base: float, alt_max: float, angulo: float) -> float:
    volume_max = longitudinal_trapezoid_volume(comp, larg_base, alt_max, angulo)
    if volume_target <= 0:
        return 0.0
    if volume_target >= volume_max:
        return alt_max
    lo, hi = 0.0, alt_max
    for _ in range(70):
        mid = (lo + hi) / 2.0
        v_mid = longitudinal_trapezoid_volume(comp, larg_base, mid, angulo)
        if v_mid < volume_target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0

def solve_upper_height_for_segment_volume(
    y_base: float, target_volume: float, comp: float, larg_base: float, alt_max: float, angulo: float
) -> float:
    y_base = max(0.0, y_base)
    if target_volume <= 0:
        return y_base
    vol_disponivel = volume_between_heights(y_base, alt_max, comp, larg_base, angulo)
    if target_volume >= vol_disponivel:
        return alt_max
    lo, hi = y_base, alt_max
    for _ in range(70):
        mid = (lo + hi) / 2.0
        v_mid = volume_between_heights(y_base, mid, comp, larg_base, angulo)
        if v_mid < target_volume:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0

# ============================================================
# 4) PILHA A - ESTRATOS POR CAMADA
# ============================================================
def prepare_pile_a_strata(
    df_res: pd.DataFrame, comp_base: float, larg_base: float, alt_max: float, angulo_rep: float
) -> dict:
    df_total = df_res.copy()
    df_total["volume_m3"] = df_total["ton_calculada"] / df_total["densidade"]
    df_total["ordem_plot"] = df_total["camada"].map(ORDEM_CONSTRUCAO).fillna(99)
    df_total = df_total.sort_values("ordem_plot").reset_index(drop=True)

    rows = []
    y_atual = 0.0

    for _, row in df_total.iterrows():
        vol = float(row["volume_m3"])
        y_topo = solve_upper_height_for_segment_volume(
            y_base=y_atual, target_volume=vol, comp=comp_base, larg_base=larg_base, alt_max=alt_max, angulo=angulo_rep
        )

        rows.append(
            {
                "camada": row["camada"],
                "ton_calculada": float(row["ton_calculada"]),
                "densidade": float(row["densidade"]),
                "volume_m3": vol,
                "y_base_m": y_atual,
                "y_topo_m": y_topo,
                "espessura_m": y_topo - y_atual,
                "largura_base_m": width_at_height(y_atual, larg_base, angulo_rep),
                "largura_topo_m": width_at_height(y_topo, larg_base, angulo_rep),
            }
        )
        y_atual = y_topo

    df_camadas = pd.DataFrame(rows)
    altura_efetiva = float(df_camadas["y_topo_m"].max()) if not df_camadas.empty else 0.0

    return {
        "df_camadas": df_camadas,
        "altura_efetiva_m": altura_efetiva,
    }

# ============================================================
# 5) PILHA B - LIFT / SUBLIFTING POR CAMADA GEOLÓGICA
# ============================================================
def prepare_pile_b_lifts(
    df_res: pd.DataFrame, comp_base: float, larg_base: float, alt_max: float, angulo_rep: float, altura_lift: float, altura_sublift: float
) -> dict:
    pile_a = prepare_pile_a_strata(
        df_res=df_res, comp_base=comp_base, larg_base=larg_base, alt_max=alt_max, angulo_rep=angulo_rep
    )

    df_camadas = pile_a["df_camadas"].copy()
    metrics = calculate_quality_metrics(df_res)
    df_layers = enrich_total_composition(df_res, metrics).copy()

    lift_rows = []
    sublift_rows = []
    lift_global_id = 1

    for _, camada_row in df_camadas.iterrows():
        camada = camada_row["camada"]
        densidade = float(camada_row["densidade"])
        y_layer_base = float(camada_row["y_base_m"])
        y_layer_topo = float(camada_row["y_topo_m"])

        y0 = y_layer_base
        lift_seq = 1

        while y0 < y_layer_topo - 1e-9:
            y1 = min(y_layer_topo, y0 + altura_lift)
            vol_lift = volume_between_heights(y0, y1, comp_base, larg_base, angulo_rep)
            massa_lift = vol_lift * densidade
            lift_nome = f"{camada}-L{lift_seq}"

            lift_rows.append(
                {
                    "lift_global": lift_global_id,
                    "camada": camada,
                    "lift_seq": lift_seq,
                    "lift_nome": lift_nome,
                    "y_base_m": y0,
                    "y_topo_m": y1,
                    "altura_lift_m": y1 - y0,
                    "largura_base_m": width_at_height(y0, larg_base, angulo_rep),
                    "largura_topo_m": width_at_height(y1, larg_base, angulo_rep),
                    "volume_lift_m3": vol_lift,
                    "massa_lift_t": massa_lift,
                    "densidade": densidade,
                }
            )

            z0 = y0
            sub_seq = 1
            while z0 < y1 - 1e-9:
                z1 = min(y1, z0 + altura_sublift)
                vol_sub = volume_between_heights(z0, z1, comp_base, larg_base, angulo_rep)
                massa_sub = vol_sub * densidade

                sublift_rows.append(
                    {
                        "lift_global": lift_global_id,
                        "camada": camada,
                        "lift_seq": lift_seq,
                        "lift_nome": lift_nome,
                        "sublift_seq": sub_seq,
                        "sublift_nome": f"{lift_nome}-S{sub_seq}",
                        "y_base_m": z0,
                        "y_topo_m": z1,
                        "altura_sublift_m": z1 - z0,
                        "largura_base_m": width_at_height(z0, larg_base, angulo_rep),
                        "largura_topo_m": width_at_height(z1, larg_base, angulo_rep),
                        "volume_sublift_m3": vol_sub,
                        "massa_sublift_t": massa_sub,
                        "densidade": densidade,
                    }
                )
                z0 = z1
                sub_seq += 1

            y0 = y1
            lift_seq += 1
            lift_global_id += 1

    df_lifts = pd.DataFrame(lift_rows)
    df_sublifts = pd.DataFrame(sublift_rows)

    return {
        "df_layers": df_layers,
        "df_camadas_base": df_camadas,
        "df_lifts": df_lifts,
        "df_sublifts": df_sublifts,
        "altura_efetiva_m": pile_a["altura_efetiva_m"],
    }

# ============================================================
# 6) FUNÇÕES DE VISUALIZAÇÃO GRÁFICA
# ============================================================
def build_pile_a_figure(df_camadas: pd.DataFrame, larg_base: float, alt_max: float, angulo_rep: float) -> go.Figure:
    fig = go.Figure()
    centro_x = larg_base / 2.0

    for _, row in df_camadas.iterrows():
        y0 = float(row["y_base_m"])
        y1 = float(row["y_topo_m"])
        camada = row["camada"]

        larg_inf = width_at_height(y0, larg_base, angulo_rep)
        larg_sup = width_at_height(y1, larg_base, angulo_rep)

        x_coords = [
            centro_x - larg_inf / 2.0,
            centro_x + larg_inf / 2.0,
            centro_x + larg_sup / 2.0,
            centro_x - larg_sup / 2.0,
            centro_x - larg_inf / 2.0,
        ]
        y_coords = [y0, y0, y1, y1, y0]

        fig.add_trace(
            go.Scatter(
                x=x_coords, y=y_coords, fill="toself", mode="lines",
                line=dict(color="white", width=1), fillcolor=CORES_CAMADAS.get(camada, "#777777"),
                name=camada, legendgroup=camada,
                text=(
                    f"Camada: {camada}"
                    f"<br>Base: {y0:.2f} m"
                    f"<br>Topo: {y1:.2f} m"
                    f"<br>Espessura: {row['espessura_m']:.2f} m"
                    f"<br>Massa: {row['ton_calculada']:,.0f} t"
                    f"<br>Volume: {row['volume_m3']:,.0f} m³"
                ),
                hoverinfo="text",
            )
        )

    fig.add_hline(y=alt_max, line_dash="dash", line_color="red", annotation_text="Altura máxima do pátio")
    fig.update_layout(
        title="Seção Transversal - Pilha A | Estratos por camada",
        xaxis_title="Largura da pilha (m)", yaxis_title="Altura (m)",
        xaxis=dict(range=[-5, larg_base + 5], tick0=0, dtick=10),
        yaxis=dict(
            range=[0, max(alt_max + 0.5, float(df_camadas["y_topo_m"].max()) + 0.5 if not df_camadas.empty else alt_max + 0.5)]
        ),
        template="plotly_white", legend_title="Camadas",
        margin=dict(l=20, r=20, t=60, b=20), hovermode="closest",
    )
    return fig

def build_pile_b_figure(model: dict, larg_base: float, alt_max: float, angulo_rep: float) -> go.Figure:
    df_lifts = model["df_lifts"].copy()
    df_sublifts = model["df_sublifts"].copy()
    altura_efetiva = model["altura_efetiva_m"]

    fig = go.Figure()
    centro_x = larg_base / 2.0

    for _, row in df_lifts.iterrows():
        camada = row["camada"]
        y0 = float(row["y_base_m"])
        y1 = float(row["y_topo_m"])

        larg_inf = width_at_height(y0, larg_base, angulo_rep)
        larg_sup = width_at_height(y1, larg_base, angulo_rep)

        x_coords = [
            centro_x - larg_inf / 2.0, centro_x + larg_inf / 2.0,
            centro_x + larg_sup / 2.0, centro_x - larg_sup / 2.0,
            centro_x - larg_inf / 2.0,
        ]
        y_coords = [y0, y0, y1, y1, y0]

        fig.add_trace(
            go.Scatter(
                x=x_coords, y=y_coords, fill="toself", mode="lines",
                line=dict(color="white", width=1), fillcolor=CORES_CAMADAS.get(camada, "#777777"),
                name=camada, legendgroup=camada, showlegend=False,
                text=(
                    f"Lift: {row['lift_nome']}"
                    f"<br>Camada: {camada}"
                    f"<br>Base: {y0:.2f} m"
                    f"<br>Topo: {y1:.2f} m"
                    f"<br>Espessura do lift: {row['altura_lift_m']:.2f} m"
                    f"<br>Massa do lift: {row['massa_lift_t']:,.0f} t"
                    f"<br>Volume do lift: {row['volume_lift_m3']:,.0f} m³"
                ),
                hoverinfo="text",
            )
        )

    for _, row in df_sublifts.iterrows():
        y1 = float(row["y_topo_m"])
        if y1 >= altura_efetiva - 1e-9:
            continue
        fig.add_hline(
            y=y1, line_dash="dot", line_color="rgba(30,30,30,0.25)",
            annotation_text=row["sublift_nome"], annotation_position="left",
        )

    for camada in df_lifts["camada"].drop_duplicates().tolist():
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=12, color=CORES_CAMADAS.get(camada, "#777777")),
                name=camada, legendgroup=camada, showlegend=True,
            )
        )

    fig.add_hline(y=alt_max, line_dash="dash", line_color="red", annotation_text="Altura máxima do pátio")
    fig.update_layout(
        title="Seção Transversal - Pilha B | Lifts geológicos / Sublifting",
        xaxis_title="Largura da pilha (m)", yaxis_title="Altura (m)",
        xaxis=dict(range=[-5, larg_base + 5], tick0=0, dtick=10),
        yaxis=dict(range=[0, max(alt_max + 0.5, altura_efetiva + 0.5)]),
        template="plotly_white", legend_title="Camadas",
        margin=dict(l=20, r=20, t=60, b=20), hovermode="closest",
    )
    return fig

# ============================================================
# 8) SIDEBAR - PARÂMETROS DE ENTRADA
# ============================================================
st.sidebar.header("Parâmetros da Pilha")
alvo_massa = st.sidebar.number_input("Massa Alvo (t)", value=50000.0, step=1000.0)

st.sidebar.header("Geometria do Pátio")
comp_base = st.sidebar.number_input("Comprimento Base (m)", value=120.0)
larg_base = st.sidebar.number_input("Largura Base (m)", value=70.0)
alt_max = st.sidebar.number_input("Altura Máxima (m)", value=5.0)
angulo_rep = st.sidebar.number_input("Ângulo de Repouso (graus)", value=37.0)

st.sidebar.header("Caso para Exibição")
modo_construtivo = st.sidebar.radio(
    "Escolha a pilha a exibir",
    options=["Pilha A - Estratos por camada", "Pilha B - Lift/Sublifting"],
)

st.sidebar.header("Parâmetros da Pilha B")
altura_lift = st.sidebar.number_input("Altura do lift (m)", min_value=0.1, value=1.0, step=0.1)
altura_sublift = st.sidebar.number_input("Altura do sublift (m)", min_value=0.1, value=0.5, step=0.1)

if altura_sublift > altura_lift:
    st.sidebar.warning("A altura do sublift foi limitada à altura do lift.")
    altura_sublift = altura_lift

st.sidebar.header("Restrições da Usina")
vm_min = st.sidebar.number_input("VM Mínimo (%)", value=19.30)
ts_max = st.sidebar.number_input("TS Máximo (%)", value=2.20)
cinza_max = st.sidebar.number_input("Cinza/CBS Máximo", value=57.17)

# ============================================================
# 9) TABELA DE DADOS EDITÁVEL
# ============================================================
st.subheader("Inventário de Frentes de Lavra (Editável)")

dados_iniciais = pd.DataFrame(
    {
        "camada": ["S6", "S5", "S4", "S3", "S2", "CS", "CI"],
        "ton_report": [0.0, 4138.0, 7926.0, 16039.0, 26914.0, 36211.0, 55195.0],
        "vm": [20.00, 21.11, 19.10, 21.07, 22.37, 19.23, 17.81],
        "ts": [1.50, 0.70, 0.94, 2.09, 1.44, 1.43, 1.14],
        "cinza": [50.00, 51.31, 58.13, 47.63, 46.47, 51.86, 59.73],
        "densidade": [1.60, 1.70, 1.80, 1.60, 1.70, 1.70, 1.70],
    }
)

df_editado = st.data_editor(dados_iniciais, num_rows="dynamic", use_container_width=True)
df_valido = df_editado[df_editado["ton_report"] > 0].copy()

# ============================================================
# 10) EXECUÇÃO DO SOLVER E APRESENTAÇÃO DOS RESULTADOS
# ============================================================
if st.button("Rodar Solver de Otimização", type="primary"):
    if df_valido.empty:
        st.error("Nenhuma camada com tonelagem válida para otimizar.")
    else:
        try:
            vol_max = longitudinal_trapezoid_volume(comp_base, larg_base, alt_max, angulo_rep)
            specs = {"vm_min": vm_min, "ts_max": ts_max, "cinza_max": cinza_max}
            c, A_ub, b_ub, A_eq, b_eq, bounds = build_linear_problem(df_valido, specs, alvo_massa, vol_max)

            res = linprog(
                c, A_ub=A_ub if A_ub else None, b_ub=b_ub if b_ub else None,
                A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs",
            )

            if not res.success:
                st.error(f"O solver não encontrou uma solução viável: {res.message}")
            else:
                st.success("Solução otimizada encontrada.")
                df_valido["ton_calculada"] = res.x
                df_res = df_valido[df_valido["ton_calculada"] > 1e-6].copy()

                metrics = calculate_quality_metrics(df_res)
                df_total = enrich_total_composition(df_res, metrics)

                pile_a = prepare_pile_a_strata(
                    df_res=df_res, comp_base=comp_base, larg_base=larg_base, alt_max=alt_max, angulo_rep=angulo_rep
                )
                pile_b = prepare_pile_b_lifts(
                    df_res=df_res, comp_base=comp_base, larg_base=larg_base, alt_max=alt_max, angulo_rep=angulo_rep,
                    altura_lift=altura_lift, altura_sublift=altura_sublift
                )

                st.divider()
                st.subheader("Resultados de Qualidade Global do Blend")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric(
                    "Massa Total", f"{metrics['massa_final']:,.0f} t", delta=f"{metrics['massa_final'] - alvo_massa:,.0f} vs alvo",
                )

                cor_vm = "normal" if metrics["vm_final"] >= vm_min else "inverse"
                c2.metric("VM Final", f"{metrics['vm_final']:.2f}%", delta=f"Mínimo: {vm_min}", delta_color=cor_vm)

                cor_ts = "normal" if metrics["ts_final"] <= ts_max else "inverse"
                c3.metric("TS Final", f"{metrics['ts_final']:.2f}%", delta=f"Máximo: {ts_max}", delta_color=cor_ts)

                cor_cz = "normal" if metrics["cinza_final"] <= cinza_max else "inverse"
                c4.metric("Cinza/CBS Final", f"{metrics['cinza_final']:.2f}", delta=f"Máximo: {cinza_max}", delta_color=cor_cz)

                st.caption(
                    f"Volume usado: {metrics['volume_total']:,.0f} m³ | "
                    f"Densidade aparente equivalente: {metrics['rho_aparente']:.3f} t/m³ | "
                    f"Volume máximo do pátio: {vol_max:,.0f} m³"
                )

                st.divider()
                st.subheader("Composição Total do Blend Otimizado")
                st.dataframe(
                    df_total[["camada", "ton_calculada", "volume_m3", "frac_massa_%", "frac_volume_%"]].style.format(
                        {
                            "ton_calculada": "{:,.0f}",
                            "volume_m3": "{:,.0f}",
                            "frac_massa_%": "{:.2f}",
                            "frac_volume_%": "{:.2f}",
                        }
                    ),
                    use_container_width=True,
                )

                st.divider()
                st.subheader("Composição, Geometria e Visualização do Caso Selecionado")

                col_tabela, col_grafico = st.columns([1.15, 1.2])

                with col_tabela:
                    st.markdown(f"**Caso exibido:** {modo_construtivo}")

                    if modo_construtivo == "Pilha A - Estratos por camada":
                        df_a = pile_a["df_camadas"].copy()
                        st.info(
                            f"**Pilha A**\n\n"
                            f"- Primeira camada esteirada como base;\n"
                            f"- Demais camadas empilhadas por cima como estratos puros;\n"
                            f"- Altura ocupada: **{pile_a['altura_efetiva_m']:.2f} m**."
                        )
                        st.dataframe(
                            df_a.style.format(
                                {
                                    "ton_calculada": "{:,.0f}",
                                    "densidade": "{:.2f}",
                                    "volume_m3": "{:,.0f}",
                                    "y_base_m": "{:.2f}",
                                    "y_topo_m": "{:.2f}",
                                    "espessura_m": "{:.2f}",
                                    "largura_base_m": "{:.2f}",
                                    "largura_topo_m": "{:.2f}",
                                }
                            ),
                            use_container_width=True,
                        )

                    else:
                        abas = st.tabs(["Lifts", "Sublifts", "Receita por camada"])

                        with abas[0]:
                            df_show_lifts = pile_b["df_lifts"][
                                [
                                    "lift_global", "lift_nome", "camada", "lift_seq", "y_base_m",
                                    "y_topo_m", "altura_lift_m", "largura_base_m", "largura_topo_m",
                                    "volume_lift_m3", "massa_lift_t",
                                ]
                            ].copy()
                            st.dataframe(
                                df_show_lifts.style.format(
                                    {
                                        "y_base_m": "{:.2f}", "y_topo_m": "{:.2f}", "altura_lift_m": "{:.2f}",
                                        "largura_base_m": "{:.2f}", "largura_topo_m": "{:.2f}",
                                        "volume_lift_m3": "{:,.0f}", "massa_lift_t": "{:,.0f}",
                                    }
                                ),
                                use_container_width=True,
                            )

                        with abas[1]:
                            df_show_sublifts = pile_b["df_sublifts"][
                                [
                                    "lift_global", "lift_nome", "sublift_nome", "camada", "sublift_seq",
                                    "y_base_m", "y_topo_m", "altura_sublift_m", "largura_base_m",
                                    "largura_topo_m", "volume_sublift_m3", "massa_sublift_t",
                                ]
                            ].copy()
                            st.dataframe(
                                df_show_sublifts.style.format(
                                    {
                                        "y_base_m": "{:.2f}", "y_topo_m": "{:.2f}", "altura_sublift_m": "{:.2f}",
                                        "largura_base_m": "{:.2f}", "largura_topo_m": "{:.2f}",
                                        "volume_sublift_m3": "{:,.0f}", "massa_sublift_t": "{:,.0f}",
                                    }
                                ),
                                use_container_width=True,
                            )

                        with abas[2]:
                            # CORREÇÃO APLICADA AQUI
                            st.dataframe(
                                pile_b["df_layers"][["camada", "ton_calculada", "volume_m3", "frac_massa_%", "frac_volume_%"]].style.format(
                                    {
                                        "ton_calculada": "{:,.0f}",
                                        "volume_m3": "{:,.0f}",
                                        "frac_massa_%": "{:.4f}",
                                        "frac_volume_%": "{:.4f}",
                                    }
                                ),
                                use_container_width=True,
                            )

                        st.info(
                            f"**Pilha B**\n\n"
                            f"- Cada camada geológica da Pilha A é subdividida em lifts sucessivos;\n"
                            f"- Cada lift pode ser subdividido em sublifts;\n"
                            f"- Altura máxima de lift: **{altura_lift:.2f} m**;\n"
                            f"- Altura máxima de sublift: **{altura_sublift:.2f} m**;\n"
                            f"- Altura ocupada: **{pile_b['altura_efetiva_m']:.2f} m**."
                        )

                with col_grafico:
                    if modo_construtivo == "Pilha A - Estratos por camada":
                        fig = build_pile_a_figure(pile_a["df_camadas"], larg_base, alt_max, angulo_rep)
                    else:
                        fig = build_pile_b_figure(pile_b, larg_base, alt_max, angulo_rep)
                    st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Erro na execução matemática: {e}")
