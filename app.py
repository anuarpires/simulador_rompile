
import math
from io import BytesIO
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import linprog

# ==========================================
# 1. CONFIGURAÇÃO DA PÁGINA E ESTILO
# ==========================================
st.set_page_config(page_title="Otimizador de Pilha ROM - Copelmi", layout="wide")
st.title("Otimizador Avançado de Blending - Pilha ROM")
st.markdown(
    "Motor de otimização por Programação Linear para maximizar o fechamento da pilha "
    "respeitando restrições. Agora com dois modos de modelagem construtiva: "
    "**espessura retroativa** e **lifts/sublifts com repetição da receita**."
)

# Ordem visual das camadas, do fundo para o topo
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

# ==========================================
# 2. FUNÇÕES MATEMÁTICAS E GEOMÉTRICAS
# ==========================================
def zscore(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return np.zeros_like(x) if s == 0 else (x - x.mean()) / s


def build_linear_problem(df: pd.DataFrame, specs: dict, target_mass: float, volume_max_m3: float | None):
    """
    Traduz o problema de blending para o formato do solver linear.
    """
    n = len(df)

    vm = df["vm"].to_numpy(dtype=float)
    ts = df["ts"].to_numpy(dtype=float)
    cinza = df["cinza"].to_numpy(dtype=float)
    rho = df["densidade"].to_numpy(dtype=float)

    bounds = [(0.0, float(row["ton_report"])) for _, row in df.iterrows()]

    A_ub, b_ub = [], []
    A_eq, b_eq = [], []

    # Massa total exata
    A_eq.append(np.ones(n))
    b_eq.append(float(target_mass))

    # Limites de qualidade
    if specs.get("ts_max") is not None:
        A_ub.append(ts - float(specs["ts_max"]))
        b_ub.append(0.0)

    if specs.get("cinza_max") is not None:
        A_ub.append(cinza - float(specs["cinza_max"]))
        b_ub.append(0.0)

    if specs.get("vm_min") is not None:
        A_ub.append(float(specs["vm_min"]) - vm)
        b_ub.append(0.0)

    # Limite volumétrico do pátio
    if volume_max_m3 is not None:
        A_ub.append(1.0 / rho)
        b_ub.append(float(volume_max_m3))

    # Função objetivo: aumentar VM e reduzir TS + cinza
    c = (-1.0 * zscore(vm) + 1.0 * zscore(ts) + 1.0 * zscore(cinza))
    return c, A_ub, b_ub, A_eq, b_eq, bounds


def width_at_height(y: float, larg_base: float, angulo: float) -> float:
    tanv = math.tan(math.radians(angulo))
    if tanv <= 0:
        return larg_base
    return max(0.0, larg_base - 2.0 * (y / tanv))


def cross_section_area_at_height(y: float, larg_base: float, angulo: float) -> float:
    return width_at_height(y, larg_base, angulo)


def cross_section_area_up_to(h: float, larg_base: float, angulo: float) -> float:
    """
    Área da seção trapezoidal acumulada até a altura h.
    """
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
    """
    Encontra a altura efetiva correspondente a um volume desejado dentro do invólucro trapezoidal.
    """
    volume_max = longitudinal_trapezoid_volume(comp, larg_base, alt_max, angulo)
    if volume_target <= 0:
        return 0.0
    if volume_target >= volume_max:
        return alt_max

    lo, hi = 0.0, alt_max
    for _ in range(60):
        mid = (lo + hi) / 2.0
        v_mid = longitudinal_trapezoid_volume(comp, larg_base, mid, angulo)
        if v_mid < volume_target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


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


def prepare_layer_totals(df_res: pd.DataFrame, comp_base: float, larg_base: float) -> pd.DataFrame:
    """
    Modo 1: espessura retroativa com base retangular.
    """
    area_base = comp_base * larg_base
    df_mode = df_res.copy()
    df_mode["volume_m3"] = df_mode["ton_calculada"] / df_mode["densidade"]
    df_mode["espessura_m"] = df_mode["volume_m3"] / area_base
    df_mode["ordem_plot"] = df_mode["camada"].map(ORDEM_CONSTRUCAO).fillna(99)
    return df_mode.sort_values("ordem_plot").reset_index(drop=True)


def prepare_lift_model(
    df_res: pd.DataFrame,
    comp_base: float,
    larg_base: float,
    alt_max: float,
    angulo_rep: float,
    altura_lift: float,
    altura_sublift: float,
):
    """
    Modo 2: constrói lifts/sublifts repetindo a mesma receita em cada lift.
    A massa de cada lift/sublift é proporcional ao volume geométrico da fatia.
    """
    metrics = calculate_quality_metrics(df_res)
    massa_total = metrics["massa_final"]
    volume_total = metrics["volume_total"]

    altura_efetiva = solve_height_for_volume(volume_total, comp_base, larg_base, alt_max, angulo_rep)

    df_layers = df_res.copy()
    df_layers["volume_m3"] = df_layers["ton_calculada"] / df_layers["densidade"]
    df_layers["mass_frac"] = df_layers["ton_calculada"] / massa_total
    df_layers["vol_frac"] = df_layers["volume_m3"] / volume_total
    df_layers["ordem_plot"] = df_layers["camada"].map(ORDEM_CONSTRUCAO).fillna(99)
    df_layers = df_layers.sort_values("ordem_plot").reset_index(drop=True)

    # --------------------
    # Tabela de lifts
    # --------------------
    lift_rows = []
    y0 = 0.0
    lift_id = 1
    while y0 < altura_efetiva - 1e-9:
        y1 = min(altura_efetiva, y0 + altura_lift)
        vol_lift = volume_between_heights(y0, y1, comp_base, larg_base, angulo_rep)
        massa_lift = massa_total * (vol_lift / volume_total) if volume_total > 0 else 0.0

        row = {
            "lift": lift_id,
            "y_base_m": y0,
            "y_topo_m": y1,
            "altura_lift_m": y1 - y0,
            "largura_base_m": width_at_height(y0, larg_base, angulo_rep),
            "largura_topo_m": width_at_height(y1, larg_base, angulo_rep),
            "volume_lift_m3": vol_lift,
            "massa_lift_t": massa_lift,
        }
        for _, layer in df_layers.iterrows():
            row[f"{layer['camada']}_t"] = massa_lift * layer["mass_frac"]
        lift_rows.append(row)

        y0 = y1
        lift_id += 1

    df_lifts = pd.DataFrame(lift_rows)

    # --------------------
    # Tabela de sublifts
    # --------------------
    sublift_rows = []
    draw_slices = []
    for _, lift in df_lifts.iterrows():
        z0 = float(lift["y_base_m"])
        sub_id = 1
        while z0 < float(lift["y_topo_m"]) - 1e-9:
            z1 = min(float(lift["y_topo_m"]), z0 + altura_sublift)
            vol_sub = volume_between_heights(z0, z1, comp_base, larg_base, angulo_rep)
            massa_sub = massa_total * (vol_sub / volume_total) if volume_total > 0 else 0.0

            row = {
                "lift": int(lift["lift"]),
                "sublift": sub_id,
                "y_base_m": z0,
                "y_topo_m": z1,
                "altura_sublift_m": z1 - z0,
                "largura_base_m": width_at_height(z0, larg_base, angulo_rep),
                "largura_topo_m": width_at_height(z1, larg_base, angulo_rep),
                "volume_sublift_m3": vol_sub,
                "massa_sublift_t": massa_sub,
            }
            for _, layer in df_layers.iterrows():
                row[f"{layer['camada']}_t"] = massa_sub * layer["mass_frac"]
            sublift_rows.append(row)

            draw_slices.append(
                {
                    "lift": int(lift["lift"]),
                    "sublift": sub_id,
                    "y0": z0,
                    "y1": z1,
                    "altura": z1 - z0,
                }
            )

            z0 = z1
            sub_id += 1

    df_sublifts = pd.DataFrame(sublift_rows)

    return {
        "df_layers": df_layers,
        "df_lifts": df_lifts,
        "df_sublifts": df_sublifts,
        "altura_efetiva_m": altura_efetiva,
        "draw_slices": draw_slices,
        "metrics": metrics,
    }


def build_retroactive_figure(df_mode: pd.DataFrame, larg_base: float, alt_max: float, angulo_rep: float) -> go.Figure:
    fig = go.Figure()
    y_atual = 0.0
    largura_atual = larg_base
    centro_x = larg_base / 2.0
    tan_alpha = math.tan(math.radians(angulo_rep))

    for _, row in df_mode.iterrows():
        camada = row["camada"]
        espessura = float(row["espessura_m"])
        y_topo = y_atual + espessura
        largura_topo = max(0.0, largura_atual - 2.0 * (espessura / tan_alpha)) if tan_alpha > 0 else largura_atual

        x_coords = [
            centro_x - largura_atual / 2.0,
            centro_x + largura_atual / 2.0,
            centro_x + largura_topo / 2.0,
            centro_x - largura_topo / 2.0,
            centro_x - largura_atual / 2.0,
        ]
        y_coords = [y_atual, y_atual, y_topo, y_topo, y_atual]

        fig.add_trace(
            go.Scatter(
                x=x_coords,
                y=y_coords,
                fill="toself",
                name=f"{camada} ({espessura:.2f} m)",
                mode="lines",
                line=dict(color="white", width=1),
                fillcolor=CORES_CAMADAS.get(camada, "#777777"),
                text=(
                    f"Camada: {camada}"
                    f"<br>Espessura equivalente: {espessura:.2f} m"
                    f"<br>Massa: {row['ton_calculada']:,.0f} t"
                ),
                hoverinfo="text",
            )
        )

        y_atual = y_topo
        largura_atual = largura_topo

    fig.add_hline(y=alt_max, line_dash="dash", line_color="red", annotation_text="Altura Máx")

    fig.update_layout(
        title="Seção Transversal - Modo Espessura Retroativa",
        xaxis_title="Largura da Pilha (m)",
        yaxis_title="Altura (m)",
        yaxis=dict(range=[0, max(alt_max + 0.5, y_atual + 0.5)]),
        xaxis=dict(range=[-5, larg_base + 5], tick0=0, dtick=10),
        margin=dict(l=20, r=20, t=50, b=20),
        template="plotly_white",
        hovermode="closest",
    )
    return fig


def build_lift_figure(model: dict, larg_base: float, alt_max: float, angulo_rep: float) -> go.Figure:
    """
    Desenha a pilha em bandas repetidas por sublift, respeitando a mesma receita em cada fatia.
    """
    df_layers = model["df_layers"]
    draw_slices = model["draw_slices"]
    altura_efetiva = model["altura_efetiva_m"]

    fig = go.Figure()
    centro_x = larg_base / 2.0

    # As frações volumétricas determinam as espessuras relativas dentro de cada lift/sublift
    vol_fracs = df_layers[["camada", "vol_frac"]].copy()

    for s in draw_slices:
        y_cursor = float(s["y0"])
        slice_height = float(s["altura"])

        for _, layer in vol_fracs.iterrows():
            camada = layer["camada"]
            frac = float(layer["vol_frac"])
            if frac <= 0:
                continue

            band_h = slice_height * frac
            y_top = y_cursor + band_h

            larg_inf = width_at_height(y_cursor, larg_base, angulo_rep)
            larg_sup = width_at_height(y_top, larg_base, angulo_rep)

            x_coords = [
                centro_x - larg_inf / 2.0,
                centro_x + larg_inf / 2.0,
                centro_x + larg_sup / 2.0,
                centro_x - larg_sup / 2.0,
                centro_x - larg_inf / 2.0,
            ]
            y_coords = [y_cursor, y_cursor, y_top, y_top, y_cursor]

            mask = model["df_sublifts"]["lift"].eq(s["lift"]) & model["df_sublifts"]["sublift"].eq(s["sublift"])
            massa_layer = None
            col_name = f"{camada}_t"
            if col_name in model["df_sublifts"].columns and mask.any():
                massa_layer = float(model["df_sublifts"].loc[mask, col_name].iloc[0])

            fig.add_trace(
                go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    fill="toself",
                    name=camada,
                    showlegend=False,
                    mode="lines",
                    line=dict(color="white", width=0.8),
                    fillcolor=CORES_CAMADAS.get(camada, "#777777"),
                    text=(
                        f"Lift {s['lift']} | Sublift {s['sublift']}"
                        f"<br>Camada: {camada}"
                        f"<br>Faixa vertical: {band_h:.3f} m"
                        + (f"<br>Massa nesta sublift: {massa_layer:,.0f} t" if massa_layer is not None else "")
                    ),
                    hoverinfo="text",
                )
            )
            y_cursor = y_top

    # Legenda manual por camada
    for camada, cor in CORES_CAMADAS.items():
        if camada in df_layers["camada"].tolist():
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=12, color=cor),
                    legendgroup=camada,
                    showlegend=True,
                    name=camada,
                )
            )

    fig.add_hline(y=alt_max, line_dash="dash", line_color="red", annotation_text="Altura Máx")

    fig.update_layout(
        title="Seção Transversal - Modo Lifts/Sublifts com Receita Repetida",
        xaxis_title="Largura da Pilha (m)",
        yaxis_title="Altura (m)",
        yaxis=dict(range=[0, max(alt_max + 0.5, altura_efetiva + 0.5)]),
        xaxis=dict(range=[-5, larg_base + 5], tick0=0, dtick=10),
        margin=dict(l=20, r=20, t=50, b=20),
        template="plotly_white",
        hovermode="closest",
        legend_title="Camadas",
    )
    return fig


def format_layer_columns_for_display(df: pd.DataFrame) -> pd.DataFrame:
    ordered = [c for c in sorted([k for k in df.columns if k.endswith("_t")], key=lambda x: ORDEM_CONSTRUCAO.get(x[:-2], 999))]
    return df[ordered]




def build_export_bundle(
    modo_construtivo: str,
    df_res: pd.DataFrame,
    metrics: dict,
    *,
    comp_base: float,
    larg_base: float,
    alt_max: float,
    angulo_rep: float,
    alvo_massa: float,
    vm_min: float,
    ts_max: float,
    cinza_max: float,
    vol_max: float,
    altura_efetiva: float,
    df_mode: pd.DataFrame | None = None,
    model: dict | None = None,
) -> dict:
    """
    Monta os dataframes e arquivos de exportação conforme o modo construtivo selecionado.
    """
    df_total = df_res.copy()
    df_total["volume_m3"] = df_total["ton_calculada"] / df_total["densidade"]
    df_total["frac_massa_%"] = 100.0 * df_total["ton_calculada"] / metrics["massa_final"]
    df_total["frac_volume_%"] = 100.0 * df_total["volume_m3"] / metrics["volume_total"]
    df_total["ordem_plot"] = df_total["camada"].map(ORDEM_CONSTRUCAO).fillna(99)
    df_total = df_total.sort_values("ordem_plot").reset_index(drop=True)

    df_resumo = pd.DataFrame(
        [
            {"item": "Modo construtivo", "valor": modo_construtivo},
            {"item": "Massa alvo (t)", "valor": alvo_massa},
            {"item": "Massa final (t)", "valor": metrics["massa_final"]},
            {"item": "VM final (%)", "valor": metrics["vm_final"]},
            {"item": "TS final (%)", "valor": metrics["ts_final"]},
            {"item": "Cinza/CBS final", "valor": metrics["cinza_final"]},
            {"item": "Volume usado (m3)", "valor": metrics["volume_total"]},
            {"item": "Densidade aparente equivalente (t/m3)", "valor": metrics["rho_aparente"]},
            {"item": "Comprimento base (m)", "valor": comp_base},
            {"item": "Largura base (m)", "valor": larg_base},
            {"item": "Altura máxima do invólucro (m)", "valor": alt_max},
            {"item": "Ângulo de repouso (graus)", "valor": angulo_rep},
            {"item": "Altura efetiva ocupada (m)", "valor": altura_efetiva},
            {"item": "Volume máximo do pátio (m3)", "valor": vol_max},
            {"item": "VM mínimo especificado (%)", "valor": vm_min},
            {"item": "TS máximo especificado (%)", "valor": ts_max},
            {"item": "Cinza/CBS máximo especificado", "valor": cinza_max},
        ]
    )

    sheets = {
        "resumo": df_resumo,
        "composicao_total": df_total[["camada", "ton_calculada", "volume_m3", "frac_massa_%", "frac_volume_%"]],
    }

    csvs = {
        "resumo.csv": df_resumo.to_csv(index=False).encode("utf-8-sig"),
        "composicao_total.csv": sheets["composicao_total"].to_csv(index=False).encode("utf-8-sig"),
    }

    if modo_construtivo == "Espessura retroativa" and df_mode is not None:
        df_geo = df_mode[["camada", "ton_calculada", "volume_m3", "espessura_m"]].copy()
        sheets["geometria_retroativa"] = df_geo
        csvs["geometria_retroativa.csv"] = df_geo.to_csv(index=False).encode("utf-8-sig")

    if modo_construtivo == "Lifts/Sublifts com receita repetida" and model is not None:
        df_lifts = model["df_lifts"].copy()
        df_sublifts = model["df_sublifts"].copy()

        lift_cols = ["lift", "y_base_m", "y_topo_m", "altura_lift_m", "largura_base_m", "largura_topo_m", "volume_lift_m3", "massa_lift_t"]
        lift_cols += [c for c in df_lifts.columns if c.endswith("_t")]
        sub_cols = ["lift", "sublift", "y_base_m", "y_topo_m", "altura_sublift_m", "largura_base_m", "largura_topo_m", "volume_sublift_m3", "massa_sublift_t"]
        sub_cols += [c for c in df_sublifts.columns if c.endswith("_t")]

        df_receita_lifts = df_lifts[lift_cols]
        df_receita_sublifts = df_sublifts[sub_cols]

        sheets["lifts"] = df_receita_lifts
        sheets["sublifts"] = df_receita_sublifts

        csvs["lifts.csv"] = df_receita_lifts.to_csv(index=False).encode("utf-8-sig")
        csvs["sublifts.csv"] = df_receita_sublifts.to_csv(index=False).encode("utf-8-sig")

    xlsx_buffer = BytesIO()
    with pd.ExcelWriter(xlsx_buffer, engine="openpyxl") as writer:
        for sheet_name, df_sheet in sheets.items():
            safe_name = sheet_name[:31]
            df_sheet.to_excel(writer, index=False, sheet_name=safe_name)
            ws = writer.book[safe_name]
            ws.freeze_panes = "A2"
            for col_cells in ws.columns:
                max_len = max(len(str(cell.value)) if cell.value is not None else 0 for cell in col_cells)
                ws.column_dimensions[col_cells[0].column_letter].width = min(max(max_len + 2, 12), 28)

    return {
        "sheets": sheets,
        "csvs": csvs,
        "xlsx_bytes": xlsx_buffer.getvalue(),
    }


def render_export_section(export_bundle: dict, modo_construtivo: str):
    st.divider()
    st.subheader("Exportação")

    col_xlsx, col_csv = st.columns([1.1, 1.4])

    with col_xlsx:
        st.download_button(
            label="Baixar relatório em Excel (.xlsx)",
            data=export_bundle["xlsx_bytes"],
            file_name="pilha_rom_otimizada.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    with col_csv:
        opcoes_csv = list(export_bundle["csvs"].keys())
        arquivo_csv = st.selectbox(
            "Escolha uma tabela para exportar em CSV",
            options=opcoes_csv,
            index=0,
        )
        st.download_button(
            label=f"Baixar {arquivo_csv}",
            data=export_bundle["csvs"][arquivo_csv],
            file_name=arquivo_csv,
            mime="text/csv",
            use_container_width=True,
        )

    st.caption(
        "O Excel inclui resumo, composição total e as tabelas coerentes com o modo construtivo selecionado."
        + (" No modo de lifts, inclui também a receita executiva por lift e por sublift." if modo_construtivo == "Lifts/Sublifts com receita repetida" else "")
    )

# ==========================================
# 3. INTERFACE DE USUÁRIO (SIDEBAR)
# ==========================================
st.sidebar.header("Parâmetros da Pilha")
alvo_massa = st.sidebar.number_input("Massa Alvo (t)", value=50000.0, step=1000.0)

st.sidebar.header("Geometria")
comp_base = st.sidebar.number_input("Comprimento Base (m)", value=120.0)
larg_base = st.sidebar.number_input("Largura Base (m)", value=70.0)
alt_max = st.sidebar.number_input("Altura Máxima (m)", value=5.0)
angulo_rep = st.sidebar.number_input("Ângulo Repouso (Graus)", value=37.0)

st.sidebar.header("Modelo Construtivo")
modo_construtivo = st.sidebar.radio(
    "Escolha como representar a pilha",
    options=["Espessura retroativa", "Lifts/Sublifts com receita repetida"],
    help=(
        "Espessura retroativa: converte massa em espessura equivalente sobre a base. "
        "Lifts/Sublifts: distribui a mesma receita em cada lift e sublift."
    ),
)

altura_lift = 1.0
altura_sublift = 0.5
if modo_construtivo == "Lifts/Sublifts com receita repetida":
    altura_lift = st.sidebar.number_input("Altura do lift (m)", min_value=0.1, value=1.0, step=0.1)
    altura_sublift = st.sidebar.number_input("Altura do sublift (m)", min_value=0.1, value=0.5, step=0.1)

    if altura_sublift > altura_lift:
        st.sidebar.warning("A altura do sublift foi limitada à altura do lift.")
        altura_sublift = altura_lift

st.sidebar.header("Restrições da Usina")
vm_min = st.sidebar.number_input("VM Mínimo (%)", value=19.30)
ts_max = st.sidebar.number_input("TS Máximo (%)", value=2.20)
cinza_max = st.sidebar.number_input("Cinza/CBS Máximo", value=57.17)

# ==========================================
# 4. TABELA DE DADOS INTERATIVA
# ==========================================
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

# ==========================================
# 5. BOTÃO DE OTIMIZAÇÃO E RESULTADOS
# ==========================================
if st.button("Rodar Solver de Otimização", type="primary"):
    if df_valido.empty:
        st.error("Nenhuma camada com tonelagem válida para otimizar.")
    else:
        try:
            vol_max = longitudinal_trapezoid_volume(comp_base, larg_base, alt_max, angulo_rep)
            specs = {"vm_min": vm_min, "ts_max": ts_max, "cinza_max": cinza_max}
            c, A_ub, b_ub, A_eq, b_eq, bounds = build_linear_problem(df_valido, specs, alvo_massa, vol_max)

            res = linprog(
                c,
                A_ub=A_ub if A_ub else None,
                b_ub=b_ub if b_ub else None,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method="highs",
            )

            if not res.success:
                st.error(f"O Solver não encontrou uma solução viável matemática para essas restrições: {res.message}")
            else:
                st.success("Solução Otimizada Encontrada!")

                df_valido["ton_calculada"] = res.x
                df_res = df_valido[df_valido["ton_calculada"] > 1e-6].copy()

                metrics = calculate_quality_metrics(df_res)
                altura_efetiva = solve_height_for_volume(
                    metrics["volume_total"], comp_base, larg_base, alt_max, angulo_rep
                )

                # ----------------------------------------
                # QUALIDADE
                # ----------------------------------------
                st.divider()
                st.subheader("Resultados de Qualidade")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric(
                    "Massa Total",
                    f"{metrics['massa_final']:,.0f} t",
                    delta=f"{metrics['massa_final'] - alvo_massa:,.0f} vs Alvo",
                )

                cor_vm = "normal" if metrics["vm_final"] >= vm_min else "inverse"
                c2.metric("VM Final", f"{metrics['vm_final']:.2f}%", delta=f"Mínimo: {vm_min}", delta_color=cor_vm)

                cor_ts = "normal" if metrics["ts_final"] <= ts_max else "inverse"
                c3.metric("TS Final", f"{metrics['ts_final']:.2f}%", delta=f"Máximo: {ts_max}", delta_color=cor_ts)

                cor_cz = "normal" if metrics["cinza_final"] <= cinza_max else "inverse"
                c4.metric(
                    "Cinza/CBS Final",
                    f"{metrics['cinza_final']:.2f}",
                    delta=f"Máximo: {cinza_max}",
                    delta_color=cor_cz,
                )

                st.caption(
                    f"Volume usado: {metrics['volume_total']:,.0f} m³ | "
                    f"Densidade aparente equivalente: {metrics['rho_aparente']:.3f} t/m³ | "
                    f"Altura efetiva no invólucro trapezoidal: {altura_efetiva:.2f} m | "
                    f"Volume máximo do pátio: {vol_max:,.0f} m³"
                )

                # ----------------------------------------
                # COMPOSIÇÃO E GEOMETRIA
                # ----------------------------------------
                st.divider()
                st.subheader("Composição, Geometria e Visualização")

                col_tabela, col_grafico = st.columns([1.15, 1.2])

                with col_tabela:
                    st.markdown(f"**Modo selecionado:** {modo_construtivo}")

                    abas = st.tabs(
                        ["Composição Total"]
                        + (["Lifts", "Sublifts"] if modo_construtivo == "Lifts/Sublifts com receita repetida" else [])
                    )

                    with abas[0]:
                        df_total = df_res.copy()
                        df_total["volume_m3"] = df_total["ton_calculada"] / df_total["densidade"]
                        df_total["frac_massa_%"] = 100.0 * df_total["ton_calculada"] / metrics["massa_final"]
                        df_total["frac_volume_%"] = 100.0 * df_total["volume_m3"] / metrics["volume_total"]
                        df_total["ordem_plot"] = df_total["camada"].map(ORDEM_CONSTRUCAO).fillna(99)
                        df_total = df_total.sort_values("ordem_plot")
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

                    if modo_construtivo == "Espessura retroativa":
                        df_mode = prepare_layer_totals(df_res, comp_base, larg_base)
                        st.info(
                            f"**Geometria retroativa:** base de {larg_base:.1f} m x {comp_base:.1f} m\n\n"
                            f"**Altura total equivalente (retroativa):** {df_mode['espessura_m'].sum():.2f} m"
                        )
                        st.dataframe(
                            df_mode[["camada", "ton_calculada", "volume_m3", "espessura_m"]].style.format(
                                {
                                    "ton_calculada": "{:,.0f}",
                                    "volume_m3": "{:,.0f}",
                                    "espessura_m": "{:.3f}",
                                }
                            ),
                            use_container_width=True,
                        )
                    else:
                        model = prepare_lift_model(
                            df_res=df_res,
                            comp_base=comp_base,
                            larg_base=larg_base,
                            alt_max=alt_max,
                            angulo_rep=angulo_rep,
                            altura_lift=altura_lift,
                            altura_sublift=altura_sublift,
                        )

                        with abas[1]:
                            df_lifts_view = model["df_lifts"].copy()
                            st.dataframe(
                                df_lifts_view.style.format(
                                    {
                                        "y_base_m": "{:.2f}",
                                        "y_topo_m": "{:.2f}",
                                        "altura_lift_m": "{:.2f}",
                                        "largura_base_m": "{:.2f}",
                                        "largura_topo_m": "{:.2f}",
                                        "volume_lift_m3": "{:,.0f}",
                                        "massa_lift_t": "{:,.0f}",
                                        **{c: "{:,.0f}" for c in df_lifts_view.columns if c.endswith("_t")},
                                    }
                                ),
                                use_container_width=True,
                            )

                        with abas[2]:
                            df_sub_view = model["df_sublifts"].copy()
                            st.dataframe(
                                df_sub_view.style.format(
                                    {
                                        "y_base_m": "{:.2f}",
                                        "y_topo_m": "{:.2f}",
                                        "altura_sublift_m": "{:.2f}",
                                        "largura_base_m": "{:.2f}",
                                        "largura_topo_m": "{:.2f}",
                                        "volume_sublift_m3": "{:,.0f}",
                                        "massa_sublift_t": "{:,.0f}",
                                        **{c: "{:,.0f}" for c in df_sub_view.columns if c.endswith("_t")},
                                    }
                                ),
                                use_container_width=True,
                            )

                        st.info(
                            f"**Geometria por lifts/sublifts:** invólucro de {larg_base:.1f} m x {comp_base:.1f} m x {alt_max:.1f} m\n\n"
                            f"**Altura efetiva ocupada:** {model['altura_efetiva_m']:.2f} m\n\n"
                            f"**Receita repetida:** a mesma composição mássica é redistribuída em cada lift "
                            f"e em cada sublift, proporcionalmente ao volume da fatia geométrica."
                        )

                with col_grafico:
                    if modo_construtivo == "Espessura retroativa":
                        fig = build_retroactive_figure(df_mode, larg_base, alt_max, angulo_rep)
                    else:
                        fig = build_lift_figure(model, larg_base, alt_max, angulo_rep)
                    st.plotly_chart(fig, use_container_width=True)

                export_bundle = build_export_bundle(
                    modo_construtivo=modo_construtivo,
                    df_res=df_res,
                    metrics=metrics,
                    comp_base=comp_base,
                    larg_base=larg_base,
                    alt_max=alt_max,
                    angulo_rep=angulo_rep,
                    alvo_massa=alvo_massa,
                    vm_min=vm_min,
                    ts_max=ts_max,
                    cinza_max=cinza_max,
                    vol_max=vol_max,
                    altura_efetiva=altura_efetiva,
                    df_mode=df_mode if modo_construtivo == "Espessura retroativa" else None,
                    model=model if modo_construtivo == "Lifts/Sublifts com receita repetida" else None,
                )
                render_export_section(export_bundle, modo_construtivo)

        except Exception as e:
            st.error(f"Erro na execução matemática: {e}")
