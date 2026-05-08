
import math
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# 1) CONFIGURAÇÃO GERAL DA APLICAÇÃO
# ============================================================
st.set_page_config(page_title="Conferência de Receita - Pilha ROM", layout="wide")
st.title("Conferência de Receita por Média Ponderada - Pilha ROM")

st.markdown(
    """
    Este aplicativo **não otimiza** a receita.

    Ele foi configurado para:
    - receber os **dados editáveis imputados** pelo usuário;
    - calcular a **média ponderada de qualidade** a partir da **massa informada para a receita**;
    - comparar o resultado obtido com as **metas da usina**;
    - verificar se a **massa/volume da receita** cabe no **volume máximo de projeto da pilha**;
    - representar a pilha em dois modos:
        - **Pilha A - Estratos por camada**
        - **Pilha B - Lift / Sublifting**
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
# 2) FUNÇÕES DE SUPORTE
# ============================================================
def fmt_br(value: float, ndigits: int = 2) -> str:
    if pd.isna(value):
        return ""
    txt = f"{value:,.{ndigits}f}"
    return txt.replace(",", "X").replace(".", ",").replace("X", ".")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    original = list(df.columns)
    normalized = []
    for col in original:
        c = str(col).strip().lower()
        c = (
            c.replace("ã", "a")
            .replace("á", "a")
            .replace("â", "a")
            .replace("é", "e")
            .replace("ê", "e")
            .replace("í", "i")
            .replace("ó", "o")
            .replace("ô", "o")
            .replace("õ", "o")
            .replace("ú", "u")
            .replace("ç", "c")
        )
        c = c.replace("%", "").replace("(", "").replace(")", "")
        c = c.replace("/", "_").replace("-", "_").replace(" ", "_")
        normalized.append(c)

    df.columns = normalized

    rename_map = {}
    aliases = {
        "camada": ["camada", "layer"],
        "ton_disponivel": [
            "ton_disponivel", "ton_report", "tons", "ton", "tonelagem", "massa_disponivel", "rom_t"
        ],
        "ton_receita": [
            "ton_receita", "ton_blend", "ton_usada", "ton_utilizada", "massa_receita", "massa_usada"
        ],
        "vm": ["vm", "materia_volatil", "volatile_matter"],
        "ts": ["ts", "enxofre", "sulfur", "teor_de_enxofre"],
        "cinza": ["cinza", "cbs", "ash"],
        "densidade": ["densidade", "density", "rho"],
    }

    for target, poss in aliases.items():
        for p in poss:
            if p in df.columns:
                rename_map[p] = target
                break

    df = df.rename(columns=rename_map)

    required_cols = ["camada", "vm", "ts", "cinza", "densidade"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {missing}")

    if "ton_disponivel" not in df.columns:
        df["ton_disponivel"] = np.nan
    if "ton_receita" not in df.columns:
        if "ton_disponivel" in df.columns:
            df["ton_receita"] = df["ton_disponivel"].fillna(0.0)
        else:
            df["ton_receita"] = 0.0

    ordered_cols = ["camada", "ton_disponivel", "ton_receita", "vm", "ts", "cinza", "densidade"]
    others = [c for c in df.columns if c not in ordered_cols]
    return df[ordered_cols + others]


def validate_recipe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    for c in ["ton_disponivel", "ton_receita", "vm", "ts", "cinza", "densidade"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["camada"] = df["camada"].astype(str).str.strip()
    df = df[df["camada"] != ""].copy()

    discarded = []

    invalid_quality = df[
        df[["vm", "ts", "cinza", "densidade"]].isna().any(axis=1) | (df["densidade"] <= 0)
    ].copy()
    if not invalid_quality.empty:
        invalid_quality["motivo_descarte"] = "Dados de qualidade/densidade inválidos"
        discarded.append(invalid_quality)

    df = df[~df.index.isin(invalid_quality.index)].copy()

    zero_or_negative = df[df["ton_receita"].fillna(0) <= 0].copy()
    if not zero_or_negative.empty:
        zero_or_negative["motivo_descarte"] = "Tonelagem de receita não informada ou <= 0"
        discarded.append(zero_or_negative)

    df = df[df["ton_receita"].fillna(0) > 0].copy()

    disponibilidade_excedida = df[
        df["ton_disponivel"].notna() & (df["ton_receita"] > df["ton_disponivel"] + 1e-9)
    ].copy()
    if not disponibilidade_excedida.empty:
        disponibilidade_excedida["motivo_descarte"] = "Receita excede a disponibilidade informada"
        discarded.append(disponibilidade_excedida)

    discarded_df = pd.concat(discarded, ignore_index=True) if discarded else pd.DataFrame()
    return df.reset_index(drop=True), discarded_df.reset_index(drop=True)


def calculate_quality_metrics(df_res: pd.DataFrame) -> dict:
    massa_final = float(df_res["ton_receita"].sum())
    volume_total = float((df_res["ton_receita"] / df_res["densidade"]).sum())
    vm_final = float(np.average(df_res["vm"], weights=df_res["ton_receita"]))
    ts_final = float(np.average(df_res["ts"], weights=df_res["ton_receita"]))
    cinza_final = float(np.average(df_res["cinza"], weights=df_res["ton_receita"]))
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
    df_total["volume_m3"] = df_total["ton_receita"] / df_total["densidade"]
    df_total["frac_massa_%"] = 100.0 * df_total["ton_receita"] / metrics["massa_final"]
    df_total["frac_volume_%"] = 100.0 * df_total["volume_m3"] / metrics["volume_total"]
    df_total["ordem_plot"] = df_total["camada"].map(ORDEM_CONSTRUCAO).fillna(99)
    return df_total.sort_values("ordem_plot").reset_index(drop=True)


def build_specs_comparison(metrics: dict, target_mass: float, specs: dict, volume_max_m3: float) -> pd.DataFrame:
    rows = [
        {
            "item": "Massa total da receita (t)",
            "resultado": metrics["massa_final"],
            "meta": target_mass,
            "criterio": "informativo",
            "status": "OK" if abs(metrics["massa_final"] - target_mass) < 1e-6 else "DIFERENTE DO ALVO",
            "folga": metrics["massa_final"] - target_mass,
        },
        {
            "item": "VM (%)",
            "resultado": metrics["vm_final"],
            "meta": specs["vm_min"],
            "criterio": ">=",
            "status": "ATENDE" if metrics["vm_final"] >= specs["vm_min"] else "NÃO ATENDE",
            "folga": metrics["vm_final"] - specs["vm_min"],
        },
        {
            "item": "TS (%)",
            "resultado": metrics["ts_final"],
            "meta": specs["ts_max"],
            "criterio": "<=",
            "status": "ATENDE" if metrics["ts_final"] <= specs["ts_max"] else "NÃO ATENDE",
            "folga": specs["ts_max"] - metrics["ts_final"],
        },
        {
            "item": "Cinza/CBS",
            "resultado": metrics["cinza_final"],
            "meta": specs["cinza_max"],
            "criterio": "<=",
            "status": "ATENDE" if metrics["cinza_final"] <= specs["cinza_max"] else "NÃO ATENDE",
            "folga": specs["cinza_max"] - metrics["cinza_final"],
        },
        {
            "item": "Volume da receita (m³)",
            "resultado": metrics["volume_total"],
            "meta": volume_max_m3,
            "criterio": "<=",
            "status": "ATENDE" if metrics["volume_total"] <= volume_max_m3 else "NÃO ATENDE",
            "folga": volume_max_m3 - metrics["volume_total"],
        },
    ]
    return pd.DataFrame(rows)


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
    df_total["volume_m3"] = df_total["ton_receita"] / df_total["densidade"]
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
                "ton_receita": float(row["ton_receita"]),
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

    return {"df_camadas": df_camadas, "altura_efetiva_m": altura_efetiva}


# ============================================================
# 5) PILHA B - LIFT / SUBLIFTING POR CAMADA GEOLÓGICA
# ============================================================
def prepare_pile_b_lifts(
    df_res: pd.DataFrame, comp_base: float, larg_base: float, alt_max: float, angulo_rep: float,
    altura_lift: float, altura_sublift: float
) -> dict:
    pile_a = prepare_pile_a_strata(df_res, comp_base, larg_base, alt_max, angulo_rep)

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

    return {
        "df_layers": df_layers,
        "df_camadas_base": df_camadas,
        "df_lifts": pd.DataFrame(lift_rows),
        "df_sublifts": pd.DataFrame(sublift_rows),
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
                x=x_coords,
                y=y_coords,
                fill="toself",
                mode="lines",
                line=dict(color="white", width=1),
                fillcolor=CORES_CAMADAS.get(camada, "#777777"),
                name=camada,
                legendgroup=camada,
                text=(
                    f"Camada: {camada}"
                    f"<br>Base: {fmt_br(y0)} m"
                    f"<br>Topo: {fmt_br(y1)} m"
                    f"<br>Espessura: {fmt_br(row['espessura_m'])} m"
                    f"<br>Massa da receita: {fmt_br(row['ton_receita'], 2)} t"
                    f"<br>Volume: {fmt_br(row['volume_m3'], 2)} m³"
                ),
                hoverinfo="text",
            )
        )

    fig.add_hline(y=alt_max, line_dash="dash", line_color="red", annotation_text="Altura máxima do pátio")
    fig.update_layout(
        title="Seção Transversal - Pilha A | Estratos por camada",
        xaxis_title="Largura da pilha (m)",
        yaxis_title="Altura (m)",
        xaxis=dict(range=[-5, larg_base + 5], tick0=0, dtick=10),
        yaxis=dict(range=[0, max(alt_max + 0.5, float(df_camadas["y_topo_m"].max()) + 0.5 if not df_camadas.empty else alt_max + 0.5)]),
        template="plotly_white",
        legend_title="Camadas",
        margin=dict(l=20, r=20, t=60, b=20),
        hovermode="closest",
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
                x=x_coords,
                y=y_coords,
                fill="toself",
                mode="lines",
                line=dict(color="white", width=1),
                fillcolor=CORES_CAMADAS.get(camada, "#777777"),
                name=camada,
                legendgroup=camada,
                showlegend=False,
                text=(
                    f"Lift: {row['lift_nome']}"
                    f"<br>Camada: {camada}"
                    f"<br>Base: {fmt_br(y0)} m"
                    f"<br>Topo: {fmt_br(y1)} m"
                    f"<br>Espessura do lift: {fmt_br(row['altura_lift_m'])} m"
                    f"<br>Massa do lift: {fmt_br(row['massa_lift_t'], 2)} t"
                    f"<br>Volume do lift: {fmt_br(row['volume_lift_m3'], 2)} m³"
                ),
                hoverinfo="text",
            )
        )

    for _, row in df_sublifts.iterrows():
        y1 = float(row["y_topo_m"])
        if y1 >= altura_efetiva - 1e-9:
            continue
        fig.add_hline(
            y=y1,
            line_dash="dot",
            line_color="rgba(30,30,30,0.25)",
            annotation_text=row["sublift_nome"],
            annotation_position="left",
        )

    for camada in df_lifts["camada"].drop_duplicates().tolist():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=12, color=CORES_CAMADAS.get(camada, "#777777")),
                name=camada,
                legendgroup=camada,
                showlegend=True,
            )
        )

    fig.add_hline(y=alt_max, line_dash="dash", line_color="red", annotation_text="Altura máxima do pátio")
    fig.update_layout(
        title="Seção Transversal - Pilha B | Lifts geológicos / Sublifting",
        xaxis_title="Largura da pilha (m)",
        yaxis_title="Altura (m)",
        xaxis=dict(range=[-5, larg_base + 5], tick0=0, dtick=10),
        yaxis=dict(range=[0, max(alt_max + 0.5, altura_efetiva + 0.5)]),
        template="plotly_white",
        legend_title="Camadas",
        margin=dict(l=20, r=20, t=60, b=20),
        hovermode="closest",
    )
    return fig


def export_excel(
    df_input: pd.DataFrame,
    df_usadas: pd.DataFrame,
    df_descartadas: pd.DataFrame,
    df_total: pd.DataFrame,
    df_specs: pd.DataFrame,
    pile_a: dict,
    pile_b: dict,
    geometry: dict,
) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_input.to_excel(writer, index=False, sheet_name="input_editado")
        df_usadas.to_excel(writer, index=False, sheet_name="receita_utilizada")
        if not df_descartadas.empty:
            df_descartadas.to_excel(writer, index=False, sheet_name="camadas_descartadas")
        df_total.to_excel(writer, index=False, sheet_name="composicao_blend")
        df_specs.to_excel(writer, index=False, sheet_name="comparacao_usina")
        pile_a["df_camadas"].to_excel(writer, index=False, sheet_name="pilha_a")
        pile_b["df_lifts"].to_excel(writer, index=False, sheet_name="pilha_b_lifts")
        pile_b["df_sublifts"].to_excel(writer, index=False, sheet_name="pilha_b_sublifts")
        pd.DataFrame([geometry]).to_excel(writer, index=False, sheet_name="geometria")
    output.seek(0)
    return output.getvalue()


# ============================================================
# 7) SIDEBAR - PARÂMETROS DE ENTRADA
# ============================================================
st.sidebar.header("Receita e Metas")
alvo_massa = st.sidebar.number_input("Massa alvo de referência (t)", value=50000.0, step=1000.0)

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

st.sidebar.header("Metas da Usina")
vm_min = st.sidebar.number_input("VM Mínimo (%)", value=19.30)
ts_max = st.sidebar.number_input("TS Máximo (%)", value=2.20)
cinza_max = st.sidebar.number_input("Cinza/CBS Máximo", value=57.17)

# ============================================================
# 8) ENTRADA DE DADOS
# ============================================================
st.subheader("Entrada de Dados Editável")

st.caption(
    "Edite diretamente a tabela ou envie um CSV. "
    "A coluna **ton_receita** é a massa que será efetivamente usada na média ponderada. "
    "A coluna **ton_disponivel** é opcional e serve apenas para conferir se a receita excede o disponível."
)

uploaded = st.file_uploader("Upload opcional de CSV", type=["csv"])

dados_iniciais = pd.DataFrame(
    {
        "camada": ["S6", "S5", "S4", "S3", "S2", "CS", "CI"],
        "ton_disponivel": [0.0, 4138.0, 7926.0, 16039.0, 33165.0, 136318.0, 118595.0],
        "ton_receita": [0.0, 4138.0, 7926.0, 16039.0, 21897.0, 0.0, 0.0],
        "vm": [20.00, 21.11, 19.10, 21.07, 22.37, 19.23, 17.81],
        "ts": [1.50, 0.70, 0.94, 2.09, 1.44, 1.43, 1.14],
        "cinza": [50.00, 51.31, 58.13, 47.63, 46.47, 51.86, 59.73],
        "densidade": [1.60, 1.70, 1.80, 1.60, 1.70, 1.70, 1.70],
    }
)

if uploaded is not None:
    try:
        df_base = pd.read_csv(uploaded)
        df_base = normalize_columns(df_base)
    except Exception as exc:
        st.error(f"Erro ao ler o CSV: {exc}")
        st.stop()
else:
    df_base = dados_iniciais.copy()

df_editado = st.data_editor(df_base, num_rows="dynamic", use_container_width=True)

# ============================================================
# 9) CÁLCULO DA MÉDIA PONDERADA
# ============================================================
if st.button("Calcular Média Ponderada da Receita", type="primary"):
    try:
        df_norm = normalize_columns(df_editado)
        df_receita, df_descartadas = validate_recipe(df_norm)

        if df_receita.empty:
            st.error("Nenhuma camada válida com tonelagem de receita > 0.")
            st.stop()

        vol_max = longitudinal_trapezoid_volume(comp_base, larg_base, alt_max, angulo_rep)
        specs = {"vm_min": vm_min, "ts_max": ts_max, "cinza_max": cinza_max}

        metrics = calculate_quality_metrics(df_receita)
        df_total = enrich_total_composition(df_receita, metrics)
        df_specs = build_specs_comparison(metrics, alvo_massa, specs, vol_max)

        pile_a = prepare_pile_a_strata(
            df_res=df_receita,
            comp_base=comp_base,
            larg_base=larg_base,
            alt_max=alt_max,
            angulo_rep=angulo_rep,
        )
        pile_b = prepare_pile_b_lifts(
            df_res=df_receita,
            comp_base=comp_base,
            larg_base=larg_base,
            alt_max=alt_max,
            angulo_rep=angulo_rep,
            altura_lift=altura_lift,
            altura_sublift=altura_sublift,
        )

        geometry = {
            "comprimento_base_m": comp_base,
            "largura_base_m": larg_base,
            "altura_maxima_m": alt_max,
            "angulo_repouso_graus": angulo_rep,
            "volume_maximo_patio_m3": vol_max,
            "volume_receita_m3": metrics["volume_total"],
            "altura_efetiva_receita_m": pile_a["altura_efetiva_m"],
        }

        st.divider()
        st.subheader("Resultado da Receita Informada")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Massa da receita", f"{fmt_br(metrics['massa_final'])} t", delta=f"{fmt_br(metrics['massa_final'] - alvo_massa)} vs alvo")
        c2.metric("VM obtido", f"{fmt_br(metrics['vm_final'])} %", delta=f"{fmt_br(metrics['vm_final'] - vm_min)} vs mínimo")
        c3.metric("TS obtido", f"{fmt_br(metrics['ts_final'])} %", delta=f"{fmt_br(ts_max - metrics['ts_final'])} folga")
        c4.metric("Cinza/CBS obtido", f"{fmt_br(metrics['cinza_final'])}", delta=f"{fmt_br(cinza_max - metrics['cinza_final'])} folga")

        if metrics["volume_total"] > vol_max:
            st.error(
                f"A receita excede a capacidade geométrica do pátio. "
                f"Volume da receita = {fmt_br(metrics['volume_total'])} m³ | "
                f"Volume máximo = {fmt_br(vol_max)} m³"
            )
        else:
            st.success(
                f"A receita cabe na pilha. "
                f"Volume da receita = {fmt_br(metrics['volume_total'])} m³ | "
                f"Volume máximo = {fmt_br(vol_max)} m³"
            )

        st.caption(
            f"Densidade aparente equivalente da receita: {fmt_br(metrics['rho_aparente'], 3)} t/m³ | "
            f"Altura efetiva estimada: {fmt_br(pile_a['altura_efetiva_m'])} m"
        )

        st.divider()
        st.subheader("Comparação com as Metas da Usina")
        st.dataframe(
            df_specs.style.format(
                {
                    "resultado": lambda x: fmt_br(x),
                    "meta": lambda x: fmt_br(x),
                    "folga": lambda x: fmt_br(x),
                }
            ),
            use_container_width=True,
        )

        st.divider()
        st.subheader("Composição da Receita por Média Ponderada")
        st.dataframe(
            df_total[["camada", "ton_receita", "volume_m3", "frac_massa_%", "frac_volume_%", "vm", "ts", "cinza", "densidade"]].style.format(
                {
                    "ton_receita": lambda x: fmt_br(x),
                    "volume_m3": lambda x: fmt_br(x),
                    "frac_massa_%": lambda x: fmt_br(x),
                    "frac_volume_%": lambda x: fmt_br(x),
                    "vm": lambda x: fmt_br(x),
                    "ts": lambda x: fmt_br(x),
                    "cinza": lambda x: fmt_br(x),
                    "densidade": lambda x: fmt_br(x),
                }
            ),
            use_container_width=True,
        )

        if not df_descartadas.empty:
            st.divider()
            st.subheader("Linhas Descartadas da Receita")
            st.dataframe(df_descartadas, use_container_width=True)

        st.divider()
        st.subheader("Composição, Geometria e Visualização do Caso Selecionado")

        col_tabela, col_grafico = st.columns([1.15, 1.2])

        with col_tabela:
            st.markdown(f"**Caso exibido:** {modo_construtivo}")

            if modo_construtivo == "Pilha A - Estratos por camada":
                df_a = pile_a["df_camadas"].copy()
                st.info(
                    f"**Pilha A**\n\n"
                    f"- Estratificação direta das massas da receita;\n"
                    f"- Altura ocupada: **{fmt_br(pile_a['altura_efetiva_m'])} m**."
                )
                st.dataframe(
                    df_a.style.format(
                        {
                            "ton_receita": lambda x: fmt_br(x),
                            "densidade": lambda x: fmt_br(x),
                            "volume_m3": lambda x: fmt_br(x),
                            "y_base_m": lambda x: fmt_br(x),
                            "y_topo_m": lambda x: fmt_br(x),
                            "espessura_m": lambda x: fmt_br(x),
                            "largura_base_m": lambda x: fmt_br(x),
                            "largura_topo_m": lambda x: fmt_br(x),
                        }
                    ),
                    use_container_width=True,
                )
            else:
                abas = st.tabs(["Lifts", "Sublifts", "Receita por camada"])

                with abas[0]:
                    st.dataframe(
                        pile_b["df_lifts"].style.format(
                            {
                                "y_base_m": lambda x: fmt_br(x),
                                "y_topo_m": lambda x: fmt_br(x),
                                "altura_lift_m": lambda x: fmt_br(x),
                                "largura_base_m": lambda x: fmt_br(x),
                                "largura_topo_m": lambda x: fmt_br(x),
                                "volume_lift_m3": lambda x: fmt_br(x),
                                "massa_lift_t": lambda x: fmt_br(x),
                            }
                        ),
                        use_container_width=True,
                    )

                with abas[1]:
                    st.dataframe(
                        pile_b["df_sublifts"].style.format(
                            {
                                "y_base_m": lambda x: fmt_br(x),
                                "y_topo_m": lambda x: fmt_br(x),
                                "altura_sublift_m": lambda x: fmt_br(x),
                                "largura_base_m": lambda x: fmt_br(x),
                                "largura_topo_m": lambda x: fmt_br(x),
                                "volume_sublift_m3": lambda x: fmt_br(x),
                                "massa_sublift_t": lambda x: fmt_br(x),
                            }
                        ),
                        use_container_width=True,
                    )

                with abas[2]:
                    st.dataframe(
                        pile_b["df_layers"][["camada", "ton_receita", "volume_m3", "frac_massa_%", "frac_volume_%"]].style.format(
                            {
                                "ton_receita": lambda x: fmt_br(x),
                                "volume_m3": lambda x: fmt_br(x),
                                "frac_massa_%": lambda x: fmt_br(x),
                                "frac_volume_%": lambda x: fmt_br(x),
                            }
                        ),
                        use_container_width=True,
                    )

                st.info(
                    f"**Pilha B**\n\n"
                    f"- A receita informada é desdobrada em lifts e sublifts para visualização executiva;\n"
                    f"- Altura máxima de lift: **{fmt_br(altura_lift)} m**;\n"
                    f"- Altura máxima de sublift: **{fmt_br(altura_sublift)} m**;\n"
                    f"- Altura ocupada: **{fmt_br(pile_b['altura_efetiva_m'])} m**."
                )

        with col_grafico:
            if modo_construtivo == "Pilha A - Estratos por camada":
                fig = build_pile_a_figure(pile_a["df_camadas"], larg_base, alt_max, angulo_rep)
            else:
                fig = build_pile_b_figure(pile_b, larg_base, alt_max, angulo_rep)
            st.plotly_chart(fig, use_container_width=True)

        excel_bytes = export_excel(
            df_input=df_norm,
            df_usadas=df_receita,
            df_descartadas=df_descartadas,
            df_total=df_total,
            df_specs=df_specs,
            pile_a=pile_a,
            pile_b=pile_b,
            geometry=geometry,
        )

        st.download_button(
            "Baixar resultado em Excel",
            data=excel_bytes,
            file_name="resultado_media_ponderada_pilha_rom.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    except Exception as e:
        st.error(f"Erro na execução matemática: {e}")
