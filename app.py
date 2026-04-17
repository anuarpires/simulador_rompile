
import math
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import linprog


# ============================================================
# 1) CONFIGURAÇÃO GERAL DA APLICAÇÃO
# ============================================================
# Ajusta o layout para tela larga e define o título principal.
st.set_page_config(page_title="Otimizador de Pilha ROM - Copelmi", layout="wide")
st.title("Otimizador Avançado de Blending - Pilha ROM")

# Texto introdutório do app.
st.markdown(
    """
    Este aplicativo otimiza o **blend global** por programação linear e, em seguida,
    representa a pilha de duas formas distintas:

    - **Pilha A - Estratos por camada**: primeira camada esteirada como base e as demais camadas empilhadas por cima;
    - **Pilha B - Lift/Sublifting**: mesma receita global repetida em cada lift e em cada sublift.

    A composição química média global do blend permanece a mesma; o que muda entre os casos
    é a **distribuição espacial da pilha**, a **tabela executiva** e a **visualização gráfica**.
    """
)

# Ordem visual das camadas, do fundo para o topo da pilha.
ORDEM_CONSTRUCAO = {"CI": 1, "CS": 2, "S2": 3, "S3": 4, "S4": 5, "S5": 6, "S6": 7}

# Paleta fixa para cada camada.
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
def zscore(x: np.ndarray) -> np.ndarray:
    """
    Padroniza um vetor para evitar que grandezas com escalas diferentes
    dominem a função objetivo.
    """
    s = x.std()
    return np.zeros_like(x) if s == 0 else (x - x.mean()) / s


def build_linear_problem(df: pd.DataFrame, specs: dict, target_mass: float, volume_max_m3: float | None):
    """
    Monta o problema de otimização linear do blend.

    Variáveis de decisão:
        x_i = toneladas escolhidas de cada camada i

    Restrições:
        - soma das toneladas = massa-alvo
        - VM mínimo
        - TS máximo
        - cinza/CBS máximo
        - volume máximo do pátio

    Função objetivo:
        - favorecer VM maior
        - penalizar TS e cinza maiores
    """
    n = len(df)

    vm = df["vm"].to_numpy(dtype=float)
    ts = df["ts"].to_numpy(dtype=float)
    cinza = df["cinza"].to_numpy(dtype=float)
    rho = df["densidade"].to_numpy(dtype=float)

    # Cada camada só pode variar entre zero e o inventário disponível.
    bounds = [(0.0, float(row["ton_report"])) for _, row in df.iterrows()]

    A_ub, b_ub = [], []
    A_eq, b_eq = [], []

    # Restrição de massa total.
    A_eq.append(np.ones(n))
    b_eq.append(float(target_mass))

    # Restrição de TS máximo.
    if specs.get("ts_max") is not None:
        A_ub.append(ts - float(specs["ts_max"]))
        b_ub.append(0.0)

    # Restrição de cinza/CBS máximo.
    if specs.get("cinza_max") is not None:
        A_ub.append(cinza - float(specs["cinza_max"]))
        b_ub.append(0.0)

    # Restrição de VM mínimo.
    if specs.get("vm_min") is not None:
        A_ub.append(float(specs["vm_min"]) - vm)
        b_ub.append(0.0)

    # Restrição de volume máximo do pátio.
    if volume_max_m3 is not None:
        A_ub.append(1.0 / rho)
        b_ub.append(float(volume_max_m3))

    # Objetivo multicritério simplificado por escore padronizado.
    c = (-1.0 * zscore(vm) + 1.0 * zscore(ts) + 1.0 * zscore(cinza))
    return c, A_ub, b_ub, A_eq, b_eq, bounds


def calculate_quality_metrics(df_res: pd.DataFrame) -> dict:
    """
    Calcula as propriedades médias ponderadas e o volume total do blend final.
    """
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
    """
    Monta a composição total enriquecida com volume e frações de massa/volume.
    """
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
    """
    Retorna a largura da seção na cota y para um invólucro trapezoidal
    com talude definido por ângulo de repouso.
    """
    tanv = math.tan(math.radians(angulo))
    if tanv <= 0:
        return larg_base
    return max(0.0, larg_base - 2.0 * (y / tanv))


def cross_section_area_up_to(h: float, larg_base: float, angulo: float) -> float:
    """
    Área acumulada da seção transversal desde a base até a altura h.
    """
    h = max(0.0, h)
    largura_topo = width_at_height(h, larg_base, angulo)
    return h * (larg_base + largura_topo) / 2.0


def longitudinal_trapezoid_volume(comp: float, larg_base: float, alt_max: float, angulo: float) -> float:
    """
    Volume máximo do invólucro longitudinal trapezoidal.
    """
    return comp * cross_section_area_up_to(alt_max, larg_base, angulo)


def volume_between_heights(y0: float, y1: float, comp: float, larg_base: float, angulo: float) -> float:
    """
    Volume contido entre duas cotas da pilha.
    """
    y0 = max(0.0, y0)
    y1 = max(y0, y1)
    return comp * (cross_section_area_up_to(y1, larg_base, angulo) - cross_section_area_up_to(y0, larg_base, angulo))


def solve_height_for_volume(volume_target: float, comp: float, larg_base: float, alt_max: float, angulo: float) -> float:
    """
    Resolve a altura total ocupada no invólucro para um volume conhecido.
    """
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
    y_base: float,
    target_volume: float,
    comp: float,
    larg_base: float,
    alt_max: float,
    angulo: float,
) -> float:
    """
    Dado um volume a ser alocado acima de y_base, encontra a cota superior y_topo
    de forma que o volume entre y_base e y_topo seja igual ao alvo.
    """
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
    df_res: pd.DataFrame,
    comp_base: float,
    larg_base: float,
    alt_max: float,
    angulo_rep: float,
) -> dict:
    """
    Constrói a Pilha A.

    Lógica:
        1. Ordena as camadas do fundo para o topo;
        2. Calcula o volume de cada camada;
        3. Esteira a primeira camada como base;
        4. Empilha as demais por cima, camada pura sobre camada pura;
        5. Cada camada ocupa um intervalo vertical próprio dentro do invólucro trapezoidal.
    """
    df_total = df_res.copy()
    df_total["volume_m3"] = df_total["ton_calculada"] / df_total["densidade"]
    df_total["ordem_plot"] = df_total["camada"].map(ORDEM_CONSTRUCAO).fillna(99)
    df_total = df_total.sort_values("ordem_plot").reset_index(drop=True)

    rows = []
    y_atual = 0.0

    for _, row in df_total.iterrows():
        vol = float(row["volume_m3"])
        y_topo = solve_upper_height_for_segment_volume(
            y_base=y_atual,
            target_volume=vol,
            comp=comp_base,
            larg_base=larg_base,
            alt_max=alt_max,
            angulo=angulo_rep,
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
# 5) PILHA B - LIFT / SUBLIFTING
# ============================================================
def prepare_pile_b_lifts(
    df_res: pd.DataFrame,
    comp_base: float,
    larg_base: float,
    alt_max: float,
    angulo_rep: float,
    altura_lift: float,
    altura_sublift: float,
) -> dict:
    """
    Constrói a Pilha B.

    Lógica:
        1. Calcula a altura total ocupada no invólucro;
        2. Divide a pilha em lifts;
        3. Divide cada lift em sublifts;
        4. Repete a mesma receita mássica em cada lift e em cada sublift;
        5. A massa de cada fatia é proporcional ao volume geométrico da fatia.
    """
    metrics = calculate_quality_metrics(df_res)
    massa_total = metrics["massa_final"]
    volume_total = metrics["volume_total"]

    df_layers = enrich_total_composition(df_res, metrics).copy()
    df_layers["mass_frac"] = df_layers["ton_calculada"] / massa_total
    df_layers["vol_frac"] = df_layers["volume_m3"] / volume_total

    altura_efetiva = solve_height_for_volume(volume_total, comp_base, larg_base, alt_max, angulo_rep)

    # -------------------------
    # Tabela de lifts
    # -------------------------
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

        # Reparte a receita total dentro de cada lift.
        for _, layer in df_layers.iterrows():
            row[f"{layer['camada']}_t"] = massa_lift * float(layer["mass_frac"])

        lift_rows.append(row)
        y0 = y1
        lift_id += 1

    df_lifts = pd.DataFrame(lift_rows)

    # -------------------------
    # Tabela de sublifts
    # -------------------------
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

            # Reparte a mesma receita total em cada sublift.
            for _, layer in df_layers.iterrows():
                row[f"{layer['camada']}_t"] = massa_sub * float(layer["mass_frac"])

            sublift_rows.append(row)

            # Guarda a fatia para desenho posterior.
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
        "draw_slices": draw_slices,
        "altura_efetiva_m": altura_efetiva,
    }


# ============================================================
# 6) FUNÇÕES DE VISUALIZAÇÃO GRÁFICA
# ============================================================
def build_pile_a_figure(df_camadas: pd.DataFrame, larg_base: float, alt_max: float, angulo_rep: float) -> go.Figure:
    """
    Desenha a Pilha A como estratos puros por camada.
    """
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
        xaxis_title="Largura da pilha (m)",
        yaxis_title="Altura (m)",
        xaxis=dict(range=[-5, larg_base + 5], tick0=0, dtick=10),
        yaxis=dict(range=[0, max(alt_max + 0.5, float(df_camadas['y_topo_m'].max()) + 0.5 if not df_camadas.empty else alt_max + 0.5)]),
        template="plotly_white",
        legend_title="Camadas",
        margin=dict(l=20, r=20, t=60, b=20),
        hovermode="closest",
    )
    return fig


def build_pile_b_figure(model: dict, larg_base: float, alt_max: float, angulo_rep: float) -> go.Figure:
    """
    Desenha a Pilha B como bandas repetidas em cada sublift.
    """
    df_layers = model["df_layers"]
    df_sublifts = model["df_sublifts"]
    draw_slices = model["draw_slices"]
    altura_efetiva = model["altura_efetiva_m"]

    fig = go.Figure()
    centro_x = larg_base / 2.0

    # Utiliza fração volumétrica para dividir a altura de cada sublift entre as camadas.
    vol_fracs = df_layers[["camada", "vol_frac"]].copy()

    for s in draw_slices:
        y_cursor = float(s["y0"])
        slice_height = float(s["altura"])

        # Desenha as bandas de receita repetida dentro da sublift.
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

            mask = df_sublifts["lift"].eq(s["lift"]) & df_sublifts["sublift"].eq(s["sublift"])
            massa_layer = None
            if f"{camada}_t" in df_sublifts.columns and mask.any():
                massa_layer = float(df_sublifts.loc[mask, f"{camada}_t"].iloc[0])

            fig.add_trace(
                go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    fill="toself",
                    mode="lines",
                    line=dict(color="white", width=0.8),
                    fillcolor=CORES_CAMADAS.get(camada, "#777777"),
                    name=camada,
                    showlegend=False,
                    text=(
                        f"Lift {s['lift']} | Sublift {s['sublift']}"
                        f"<br>Camada: {camada}"
                        f"<br>Faixa vertical: {band_h:.3f} m"
                        + (f"<br>Massa da camada nesta sublift: {massa_layer:,.0f} t" if massa_layer is not None else "")
                    ),
                    hoverinfo="text",
                )
            )
            y_cursor = y_top

        # Desenha uma linha horizontal pontilhada na transição de cada sublift.
        fig.add_hline(
            y=float(s["y1"]),
            line_dash="dot",
            line_color="rgba(30,30,30,0.35)",
            annotation_text=f"L{s['lift']}-S{s['sublift']}",
            annotation_position="left",
        )

    # Adiciona a legenda manual das camadas.
    for camada in df_layers["camada"].tolist():
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
        title="Seção Transversal - Pilha B | Lift / Sublifting",
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


# ============================================================
# 7) FUNÇÕES DE EXPORTAÇÃO EXCEL
# ============================================================
def build_excel_bytes_safe(sheets: dict) -> tuple[bytes | None, str | None]:
    """
    Gera um arquivo Excel tentando engines opcionais.
    Se nenhuma estiver disponível, retorna None para não quebrar a interface.
    """
    engines = ["xlsxwriter", "openpyxl"]
    last_error = None

    for engine in engines:
        try:
            xlsx_buffer = BytesIO()

            # Abre o writer do Excel com a engine disponível.
            with pd.ExcelWriter(xlsx_buffer, engine=engine) as writer:
                for sheet_name, df_sheet in sheets.items():
                    safe_name = sheet_name[:31]
                    df_sheet.to_excel(writer, index=False, sheet_name=safe_name)

                    # Ajustes cosméticos para facilitar a leitura do arquivo exportado.
                    if engine == "xlsxwriter":
                        ws = writer.sheets[safe_name]
                        ws.freeze_panes(1, 0)
                        for idx, col in enumerate(df_sheet.columns):
                            max_len = max([len(str(col))] + [len(str(v)) if v is not None else 0 for v in df_sheet[col].tolist()])
                            ws.set_column(idx, idx, min(max(max_len + 2, 12), 28))
                    elif engine == "openpyxl":
                        ws = writer.book[safe_name]
                        ws.freeze_panes = "A2"
                        for col_cells in ws.columns:
                            max_len = max(len(str(cell.value)) if cell.value is not None else 0 for cell in col_cells)
                            ws.column_dimensions[col_cells[0].column_letter].width = min(max(max_len + 2, 12), 28)

            return xlsx_buffer.getvalue(), engine

        except ModuleNotFoundError as e:
            last_error = str(e)
        except ImportError as e:
            last_error = str(e)
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"

    return None, last_error


def build_export_workbook(
    *,
    modo_selecionado: str,
    params: dict,
    metrics: dict,
    df_total: pd.DataFrame,
    pile_a: dict,
    pile_b: dict,
) -> dict:
    """
    Monta todas as abas do Excel contendo as informações das duas pilhas.
    """
    df_resumo = pd.DataFrame(
        [
            {"item": "Modo exibido na tela", "valor": modo_selecionado},
            {"item": "Massa alvo (t)", "valor": params["alvo_massa"]},
            {"item": "Massa final (t)", "valor": metrics["massa_final"]},
            {"item": "VM final (%)", "valor": metrics["vm_final"]},
            {"item": "TS final (%)", "valor": metrics["ts_final"]},
            {"item": "Cinza/CBS final", "valor": metrics["cinza_final"]},
            {"item": "Volume usado (m3)", "valor": metrics["volume_total"]},
            {"item": "Densidade aparente equivalente (t/m3)", "valor": metrics["rho_aparente"]},
            {"item": "Comprimento base (m)", "valor": params["comp_base"]},
            {"item": "Largura base (m)", "valor": params["larg_base"]},
            {"item": "Altura máxima do pátio (m)", "valor": params["alt_max"]},
            {"item": "Ângulo de repouso (graus)", "valor": params["angulo_rep"]},
            {"item": "Volume máximo do pátio (m3)", "valor": params["vol_max"]},
            {"item": "VM mínimo especificado (%)", "valor": params["vm_min"]},
            {"item": "TS máximo especificado (%)", "valor": params["ts_max"]},
            {"item": "Cinza/CBS máximo especificado", "valor": params["cinza_max"]},
            {"item": "Altura ocupada Pilha A (m)", "valor": pile_a["altura_efetiva_m"]},
            {"item": "Altura ocupada Pilha B (m)", "valor": pile_b["altura_efetiva_m"]},
            {"item": "Altura de lift (m)", "valor": params["altura_lift"]},
            {"item": "Altura de sublift (m)", "valor": params["altura_sublift"]},
        ]
    )

    # Seleciona as colunas principais da composição total.
    df_total_export = df_total[["camada", "ton_calculada", "volume_m3", "frac_massa_%", "frac_volume_%"]].copy()

    # Seleciona as colunas da Pilha A.
    df_a = pile_a["df_camadas"].copy()

    # Seleciona as colunas principais dos lifts e sublifts da Pilha B.
    df_b_lifts = pile_b["df_lifts"].copy()
    df_b_sublifts = pile_b["df_sublifts"].copy()

    sheets = {
        "resumo_geral": df_resumo,
        "composicao_total": df_total_export,
        "pilha_A_estratos": df_a,
        "pilha_B_lifts": df_b_lifts,
        "pilha_B_sublifts": df_b_sublifts,
    }

    xlsx_bytes, xlsx_status = build_excel_bytes_safe(sheets)
    return {
        "xlsx_bytes": xlsx_bytes,
        "xlsx_status": xlsx_status,
        "sheets": sheets,
    }


def render_excel_download(export_bundle: dict):
    """
    Exibe o botão final de download do relatório Excel.
    """
    st.divider()
    st.subheader("Output em Excel")

    if export_bundle.get("xlsx_bytes") is not None:
        st.download_button(
            label="Baixar Excel com informações completas da Pilha A e da Pilha B",
            data=export_bundle["xlsx_bytes"],
            file_name="pilhas_rom_casos_A_e_B.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
        st.caption(f"Engine de exportação utilizada: {export_bundle.get('xlsx_status')}")
    else:
        st.warning("A exportação Excel não está disponível neste ambiente.")
        if export_bundle.get("xlsx_status"):
            st.caption(f"Detalhe técnico: {export_bundle.get('xlsx_status')}")


# ============================================================
# 8) SIDEBAR - PARÂMETROS DE ENTRADA
# ============================================================
# Parâmetros de massa alvo.
st.sidebar.header("Parâmetros da Pilha")
alvo_massa = st.sidebar.number_input("Massa Alvo (t)", value=50000.0, step=1000.0)

# Parâmetros geométricos do pátio.
st.sidebar.header("Geometria do Pátio")
comp_base = st.sidebar.number_input("Comprimento Base (m)", value=120.0)
larg_base = st.sidebar.number_input("Largura Base (m)", value=70.0)
alt_max = st.sidebar.number_input("Altura Máxima (m)", value=5.0)
angulo_rep = st.sidebar.number_input("Ângulo de Repouso (graus)", value=37.0)

# Seleção de qual pilha será exibida em tela.
st.sidebar.header("Caso para Exibição")
modo_construtivo = st.sidebar.radio(
    "Escolha a pilha a exibir",
    options=["Pilha A - Estratos por camada", "Pilha B - Lift/Sublifting"],
)

# Parâmetros específicos da Pilha B.
st.sidebar.header("Parâmetros da Pilha B")
altura_lift = st.sidebar.number_input("Altura do lift (m)", min_value=0.1, value=1.0, step=0.1)
altura_sublift = st.sidebar.number_input("Altura do sublift (m)", min_value=0.1, value=0.5, step=0.1)

# Garante que o sublift nunca seja maior que o lift.
if altura_sublift > altura_lift:
    st.sidebar.warning("A altura do sublift foi limitada à altura do lift.")
    altura_sublift = altura_lift

# Restrições de qualidade da usina.
st.sidebar.header("Restrições da Usina")
vm_min = st.sidebar.number_input("VM Mínimo (%)", value=19.30)
ts_max = st.sidebar.number_input("TS Máximo (%)", value=2.20)
cinza_max = st.sidebar.number_input("Cinza/CBS Máximo", value=57.17)


# ============================================================
# 9) TABELA DE DADOS EDITÁVEL
# ============================================================
st.subheader("Inventário de Frentes de Lavra (Editável)")

# Dados padrão do inventário.
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

# Permite que o usuário edite os dados diretamente na página.
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
            # Calcula o volume máximo do pátio a partir do invólucro trapezoidal.
            vol_max = longitudinal_trapezoid_volume(comp_base, larg_base, alt_max, angulo_rep)

            # Monta e resolve o problema de otimização do blend.
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

            # Caso o solver não encontre solução viável, avisa o usuário.
            if not res.success:
                st.error(f"O solver não encontrou uma solução viável: {res.message}")
            else:
                st.success("Solução otimizada encontrada.")

                # Guarda as toneladas calculadas pelo solver.
                df_valido["ton_calculada"] = res.x
                df_res = df_valido[df_valido["ton_calculada"] > 1e-6].copy()

                # Calcula as métricas globais do blend.
                metrics = calculate_quality_metrics(df_res)
                df_total = enrich_total_composition(df_res, metrics)

                # Constrói as duas pilhas a partir do mesmo blend global.
                pile_a = prepare_pile_a_strata(
                    df_res=df_res,
                    comp_base=comp_base,
                    larg_base=larg_base,
                    alt_max=alt_max,
                    angulo_rep=angulo_rep,
                )

                pile_b = prepare_pile_b_lifts(
                    df_res=df_res,
                    comp_base=comp_base,
                    larg_base=larg_base,
                    alt_max=alt_max,
                    angulo_rep=angulo_rep,
                    altura_lift=altura_lift,
                    altura_sublift=altura_sublift,
                )

                # -----------------------------
                # BLOCO DE QUALIDADE GLOBAL
                # -----------------------------
                st.divider()
                st.subheader("Resultados de Qualidade Global do Blend")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric(
                    "Massa Total",
                    f"{metrics['massa_final']:,.0f} t",
                    delta=f"{metrics['massa_final'] - alvo_massa:,.0f} vs alvo",
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

                # -----------------------------
                # BLOCO DE COMPOSIÇÃO TOTAL
                # -----------------------------
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

                # -----------------------------
                # BLOCO DO CASO SELECIONADO
                # -----------------------------
                st.divider()
                st.subheader("Composição, Geometria e Visualização do Caso Selecionado")

                col_tabela, col_grafico = st.columns([1.15, 1.2])

                with col_tabela:
                    st.markdown(f"**Caso exibido:** {modo_construtivo}")

                    # -------- PILHA A --------
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

                    # -------- PILHA B --------
                    else:
                        abas = st.tabs(["Lifts", "Sublifts", "Receita por camada"])

                        with abas[0]:
                            st.dataframe(
                                pile_b["df_lifts"].style.format(
                                    {
                                        "y_base_m": "{:.2f}",
                                        "y_topo_m": "{:.2f}",
                                        "altura_lift_m": "{:.2f}",
                                        "largura_base_m": "{:.2f}",
                                        "largura_topo_m": "{:.2f}",
                                        "volume_lift_m3": "{:,.0f}",
                                        "massa_lift_t": "{:,.0f}",
                                        **{c: "{:,.0f}" for c in pile_b["df_lifts"].columns if c.endswith("_t")},
                                    }
                                ),
                                use_container_width=True,
                            )

                        with abas[1]:
                            st.dataframe(
                                pile_b["df_sublifts"].style.format(
                                    {
                                        "y_base_m": "{:.2f}",
                                        "y_topo_m": "{:.2f}",
                                        "altura_sublift_m": "{:.2f}",
                                        "largura_base_m": "{:.2f}",
                                        "largura_topo_m": "{:.2f}",
                                        "volume_sublift_m3": "{:,.0f}",
                                        "massa_sublift_t": "{:,.0f}",
                                        **{c: "{:,.0f}" for c in pile_b["df_sublifts"].columns if c.endswith("_t")},
                                    }
                                ),
                                use_container_width=True,
                            )

                        with abas[2]:
                            st.dataframe(
                                pile_b["df_layers"][["camada", "ton_calculada", "volume_m3", "mass_frac", "vol_frac"]].style.format(
                                    {
                                        "ton_calculada": "{:,.0f}",
                                        "volume_m3": "{:,.0f}",
                                        "mass_frac": "{:.4f}",
                                        "vol_frac": "{:.4f}",
                                    }
                                ),
                                use_container_width=True,
                            )

                        st.info(
                            f"**Pilha B**\n\n"
                            f"- Receita global repetida em cada lift e em cada sublift;\n"
                            f"- Altura de lift: **{altura_lift:.2f} m**;\n"
                            f"- Altura de sublift: **{altura_sublift:.2f} m**;\n"
                            f"- Altura ocupada: **{pile_b['altura_efetiva_m']:.2f} m**."
                        )

                # Escolhe o gráfico correto conforme o caso exibido.
                with col_grafico:
                    if modo_construtivo == "Pilha A - Estratos por camada":
                        fig = build_pile_a_figure(pile_a["df_camadas"], larg_base, alt_max, angulo_rep)
                    else:
                        fig = build_pile_b_figure(pile_b, larg_base, alt_max, angulo_rep)

                    st.plotly_chart(fig, use_container_width=True)

                # -----------------------------
                # BLOCO DE EXPORTAÇÃO EXCEL
                # -----------------------------
                export_bundle = build_export_workbook(
                    modo_selecionado=modo_construtivo,
                    params={
                        "alvo_massa": alvo_massa,
                        "comp_base": comp_base,
                        "larg_base": larg_base,
                        "alt_max": alt_max,
                        "angulo_rep": angulo_rep,
                        "vm_min": vm_min,
                        "ts_max": ts_max,
                        "cinza_max": cinza_max,
                        "vol_max": vol_max,
                        "altura_lift": altura_lift,
                        "altura_sublift": altura_sublift,
                    },
                    metrics=metrics,
                    df_total=df_total,
                    pile_a=pile_a,
                    pile_b=pile_b,
                )

                render_excel_download(export_bundle)

        except Exception as e:
            st.error(f"Erro na execução matemática: {e}")
