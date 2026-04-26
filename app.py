import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ==============================================================================
# 1. CONSTANTES Y PARÁMETROS (Tabla X.4 — Capítulo X)
# ==============================================================================

Cpa_BASE      = 4.18      # Capacidad calorífica del agua (J/g ºC)
Tain_BASE     = 20.0      # Temperatura del agua de entrada al calderín (ºC)
Teb_BASE      = 100.0     # Temperatura de ebullición (ºC)
lambda_a_BASE = 2257.0    # Calor específico de vaporización del agua (J/g)

VMV_BASE      = 2000.0    # Volumen de Material Vegetal (cm³) — 1 canasto
C0_BASE       = 0.0063    # Concentración inicial de AE en MV (g/cm³) — valor propuesto
h_BASE        = 10.0      # Semiespesor del lecho (cm)
D_BASE        = 1.0e-2    # Coeficiente de difusión del AE en el sólido (cm²/s) ≈ 1e-6 m²/s
rho_AE_BASE   = 0.84      # Densidad del AE (g/mL) — verificar experimentalmente

PrecioAE_BASE  = 4200.0   # Precio de mercado del AE ($/mL) — Aromáticas Alto Valle, 2026
PrecioEnv_BASE = 630.0    # Precio del envase de 10 mL ($/unidad) — MercadoLibre, 2026
VolumenEnv     = 10.0     # Volumen del envase (mL)
CF_BASE        = 14520.0  # Costos fijos ($/lote): 8 h × $1.815/h — SMVM Res. 9/2025

# Tarifa EDESAL (Tabla X.3 — Capítulo X)
Cf_EDESAL  = 9634.9154    # Cargo fijo ($/mes)
Cr_EDESAL  = 18963.3389   # Cargo uso de red ($/mes)
Cv_BASE    = 117.9787     # Cargo variable ($/kWh)
factor_imp = 1.27383      # Factor impositivo (IVA 21% + Contrib. Municipal 6,383%)

# ==============================================================================
# 2. FUNCIONES DE CÁLCULO (Ecuaciones del Capítulo X)
# ==============================================================================

def calcular_mv(PeR, eta_c, Cpa, Teb, Tain, lambda_a):
    """Caudal de vapor generado mv (g/s) — Ecuación 26."""
    numerador   = (1 - eta_c) * PeR
    denominador = Cpa * (Teb - Tain) + lambda_a
    if denominador == 0:
        return 0.0
    return numerador / denominador

def mAE_acum(tf, F1, F2, n_terms=10):
    """Masa acumulada de AE en el tiempo tf (g) — Ecuación 14."""
    total = 0.0
    for n in range(n_terms):
        term = (2 * n + 1)**2
        if F2 * term == 0:
            continue
        total += (1 - np.exp(-F2 * term * tf)) / (F2 * term)
    return F1 * total

def calcular_xc(mAE_acum_val, mv, tf):
    """Composición del condensado xc (g AE/g condensado) — Ecuación 17."""
    mc = mAE_acum_val + mv * tf
    if mc == 0:
        return 0.0
    return mAE_acum_val / mc

def calcular_rendimiento(mAE_acum_val, VMV, C0):
    """Rendimiento de extracción %Rend — Ecuación 22."""
    MAE0 = VMV * C0
    if MAE0 == 0:
        return 0.0
    return (mAE_acum_val / MAE0) * 100.0

def mAE_acum_vol(mAE_acum_val, rho_AE):
    """Conversión masa AE a volumen (mL) — Ecuación 33."""
    if rho_AE == 0:
        return 0.0
    return mAE_acum_val / rho_AE

def calcular_ingresos(mAE_acum_val, PrecioAE, rho_AE):
    """Ingresos I(tf) ($/lote) — Ecuación 33."""
    return mAE_acum_vol(mAE_acum_val, rho_AE) * PrecioAE

def calcular_N(mAE_acum_val, rho_AE, VolumenEnv):
    """Número de envases N — Ecuación 36."""
    V_AE = mAE_acum_vol(mAE_acum_val, rho_AE)
    return int(np.floor(V_AE / VolumenEnv))

def calcular_Celect(tf, PeR, Cv):
    """Costo de energía eléctrica CEE ($/lote) — Ecuación 35."""
    PeR_kW = PeR / 1000.0
    tf_h   = tf / 3600.0
    return factor_imp * (Cf_EDESAL + Cr_EDESAL + Cv * PeR_kW * tf_h)

def calcular_CO(tf, PeR, Cv, PrecioEnv, rho_AE, mAE_acum_val):
    """Costo de operación CO(tf) ($/lote) — Ecuación 34."""
    Celect = calcular_Celect(tf, PeR, Cv)
    N      = calcular_N(mAE_acum_val, rho_AE, VolumenEnv)
    return Celect + N * PrecioEnv, N, Celect

def calcular_G(I, CO, CF):
    """Ganancia neta G(tf) ($/lote) — Ecuación 32."""
    return I - CO - CF

# ==============================================================================
# 3. CONFIGURACIÓN DE PÁGINA
# ==============================================================================

st.set_page_config(
    layout      = "wide",
    page_title  = "Simulador Destilación AE — Schinus Areira L.",
    page_icon   = "🌿"
)

st.title("🌿 Simulador de Destilación por Arrastre con Vapor")
st.markdown(
    "Simulación del proceso de extracción de Aceite Esencial de *Schinus Areira L.* — "
    "Capítulo X: Modelado y simulación"
)
st.markdown("---")

# ==============================================================================
# 4. SIDEBAR — PARÁMETROS DE ENTRADA
# ==============================================================================

with st.sidebar:
    st.header("⚙️ Parámetros de entrada")
    st.caption("Valores por defecto: Tabla X.4, Capítulo X")

    # --- Operación ---
    st.subheader("1. Condiciones de operación")
    PeR    = st.number_input("Potencia resistencia PeR (W)",
                              value=1500.0, step=100.0, format="%.1f")
    eta_c  = st.slider("Fracción calor perdido ηc",
                        min_value=0.0, max_value=0.5,
                        value=0.1, step=0.01, format="%.2f")
    Tain   = st.number_input("Temp. agua entrada Tain (ºC)",
                              value=Tain_BASE, step=1.0, format="%.1f")
    tf_min = st.slider("Tiempo de operación tf (min)",
                        min_value=10, max_value=300, value=152, step=5)
    tf_s   = tf_min * 60.0

    # --- Material vegetal ---
    st.subheader("2. Material vegetal y difusión")
    VMV    = st.number_input("Volumen MV — VMV (cm³)",
                              value=VMV_BASE, min_value=0.0, max_value=8000.0,
                              step=500.0, format="%.0f",
                              help="Capacidad máxima: 4 canastos × 2000 cm³ = 8000 cm³ (lote completo)")
    C0     = st.number_input("Conc. inicial AE — C₀ (g/cm³)",
                              value=C0_BASE, step=0.001, format="%.4f")
    h      = st.number_input("Semiespesor MV — h (cm)",
                              value=h_BASE, step=0.5, format="%.2f")
    D_input= st.number_input("Coef. difusión D (cm²/s)",
                              value=D_BASE, step=1e-3,
                              format="%.4f",
                              help="Valor base ≈ 1×10⁻² cm²/s (≈ 1×10⁻⁶ m²/s). Ajustar con datos experimentales.")
    D      = D_input
    rho_AE = st.number_input("Densidad AE — ρ (g/mL)",
                              value=rho_AE_BASE, step=0.01, format="%.3f",
                              help="Verificar experimentalmente. Referencia: ~0,84 g/mL")

    # --- Económicos ---
    st.subheader("3. Parámetros económicos")
    PrecioAE  = st.number_input("Precio AE ($/mL)",
                                 value=PrecioAE_BASE, step=100.0, format="%.2f",
                                 help="Aromáticas Alto Valle, 2026")
    PrecioEnv = st.number_input("Costo envase 10 mL ($/unidad)",
                                 value=PrecioEnv_BASE, step=10.0, format="%.2f",
                                 help="MercadoLibre, 2026")
    CF        = st.number_input("Costos fijos CF ($/lote)",
                                 value=CF_BASE, step=500.0, format="%.2f",
                                 help="8 h mano de obra × $1.815/h (SMVM, Res. 9/2025)")
    Cv        = st.number_input("Cargo variable electricidad Cv ($/kWh)",
                                 value=Cv_BASE, step=1.0, format="%.4f",
                                 help="EDESAL Tarifa T1R-3, 2026")

    st.markdown("---")
    st.caption("📌 Cargos fijos EDESAL incluidos en CEE (Ec. 35):\n"
               f"Cf = ${Cf_EDESAL:,.2f}/mes | Cr = ${Cr_EDESAL:,.2f}/mes")

# ==============================================================================
# 5. FUNCIÓN DE FORMATO NUMÉRICO (Argentina: coma decimal, sin separador de miles)
# ==============================================================================

def fmt(val, decimals=0):
    """Formato argentino: sin separador de miles, coma decimal."""
    return f"{val:.{decimals}f}".replace(".", ",")

# ==============================================================================
# 6. CÁLCULOS
# ==============================================================================

try:
    mv = calcular_mv(PeR, eta_c, Cpa_BASE, Teb_BASE, Tain, lambda_a_BASE)
    F1 = 2 * VMV * C0 * D / (h**2)
    F2 = D * np.pi**2 / (4 * h**2)

    if mv <= 0 or F1 <= 0 or F2 <= 0:
        raise ValueError("Parámetros inválidos: verificar PeR, ηc, D, VMV, C₀ o h.")

    tc_s   = 1.0 / F2
    tc_h   = tc_s / 3600.0
    tc_min = tc_s / 60.0

    # Resultados al tiempo tf
    mAE_val      = mAE_acum(tf_s, F1, F2)
    xc_val       = calcular_xc(mAE_val, mv, tf_s)
    rend_val     = calcular_rendimiento(mAE_val, VMV, C0)
    V_AE_val     = mAE_acum_vol(mAE_val, rho_AE)
    I_val        = calcular_ingresos(mAE_val, PrecioAE, rho_AE)
    CO_val, N_val, Celect_val = calcular_CO(tf_s, PeR, Cv, PrecioEnv, rho_AE, mAE_val)
    G_val        = calcular_G(I_val, CO_val, CF)
    MAE0         = VMV * C0
    MAE_rem      = MAE0 - mAE_val

    # Serie temporal para gráfico
    t_arr = np.arange(60, tf_s + 60, 60)
    G_arr, I_arr, CT_arr = [], [], []

    for t_i in t_arr:
        mAE_i        = mAE_acum(t_i, F1, F2)
        I_i          = calcular_ingresos(mAE_i, PrecioAE, rho_AE)
        CO_i, _, _   = calcular_CO(t_i, PeR, Cv, PrecioEnv, rho_AE, mAE_i)
        G_i          = calcular_G(I_i, CO_i, CF)
        G_arr.append(G_i)
        I_arr.append(I_i)
        CT_arr.append(CO_i + CF)

    G_arr  = np.array(G_arr)
    t_min  = t_arr / 60.0

    # Punto de rentabilidad
    t_eq = None
    if G_arr.max() > 0 and G_arr.min() < 0:
        t_eq = float(np.interp(0, G_arr, t_min))

    # Tiempo óptimo (máximo de G)
    idx_opt = int(np.argmax(G_arr))
    t_opt   = t_min[idx_opt]
    G_opt   = G_arr[idx_opt]

    # ==============================================================================
    # 6. RESULTADOS EN PANTALLA
    # ==============================================================================

    # Barra de info superior
    st.info(
        f"⏱ **Tiempo característico:** tc = {fmt(tc_min,1)} min   |   "
        f"💧 **Caudal de vapor:** mv = {fmt(mv*1000,3)} g/s   |   "
        f"⚖️ **Masa inicial AE:** MAE₀ = {fmt(MAE0,2)} g"
    )

    tab1, tab2 = st.tabs(["📊 Resultados", "📈 Gráfico G(t)"])

    # --- TAB 1: RESULTADOS ---
    with tab1:
        st.subheader(f"Resultados a tf = {tf_min} min")

        # Proceso
        st.markdown("##### 🔬 Proceso")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rendimiento %Rend",     f"{fmt(rend_val,2)} %")
        c2.metric("Calidad del producto xc", f"{fmt(xc_val*100,3)} %")
        c3.metric("Masa AE acumulada",     f"{fmt(mAE_val,3)} g")
        c4.metric("Volumen AE producido",  f"{fmt(V_AE_val,3)} mL")

        st.markdown("---")

        # Económico
        st.markdown("##### 💰 Económico")
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Ingresos I(tf)",          f"$ {fmt(I_val,0)}")
        e2.metric("Costo operación CO(tf)",  f"$ {fmt(CO_val,0)}")
        e3.metric("Costos fijos CF",         f"$ {fmt(CF,0)}")
        e4.metric("Ganancia neta G(tf)",     f"$ {fmt(G_val,0)}",
                  delta="positiva ✅" if G_val > 0 else "negativa ❌",
                  delta_color="normal" if G_val > 0 else "inverse")

        st.markdown("---")

        # Adicional
        st.markdown("##### 📦 Detalles")
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Envases completos (10 mL)", f"{N_val} unidades")
        a2.metric("Costo eléctrico CEE",       f"$ {fmt(Celect_val,0)}")
        a3.metric("AE remanente en MV",        f"{fmt(MAE_rem,3)} g")
        a4.metric("Tiempo óptimo t_opt",       f"{fmt(t_opt,0)} min  |  G = $ {fmt(G_opt,0)}")

        st.markdown("---")

        # Mensaje de rentabilidad
        if t_eq:
            st.success(f"✅ El proceso es rentable a partir de **{fmt(t_eq,1)} min**. "
                       f"La ganancia máxima es **$ {fmt(G_opt,0)}** a los **{fmt(t_opt,0)} min**.")
        else:
            st.warning("⚠️ Con los parámetros actuales el proceso no alcanza rentabilidad "
                       "en el tiempo simulado. Aumentá tf o revisá los parámetros económicos.")

    # --- TAB 2: GRÁFICO ---
    with tab2:
        st.subheader("Análisis económico: Ingresos, Costos y Ganancia neta vs. tiempo")

        fig, ax = plt.subplots(figsize=(11, 5))

        ax.plot(t_min, I_arr,  color='steelblue', lw=2,   label='Ingresos $I(t)$')
        ax.plot(t_min, CT_arr, color='firebrick',  lw=2,   label='Costos totales $CO(t) + C_F$')
        ax.plot(t_min, G_arr,  color='seagreen',   lw=2.5, label='Ganancia neta $G(t)$')
        ax.axhline(0, color='black', lw=0.8, ls='-')

        if t_eq:
            ax.axvline(t_eq, color='tomato', lw=1.5, ls='--',
                       label=f'Rentable desde {fmt(t_eq,1)} min')
            ax.plot(t_eq, 0, 'o', color='tomato', ms=7)

        ax.axvline(t_opt, color='seagreen', lw=1.5, ls=':',
                   label=f'Tiempo óptimo {fmt(t_opt,0)} min (G = ${fmt(G_opt,0)})')
        ax.plot(t_opt, G_opt, 's', color='seagreen', ms=8)

        ax.set_xlabel('Tiempo de operación (min)', fontsize=11)
        ax.set_ylabel('$ (pesos / lote)',           fontsize=11)
        ax.set_title('Ganancia neta G(t) en función del tiempo de operación', fontsize=12)
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_xlim([0, t_min[-1]])

        st.pyplot(fig, use_container_width=True)

        if t_eq:
            st.success(f"✅ Rentable desde **{fmt(t_eq,1)} min** | "
                       f"Tiempo óptimo: **{fmt(t_opt,0)} min** | "
                       f"Ganancia máxima: **$ {fmt(G_opt,0)} /lote**")

except ValueError as ve:
    st.error(f"❌ Error en el cálculo: {ve}")
except Exception as e:
    st.error("❌ Error inesperado. Por favor revisá los parámetros de entrada.")
    st.exception(e)
