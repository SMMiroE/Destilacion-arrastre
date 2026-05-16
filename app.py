import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ==============================================================================
# 1. CONSTANTES Y PARÁMETROS
# ==============================================================================
Cpa_BASE      = 4.18
Tain_BASE     = 20.0
Teb_BASE      = 90.0
lambda_a_BASE = 2283.0

VMV_BASE      = 1500.0
C0_BASE       = 0.01103
Deff_BASE     = 4.8336e-4
rho_AE_BASE   = 0.88

PrecioAE_BASE = 4000.0
CF_BASE       = 14520.0
Cv_BASE       = 117.9787
factor_imp    = 1.27383

# ==============================================================================
# 2. FUNCIONES DE CÁLCULO
# ==============================================================================
def calcular_mv(PeR, eta_c, Cpa, Teb, Tain, lambda_a):
    """Caudal de vapor mv (g/s) — Ec. X.28"""
    den = Cpa*(Teb-Tain) + lambda_a
    return (1-eta_c)*PeR/den if den > 0 else 0.0

def mAE_acum(tf, F1, F2, n_terms=10):
    """Masa acumulada AE (g) — Ec. X.16"""
    total = 0.0
    for n in range(n_terms):
        term = (2*n+1)**2
        if F2*term > 0:
            total += (1-np.exp(-F2*term*tf))/(F2*term)
    return F1*total

def calcular_xc(mAE_val, mv, tf):
    """Calidad del condensado xc — Ec. X.19"""
    mc = mAE_val + mv*tf
    return mAE_val/mc if mc > 0 else 0.0

def calcular_rendimiento(mAE_val, VMV, C0):
    """Rendimiento %Rend — Ec. X.24"""
    MAE0 = VMV*C0
    return (mAE_val/MAE0)*100 if MAE0 > 0 else 0.0

def calcular_ingresos(mAE_val, PrecioAE, rho_AE):
    """Ingresos I(tf) — Ec. X.37"""
    return (mAE_val/rho_AE)*PrecioAE

def calcular_Celect(t_dest_s, t_cal_s, PeR, Cv):
    """Costo energía eléctrica — Ec. X.39 (tiempo total = dest + calentamiento)"""
    tf_total_h = (t_dest_s + t_cal_s)/3600.0
    return factor_imp*Cv*(PeR/1000)*tf_total_h

def calcular_G(I, Celect, CF):
    """Ganancia neta G — Ec. X.36"""
    return I - Celect - CF

def fmt(val, decimals=0):
    return f"{val:.{decimals}f}".replace(".", ",")

# ==============================================================================
# 3. CONFIGURACIÓN DE PÁGINA
# ==============================================================================
st.set_page_config(layout="wide",
                   page_title="Simulador Destilación AE — Schinus Areira L.",
                   page_icon="🌿")

st.title("🌿 Simulador de Destilación por Arrastre con Vapor")
st.markdown("Simulación del proceso de extracción de Aceite Esencial de *Schinus Areira L.* — "
            "Capítulo X: Modelado y simulación")
st.markdown("---")

# ==============================================================================
# 4. SIDEBAR
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Parámetros de entrada")

    st.subheader("1. Condiciones de operación")
    PeR    = st.number_input("Potencia resistencia PeR (W)", value=1000.0, step=100.0, format="%.1f")
    eta_c  = st.slider("Fracción calor perdido ηc", 0.0, 0.5, 0.0, 0.01, format="%.2f")
    Tain   = st.number_input("Temp. agua entrada Tain (°C)", value=Tain_BASE, step=1.0, format="%.1f")
    t_cal  = st.number_input("Tiempo de puesta en marcha (min)", value=33.0, step=1.0, format="%.0f",
                              help="Tiempo desde el encendido hasta la primera gota de condensado")
    tf_min = st.slider("Tiempo de destilación tf (min)", 10, 300, 88, 5)
    tf_s   = tf_min*60.0
    t_cal_s= t_cal*60.0
    t_total_min = tf_min + t_cal

    st.subheader("2. Material vegetal y difusión")
    VMV    = st.number_input("Volumen MV — VMV (cm³)", value=VMV_BASE, step=100.0, format="%.0f")
    C0     = st.number_input("Conc. inicial AE — C₀ (g/cm³)", value=C0_BASE, step=0.001, format="%.5f")
    Deff   = st.number_input("Difusividad efectiva Deff (1/s)", value=Deff_BASE,
                              step=1e-5, format="%.5f",
                              help="Obtenida por ajuste de mínimos cuadrados con datos experimentales")
    rho_AE = st.number_input("Densidad AE — ρ (g/mL)", value=rho_AE_BASE, step=0.01, format="%.3f")

    st.subheader("3. Parámetros económicos")
    PrecioAE = st.number_input("Precio AE ($/mL)", value=PrecioAE_BASE, step=100.0, format="%.0f",
                                help="Precio de venta a granel")
    CF       = st.number_input("Costos fijos CF ($/lote)", value=CF_BASE, step=500.0, format="%.0f",
                                help="8 h mano de obra × $1.815/h (SMVM, Res. 9/2025)")
    Cv       = st.number_input("Cargo variable electricidad Cv ($/kWh)", value=Cv_BASE,
                                step=1.0, format="%.4f", help="EDESAL Tarifa T1R-3, 2026")

    st.markdown("---")
    st.caption(f"⏱ Tiempo total de operación: {t_total_min:.0f} min")
    st.caption("📌 Factor impositivo 1,27383 incluye IVA y Contrib. Municipal.")

# ==============================================================================
# 5. CÁLCULOS
# ==============================================================================
try:
    mv  = calcular_mv(PeR, eta_c, Cpa_BASE, Teb_BASE, Tain, lambda_a_BASE)
    F1  = 2*VMV*C0*Deff
    F2  = (np.pi**2/4)*Deff
    tc_min = 1/F2/60 if F2 > 0 else 0

    if mv <= 0 or F1 <= 0 or F2 <= 0:
        raise ValueError("Parámetros inválidos.")

    MAE0       = VMV*C0
    mAE_val    = mAE_acum(tf_s, F1, F2)
    xc_val     = calcular_xc(mAE_val, mv, tf_s)
    rend_val   = calcular_rendimiento(mAE_val, VMV, C0)
    V_AE_val   = mAE_val/rho_AE
    I_val      = calcular_ingresos(mAE_val, PrecioAE, rho_AE)
    Celect_val = calcular_Celect(tf_s, t_cal_s, PeR, Cv)
    G_val      = calcular_G(I_val, Celect_val, CF)
    MAE_rem    = MAE0 - mAE_val
    PrecioAE_min = (Celect_val+CF)*rho_AE/mAE_val if mAE_val > 0 else 0

    # Serie temporal — eje X en tiempo TOTAL desde encendido
    # Durante calentamiento: I=0, G negativa por CF
    t_cal_arr  = np.arange(60, t_cal_s+60, 60)  # puntos durante calentamiento
    t_dest_arr = np.arange(60, tf_s+60, 60)      # puntos durante destilación

    # Tiempo total en minutos para el eje X
    t_total_arr = np.concatenate([
        t_cal_arr/60,                          # calentamiento
        t_dest_arr/60 + t_cal               # destilación (desplazado)
    ])

    G_arr, I_arr, CT_arr = [], [], []

    # Durante calentamiento: I=0, CO crece, G negativa
    for t_i in t_cal_arr:
        Ce_i = calcular_Celect(0, t_i, PeR, Cv)  # solo calentamiento
        G_arr.append(calcular_G(0, Ce_i, CF))
        I_arr.append(0.0)
        CT_arr.append(Ce_i + CF)

    # Durante destilación
    for t_i in t_dest_arr:
        mAE_i  = mAE_acum(t_i, F1, F2)
        I_i    = calcular_ingresos(mAE_i, PrecioAE, rho_AE)
        Ce_i   = calcular_Celect(t_i, t_cal_s, PeR, Cv)
        G_i    = calcular_G(I_i, Ce_i, CF)
        G_arr.append(G_i); I_arr.append(I_i); CT_arr.append(Ce_i+CF)

    G_arr = np.array(G_arr)

    # Tiempo de rentabilidad
    t_eq = None
    for i in range(len(G_arr)-1):
        if G_arr[i] < 0 and G_arr[i+1] >= 0:
            t_eq = float(np.interp(0, [G_arr[i], G_arr[i+1]],
                                      [t_total_arr[i], t_total_arr[i+1]]))
            break

    # ==============================================================================
    # 6. RESULTADOS
    # ==============================================================================
    st.info(f"⏱ **Tiempo característico:** tc = {fmt(tc_min,1)} min   |   "
            f"💧 **Caudal de vapor:** mv = {fmt(mv*60,3)} g/min   |   "
            f"⚖️ **Masa inicial AE:** MAE₀ = {fmt(MAE0,2)} g   |   "
            f"🕐 **Tiempo total:** {fmt(t_total_min,0)} min")

    tab1, tab2 = st.tabs(["📊 Resultados", "📈 Gráfico G(t)"])

    with tab1:
        st.subheader(f"Resultados a tf = {tf_min} min de destilación (total {t_total_min:.0f} min)")

        st.markdown("##### 🔬 Proceso")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rendimiento %Rend",    f"{fmt(rend_val,2)} %")
        c2.metric("Calidad xc",           f"{fmt(xc_val*100,3)} %")
        c3.metric("Masa AE acumulada",    f"{fmt(mAE_val,3)} g")
        c4.metric("Volumen AE producido", f"{fmt(V_AE_val,3)} mL")

        st.markdown("---")
        st.markdown("##### 💰 Económico")
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Ingresos I(tf)",       f"$ {fmt(I_val,0)}")
        e2.metric("Costo eléctrico",      f"$ {fmt(Celect_val,0)}")
        e3.metric("Costos fijos CF",      f"$ {fmt(CF,0)}")
        e4.metric("Ganancia neta G(tf)",  f"$ {fmt(G_val,0)}",
                  delta="positiva ✅" if G_val > 0 else "negativa ❌",
                  delta_color="normal" if G_val > 0 else "inverse")

        st.markdown("---")
        st.markdown("##### 📦 Detalles")
        a1, a2, a3 = st.columns(3)
        a1.metric("AE remanente en MV",     f"{fmt(MAE_rem,3)} g")
        a2.metric("Precio mínimo AE (G=0)", f"$ {fmt(PrecioAE_min,0)} /mL")
        a3.metric("Tiempo total operación", f"{fmt(t_total_min,0)} min")

        st.markdown("---")
        if t_eq:
            st.success(f"✅ El proceso es rentable a partir de **{fmt(t_eq,1)} min** "
                       f"desde el encendido del equipo.")
        else:
            if G_val > 0:
                st.success("✅ El proceso es rentable en todo el rango simulado.")
            else:
                st.warning("⚠️ El proceso no es rentable. Revisá los parámetros económicos.")

    with tab2:
        st.subheader("Análisis económico — Tiempo total desde encendido del equipo")
        fig, ax = plt.subplots(figsize=(11, 5))

        ax.plot(t_total_arr, I_arr,  color='steelblue', lw=2,   label='Ingresos $I(t)$')
        ax.plot(t_total_arr, CT_arr, color='firebrick',  lw=2,   label='Costos totales $CO(t) + C_F$')
        ax.plot(t_total_arr, G_arr,  color='seagreen',   lw=2.5, label='Ganancia neta $G(t)$')
        ax.axhline(0, color='black', lw=0.8)

        # Línea de inicio de destilación
        ax.axvline(t_cal, color='orange', lw=1.5, ls='--',
                   label=f'Inicio destilación (t = {t_cal:.0f} min)')
        ax.axvspan(0, t_cal, alpha=0.05, color='orange')
        ax.text(t_cal/2, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 0,
                'Puesta\nen marcha', ha='center', fontsize=8, color='darkorange')

        if t_eq:
            ax.axvline(t_eq, color='tomato', lw=1.5, ls=':',
                       label=f'Rentable desde {fmt(t_eq,1)} min')
            ax.plot(t_eq, 0, 'o', color='tomato', ms=7)

        ax.set_xlabel('Tiempo desde encendido (min)', fontsize=11)
        ax.set_ylabel('$ (pesos / lote)', fontsize=11)
        ax.set_title('Ganancia neta G(t) en función del tiempo de operación', fontsize=12)
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_xlim([0, t_total_arr[-1]])
        st.pyplot(fig, use_container_width=True)

        if t_eq:
            st.success(f"✅ Rentable desde **{fmt(t_eq,1)} min** desde el encendido.")

except ValueError as ve:
    st.error(f"❌ Error en el cálculo: {ve}")
except Exception as e:
    st.error("❌ Error inesperado.")
    st.exception(e)
