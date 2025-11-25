import numpy as np
import pandas as pd
import streamlit as st

# ==============================================================================
# 1. CONSTANTES Y PARÁMETROS (Basados en Tabla 2 y texto)
# ==============================================================================

# Parámetros del calderín
Cpa_BASE = 4.18    # Capacidad calorífica del agua (J/g ºC) [cite: 216]
Tain_BASE = 30.0   # Temperatura del agua de entrada al calderín (ºC) [cite: 216]
Teb_BASE = 100.0   # Temperatura de ebullición (ºC) [cite: 216]
lambda_a_BASE = 2257.0 # Calor específico de vaporización del agua (J/g) [cite: 216]

# Parámetros del material vegetal y difusión
VMV_BASE = 500.0   # Volumen de Material Vegetal (cm³) [cite: 216]
C0_BASE = 0.05     # Concentración inicial de AE en MV (g/cm³) [cite: 216]
h_BASE = 0.2       # Semiespesor de la lámina plana (cm) [cite: 216]
D_BASE = 1.0e-6    # Coeficiente de difusión del AE en el sólido (cm²/s) [cite: 216]
P_BASE = 1.0       # Presión de operación (atm) [cite: 216]

# Parámetros económicos
PrecioAE_BASE = 3000.0 # Precio de mercado del AE ($/cm³) [cite: 275]
PrecioEnv_BASE = 600.0 # Precio del envase de 10 ml ($/unidad) [cite: 273]
VolumenEnv = 10.0  # Volumen del envase (cm³)
CF_BASE = 25000.0  # Costos Fijos por lote ($/lote) [cite: 274]
rho_AE_BASE = 0.9  # Densidad del AE (g/cm³) [cite: 276]

# Costos de energía eléctrica (Tabla xx)
costo_electrico = {
    'Cf': 7250.9896,  # Cargo Fijo ($/mes) [cite: 200]
    'Cr': 16690.7292, # Cargo Uso de red ($/mes) [cite: 200]
    'Cv': 95.9532     # Cargo variable ($/kWh) [cite: 200]
}
# Factor impositivo: 1 + IVA (21%) + Contr Municipal (6.383%) = 1.27383 (implícito en Ec. 33)
# Usaremos solo el costo variable para CO(tf) [cite: 202, 197]
factor_impositivo = 1.27383 


# ==============================================================================
# 2. FUNCIONES DE CÁLCULO (Ecuaciones del documento)
# ==============================================================================

def calcular_mv(PeR, eta_c, Cpa, Teb, Tain, lambda_a):
    """Calcula el caudal de vapor generado mv (g/s) - Ecuación 24."""
    # mv =(1-ηc) PeR / (Cpa*(Teb-Tain)+a)
    numerador = (1 - eta_c) * PeR
    denominador = Cpa * (Teb - Tain) + lambda_a
    # La PeR debe ser en J/s (W). La Cpa y lambda_a en J/g. mv en g/s.
    mv = numerador / denominador
    return mv 

def mAE_acum(tf, F1, F2, n_terms=10):
    """Calcula la masa acumulada de AE en el tiempo tf (g) - Ecuación 13."""
    mAE_total = 0.0
    # Sumatoria desde n=0 hasta 9
    for n in range(n_terms):
        term_factor = (2 * n + 1)**2
        # Ecuación 13: F1 * Sumatoria [ (1 - e^(-F2 * (2n+1)^2 * t)) / (F2 * (2n+1)^2) ]
        mAE_total += (1 - np.exp(-F2 * term_factor * tf)) / (F2 * term_factor)
    
    mAE_acum_val = F1 * mAE_total
    return mAE_acum_val 

def calcular_xc(mAE_acum_val, mv, tf, F1, F2):
    """Calcula la composición del condensado xc (masa de AE/masa de condensado) - Ecuación 16."""
    # mvacum = mv * tf (Ecuación 14)
    mvacum = mv * tf
    # mc(t) = mAE_acum(t) + mv * t (Ecuación 15)
    mc_t = mAE_acum_val + mvacum
    # xc = mAE_acum / mc (Ecuación 16)
    if mc_t == 0:
        return 0
    xc_val = mAE_acum_val / mc_t
    return xc_val

def calcular_rendimiento(mAE_acum_val, VMV, C0):
    """Calcula el rendimiento porcentual (%Rend) - Ecuación 20."""
    MAE0 = VMV * C0 # Masa inicial de AE (Ecuación 17)
    # %Rend = (mAE_acum / MAE0) * 100 (Ecuación 20, con un error en la fuente que usa MAE(t))
    # Asumimos la definición estándar de Rendimiento = (Masa extraída / Masa inicial) * 100
    if MAE0 == 0:
        return 0
    return (mAE_acum_val / MAE0) * 100.0

# --- Funciones Económicas ---

def mAE_acum_vol(mAE_acum_val, rho_AE):
    """Convierte la masa acumulada de AE a volumen (cm³)."""
    if rho_AE == 0:
        return 0
    return mAE_acum_val / rho_AE 

def ingresos(mAE_acum_val, PrecioAE, rho_AE):
    """Calcula los ingresos I(tf) ($) - Ecuación 31."""
    # I(tf) = mAE_acum(tf) * (PrecioAE / rho_AE)
    V_AE = mAE_acum_vol(mAE_acum_val, rho_AE)
    I_tf = V_AE * PrecioAE
    return I_tf 

def costo_operacion(tf, PeR, Cv, PrecioEnv, rho_AE, VolumenEnv, mAE_acum_val):
    """Calcula el costo de operación CO(tf) ($) - Ecuación 32."""
    
    # 1. Costo Eléctrico (Celect) - Usamos solo el término variable para el costo operativo por lote.
    # PeR en W, tf en s. Cv en $/kWh. Convertimos PeR*tf a kWh
    PeR_kWh = PeR / 1000.0 
    tf_h = tf / 3600.0     
    Celect_tf = Cv * PeR_kWh * tf_h 
    
    # 2. Número de envases (N)
    V_AE = mAE_acum_vol(mAE_acum_val, rho_AE)
    N = np.ceil(V_AE / VolumenEnv)
    
    # 3. CO(tf) = Celect(tf) + N * PrecioEnv
    CO_tf = Celect_tf + N * PrecioEnv
    return CO_tf, N

def ganancia_neta(I_tf, CO_tf, CF):
    """Calcula la ganancia neta G(tf) ($) - Ecuación 30."""
    # G(tf) = I(tf) - CO(tf) - CF
    G_tf = I_tf - CO_tf - CF
    return G_tf 

# ==============================================================================
# 3. INTERFAZ STREAMLIT
# ==============================================================================

st.set_page_config(layout="wide", page_title="Simulador de Destilación AE")

st.title("🌱 Simulador de Destilación por Arrastre con Vapor")
st.markdown("Calcula el Rendimiento, la Calidad y la Ganancia Neta de la extracción de Aceite Esencial (AE) de *Schinus Areira* [cite: 206] basado en el modelo de difusión-controlada[cite: 26, 284].")

col_op, col_mat, col_econ = st.columns([1, 1, 1])

# --- Columna 1: Parámetros de Operación (Calderín/Vapor) ---
with col_op:
    st.header("1. Condiciones de Operación")
    
    PeR = st.number_input("Potencia de Resistencia ($P_{eR}$, W) [cite: 216]", value=1000.0, step=100.0, format="%.1f")
    eta_c = st.slider("Fracción de Calor Perdido ($\eta_c$) [cite: 216]", min_value=0.0, max_value=0.2, value=0.1, step=0.01, format="%.2f")
    Tain = st.number_input("Temp. Agua de Entrada ($T_{in}$, ºC) [cite: 216]", value=Tain_BASE, step=1.0, format="%.1f")
    
    st.subheader("Tiempo de Operación")
    tf_min = st.slider("Tiempo Total ($t_f$, min)", min_value=1, max_value=300, value=120, step=5)
    
    tf_s = tf_min * 60.0 # Convertir a segundos para los cálculos

# --- Columna 2: Parámetros de Material y Difusión ---
with col_mat:
    st.header("2. Parámetros del Material Vegetal")
    
    VMV = st.number_input("Volumen de MV ($V_{MV}$, cm³) [cite: 216]", value=VMV_BASE, step=10.0, format="%.1f")
    C0 = st.number_input("Conc. Inicial AE ($C_0$, g/cm³) [cite: 216]", value=C0_BASE, step=0.005, format="%.4f")
    h = st.number_input("Semiespesor del MV ($h$, cm) [cite: 216]", value=h_BASE, step=0.01, format="%.3f")
    D = st.number_input("Coef. Difusión ($D, 10^{-6}$ cm²/s) [cite: 216]", value=D_BASE * 1e6, step=0.1, format="%.3f") * 1e-6
    rho_AE = st.number_input("Densidad AE (g/cm³) [cite: 276]", value=rho_AE_BASE, step=0.05, format="%.2f")

# --- Columna 3: Parámetros Económicos ---
with col_econ:
    st.header("3. Parámetros Económicos")
    
    PrecioAE = st.number_input("Precio AE (\$/cm³) [cite: 275]", value=PrecioAE_BASE, step=100.0, format="%.2f")
    PrecioEnv = st.number_input("Costo Envase (\$/unidad) [cite: 273]", value=PrecioEnv_BASE, step=10.0, format="%.2f")
    CF = st.number_input("Costos Fijos ($C_F$, \$/lote) [cite: 274]", value=CF_BASE, step=1000.0, format="%.2f")
    Cv = st.number_input("Costo variable $C_v$ (\$/kWh) [cite: 200]", value=costo_electrico['Cv'], step=1.0, format="%.2f")
    
# Separador de secciones
st.markdown("---")

# ==============================================================================
# 4. CÁLCULOS PRINCIPALES Y RESULTADOS
# ==============================================================================

# 4.1. Cálculos de Factores y Caudal de Vapor
try:
    # Caudal de vapor (constante)
    mv = calcular_mv(PeR, eta_c, Cpa_BASE, Teb_BASE, Tain)
    
    # Factores F1 y F2 para la difusión (Ecuación 82)
    F1 = 2 * VMV * C0 * D / (h**2)
    F2 = D * np.pi**2 / (4 * h**2)
    
    # 4.2. Cálculos de Balance de Materia
    mAE_acum_val = mAE_acum(tf_s, F1, F2)
    xc_val = calcular_xc(mAE_acum_val, mv, tf_s, F1, F2)
    rendimiento_val = calcular_rendimiento(mAE_acum_val, VMV, C0)
    
    # 4.3. Cálculos Económicos
    I_tf = ingresos(mAE_acum_val, PrecioAE, rho_AE)
    CO_tf, N_envases = costo_operacion(tf_s, PeR, Cv, PrecioEnv, rho_AE, VolumenEnv, mAE_acum_val)
    G_tf = ganancia_neta(I_tf, CO_tf, CF)
    V_AE_val = mAE_acum_vol(mAE_acum_val, rho_AE)
    
    # 4.4. Presentación de Resultados
    st.header("Resultado de la Simulación a $t_f = {} \min$".format(tf_min))
    
    col_out1, col_out2, col_out3 = st.columns(3)
    
    with col_out1:
        st.metric("Rendimiento %Rend", "{:.2f} %".format(rendimiento_val), 
                  help="Masa de AE extraída respecto a la masa inicial (Ecuación 20).")
        st.metric("Calidad del Producto $x_c$", "{:.4f} g AE/g cond.".format(xc_val), 
                  help="Masa de AE sobre masa total de condensado (Ecuación 16).")
        st.caption("Masa de AE Acumulada: {:.2f} g".format(mAE_acum_val))

    with col_out2:
        st.metric("Ganancia Neta $G$", "US$ {:.2f}".format(G_tf), 
                  help="Ingresos - Costos de Operación - Costos Fijos (Ecuación 30).")
        st.metric("Volumen de AE Producido", "{:.2f} cm³".format(V_AE_val), 
                  help="Volumen total de Aceite Esencial puro.")
        st.caption("Caudal de Vapor $m_v$: {:.3f} g/s".format(mv))

    with col_out3:
        st.metric("Ingresos $I(t_f)$", "US$ {:.2f}".format(I_tf))
        st.metric("Costo Op. $CO(t_f)$", "US$ {:.2f}".format(CO_tf))
        st.caption("Envases requeridos: {} unidades".format(int(N_envases)))
        
except Exception as e:
    st.error("Error en el cálculo: Por favor, revise los parámetros de entrada.")
    st.exception(e)

st.markdown("---")

st.info("Para correr esta aplicación en tu PC, guarda el código como `app.py` y ejecútalo con `streamlit run app.py` en tu terminal.")

# --- 5. GRÁFICA DE LA GANANCIA NETAA ---

st.subheader("Gráfico de Ganancia Neta vs. Tiempo")

if tf_min >= 60:
    # 5.1 Recalculo para la gráfica
    time_points_s_plot = np.arange(60, tf_s + 60, 60)
    resultados_plot = []

    for tf_s_plot in time_points_s_plot:
        try:
            mAE_acum_val_plot = mAE_acum(tf_s_plot, F1, F2)
            I_tf_plot = ingresos(mAE_acum_val_plot, PrecioAE, rho_AE)
            CO_tf_plot, _ = costo_operacion(tf_s_plot, PeR, Cv, PrecioEnv, rho_AE, VolumenEnv, mAE_acum_val_plot)
            G_tf_plot = ganancia_neta(I_tf_plot, CO_tf_plot, CF)
            
            resultados_plot.append({
                'Tiempo_min': tf_s_plot / 60.0,
                'Ganancia_neta': G_tf_plot
            })
        except:
            # En caso de error en algún punto, se salta el punto
            continue

    if resultados_plot:
        df_plot = pd.DataFrame(resultados_plot)

        # Encontrar el punto de equilibrio (G=0)
        tiempo_equilibrio_s = np.interp(0, df_plot['Ganancia_neta'], df_plot['Tiempo_min'])
        
        # Generar el gráfico
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df_plot['Tiempo_min'], df_plot['Ganancia_neta'], label='Ganancia Neta G(t)', color='green')
        ax.axhline(0, color='red', linestyle='--', linewidth=0.8, label='Punto de Equilibrio (G=0)')
        
        # Marcar el punto de equilibrio
        if tiempo_equilibrio_s < df_plot['Tiempo_min'].max():
            ax.axvline(tiempo_equilibrio_s, color='red', linestyle=':', linewidth=0.8)
            ax.plot(tiempo_equilibrio_s, 0, 'ro', label=f'Rentable a {tiempo_equilibrio_s:.1f} min')

        ax.set_title('Ganancia Neta Obtenida vs. Tiempo de Operación', fontsize=12)
        ax.set_xlabel('Tiempo de Operación ($t$) [min]', fontsize=10)
        ax.set_ylabel('Ganancia Neta $G$ [\$]', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        st.pyplot(fig)
        
        if tiempo_equilibrio_s < df_plot['Tiempo_min'].max():
            st.success(f"El proceso comienza a ser rentable a partir de los **{tiempo_equilibrio_s:.1f} minutos**.")
        else:
            st.warning("El tiempo de operación actual es insuficiente para cubrir los Costos Fijos.")
    else:
        st.warning("El tiempo de operación es demasiado corto para generar una curva de ganancia significativa (mínimo 1 minuto).")
else:
     st.warning("Aumenta el tiempo de operación para ver el gráfico de Ganancia Neta.")