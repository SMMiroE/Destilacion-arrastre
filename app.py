import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ==============================================================================
# 1. CONSTANTES Y PARÁMETROS (Basados en Tabla 2 y archivos adjuntos)
# ==============================================================================

# Parámetros del calderín
Cpa_BASE = 4.18    # Capacidad calorífica del agua (J/g ºC) 
Tain_BASE = 30.0   # Temperatura del agua de entrada al calderín (ºC) 
Teb_BASE = 100.0   # Temperatura de ebullición (ºC) 
lambda_a_BASE = 2257.0 # Calor específico de vaporización del agua (J/g) 

# Parámetros del material vegetal y difusión
VMV_BASE = 500.0   # Volumen de Material Vegetal (cm³) 
C0_BASE = 0.05     # Concentración inicial de AE en MV (g/cm³) 
h_BASE = 0.2       # Semiespesor de la lámina plana (cm) 
D_BASE = 1.0e-6    # Coeficiente de difusión del AE en el sólido (cm²/s) 

# Parámetros económicos
PrecioAE_BASE = 3000.0 # Precio de mercado del AE ($/cm³) [cite: 275]
PrecioEnv_BASE = 600.0 # Precio del envase de 10 ml ($/unidad) [cite: 273]
VolumenEnv = 10.0  # Volumen del envase (cm³) (10 ml) [cite: 280]
CF_BASE = 25000.0  # Costos Fijos por lote ($/lote) 
rho_AE_BASE = 0.9  # Densidad del AE (g/cm³) [cite: 276]

# Costos de energía eléctrica (Tabla xx y archivo adjunto)
costo_electrico = {
    'Cv': 95.9532     # Cargo variable ($/kWh) 
}

# ==============================================================================
# 2. FUNCIONES DE CÁLCULO (Ecuaciones del documento)
# ==============================================================================

def calcular_mv(PeR, eta_c, Cpa, Teb, Tain, lambda_a):
    """Calcula el caudal de vapor generado mv (g/s) - Ecuación 24."""
    # mv =(1-ηc) PeR / (Cpa*(Teb-Tain)+a)
    numerador = (1 - eta_c) * PeR
    denominador = Cpa * (Teb - Tain) + lambda_a
    if denominador == 0:
        return 0.0
    mv = numerador / denominador
    return mv 

def mAE_acum(tf, F1, F2, n_terms=10):
    """Calcula la masa acumulada de AE en el tiempo tf (g) - Ecuación 13."""
    mAE_total = 0.0
    for n in range(n_terms):
        term_factor = (2 * n + 1)**2
        if F2 * term_factor == 0:
            continue
        mAE_total += (1 - np.exp(-F2 * term_factor * tf)) / (F2 * term_factor)
    
    mAE_acum_val = F1 * mAE_total
    return mAE_acum_val 

def calcular_xc(mAE_acum_val, mv, tf):
    """Calcula la composición del condensado xc (masa de AE/masa de condensado) - Ecuación 16."""
    mvacum = mv * tf
    mc_t = mAE_acum_val + mvacum
    if mc_t == 0:
        return 0.0
    xc_val = mAE_acum_val / mc_t
    return xc_val

def calcular_rendimiento(mAE_acum_val, VMV, C0):
    """Calcula el rendimiento porcentual (%Rend) - Ecuación 20."""
    MAE0 = VMV * C0 # Masa inicial de AE (Ecuación 17)
    if MAE0 == 0:
        return 0.0
    # Ecuación 20: %Rend=MAE(t)/MAE0 * 100.
    # El documento parece tener un error en la Ecuación 20 y usa MAE(t) (remanente), pero la definición es de mAE_acum.
    # Usaremos la definición estándar (masa extraída/masa inicial) que es consistente con el resultado de 41.24%.
    return (mAE_acum_val / MAE0) * 100.0

def mAE_acum_vol(mAE_acum_val, rho_AE):
    """Convierte la masa acumulada de AE a volumen (cm³)."""
    if rho_AE == 0:
        return 0.0
    return mAE_acum_val / rho_AE 

def ingresos(mAE_acum_val, PrecioAE, rho_AE):
    """Calcula los ingresos I(tf) ($) - Ecuación 31."""
    V_AE = mAE_acum_vol(mAE_acum_val, rho_AE)
    I_tf = V_AE * PrecioAE
    return I_tf 

def costo_operacion(tf, PeR, Cv, PrecioEnv, rho_AE, VolumenEnv, mAE_acum_val):
    """Calcula el costo de operación CO(tf) ($) - Ecuación 32."""
    
    # 1. Costo Eléctrico (Celect) - Solo la parte variable por lote (tf) para CO(tf).
    # PeR en W, tf en s. Cv en $/kWh. Convertimos PeR*tf a kWh
    PeR_kWh = PeR / 1000.0 
    tf_h = tf / 3600.0     
    Celect_tf = Cv * PeR_kWh * tf_h 
    
    # 2. Número de envases (N)
    V_AE = mAE_acum_vol(mAE_acum_val, rho_AE)
    N = np.ceil(V_AE / VolumenEnv)
    
    # 3. CO(tf) = Celect(tf) + N * PrecioEnv (Ecuación 32)
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
st.markdown("Cálculo de Rendimiento, Calidad y Ganancia Neta de la extracción de Aceite Esencial de *Schinus Areira*.")

col_op, col_mat, col_econ = st.columns([1, 1, 1])

# --- Columna 1: Parámetros de Operación ---
with col_op:
    st.header("1. Condiciones de Operación")
    
    PeR = st.number_input("Potencia de Resistencia ($P_{eR}$, W)", value=1000.0, step=100.0, format="%.1f")
    eta_c = st.slider("Fracción de Calor Perdido ($\eta_c$)", min_value=0.0, max_value=0.2, value=0.1, step=0.01, format="%.2f")
    Tain = st.number_input("Temp. Agua de Entrada ($T_{in}$, ºC)", value=Tain_BASE, step=1.0, format="%.1f")
    
    st.subheader("Tiempo de Operación")
    tf_min = st.slider("Tiempo Total ($t_f$, min)", min_value=1, max_value=300, value=120, step=5)
    
    tf_s = tf_min * 60.0 # Convertir a segundos

# --- Columna 2: Parámetros de Material y Difusión ---
with col_mat:
    st.header("2. Parámetros del Material Vegetal")
    
    VMV = st.number_input("Volumen de MV ($V_{MV}$, cm³)", value=VMV_BASE, step=10.0, format="%.1f")
    C0 = st.number_input("Conc. Inicial AE ($C_0$, g/cm³)", value=C0_BASE, step=0.005, format="%.4f")
    h = st.number_input("Semiespesor del MV ($h$, cm)", value=h_BASE, step=0.01, format="%.3f")
    # Multiplicamos por 1e-6 ya que el input es el factor 10^-6
    D = st.number_input(r"Coef. Difusión ($D, 10^{-6}$ cm²/s)", value=D_BASE * 1e6, step=0.1, format="%.3f") * 1e-6
    rho_AE = st.number_input("Densidad AE (g/cm³)", value=rho_AE_BASE, step=0.05, format="%.2f")

# --- Columna 3: Parámetros Económicos ---
with col_econ:
    st.header("3. Parámetros Económicos")
    
    PrecioAE = st.number_input("Precio AE ($\$/cm³$)", value=PrecioAE_BASE, step=100.0, format="%.2f")
    PrecioEnv = st.number_input("Costo Envase ($\$/unidad$)", value=PrecioEnv_BASE, step=10.0, format="%.2f")
    CF = st.number_input("Costos Fijos ($C_F$, $\$/lote$)", value=CF_BASE, step=1000.0, format="%.2f")
    Cv = st.number_input("Costo variable $C_v$ ($\$/kWh$)", value=costo_electrico['Cv'], step=1.0, format="%.2f")
    
st.markdown("---")

# ==============================================================================
# 4. CÁLCULOS PRINCIPALES Y RESULTADOS
# ==============================================================================

try:
    # 4.1. Cálculos de Factores y Caudal de Vapor
    mv = calcular_mv(PeR, eta_c, Cpa_BASE, Teb_BASE, Tain, lambda_a_BASE)
    
    # Factores F1 y F2 para la difusión (Ecuación 82)
    F1 = 2 * VMV * C0 * D / (h**2) # [cite: 82]
    F2 = D * np.pi**2 / (4 * h**2) # [cite: 82]
    
    # Validaciones básicas
    if mv <= 0 or F1 <= 0 or F2 <= 0:
        raise ValueError("El Caudal de Vapor o los Factores de Difusión son cero o negativos. Revise $P_{eR}$, $\eta_c$, $D$, $V_{MV}$, $C_0$ o $h$.")
    
    # 4.2. Cálculos de Balance de Materia
    mAE_acum_val = mAE_acum(tf_s, F1, F2) # [cite: 97]
    xc_val = calcular_xc(mAE_acum_val, mv, tf_s) # [cite: 103]
    rendimiento_val = calcular_rendimiento(mAE_acum_val, VMV, C0) # [cite: 115]
    
    # 4.3. Cálculos Económicos
    I_tf = ingresos(mAE_acum_val, PrecioAE, rho_AE) # [cite: 193]
    CO_tf, N_envases = costo_operacion(tf_s, PeR, Cv, PrecioEnv, rho_AE, VolumenEnv, mAE_acum_val) # 
    G_tf = ganancia_neta(I_tf, CO_tf, CF) # 
    V_AE_val = mAE_acum_vol(mAE_acum_val, rho_AE)
    
    # 4.4. Presentación de Resultados
    st.header("Resultado de la Simulación a $t_f = {} \min$".format(tf_min))
    
    col_out1, col_out2, col_out3 = st.columns(3)
    
    with col_out1:
        st.metric("Rendimiento %Rend", "{:.2f} %".format(rendimiento_val))
        st.metric("Calidad del Producto $x_c$", "{:.4f} g AE/g cond.".format(xc_val))
        st.caption("Masa de AE Acumulada: {:.2f} g".format(mAE_acum_val))

    with col_out2:
        st.metric("Ganancia Neta $G$", "$ARG {:.2f}".format(G_tf), 
                  help="Ingresos - Costos de Operación - Costos Fijos (Ecuación 30).")
        st.metric("Volumen de AE Producido", "{:.2f} cm³".format(V_AE_val))
        st.caption("Caudal de Vapor $m_v$: {:.3f} g/s".format(mv))

    with col_out3:
        st.metric("Ingresos $I(t_f)$", "$ARG {:.2f}".format(I_tf))
        st.metric("Costo Op. $CO(t_f)$", "$ARG {:.2f}".format(CO_tf))
        st.caption("Envases requeridos: {} unidades".format(int(N_envases)))
        
except ValueError as ve:
    st.error(f"Error en el cálculo: {ve}")
except Exception as e:
    st.error("Error en el cálculo: Por favor, revise los parámetros de entrada.")
    
st.markdown("---")

# --- 5. GRÁFICA DE LA GANANCIA NETA ---

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
            continue

    if resultados_plot:
        df_plot = pd.DataFrame(resultados_plot)

        # Encontrar el punto de equilibrio (G=0)
        if df_plot['Ganancia_neta'].max() > 0:
            tiempo_equilibrio_min = np.interp(0, df_plot['Ganancia_neta'], df_plot['Tiempo_min'])
        else:
            tiempo_equilibrio_min = 0 
        
        # Generar el gráfico
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df_plot['Tiempo_min'], df_plot['Ganancia_neta'], label='Ganancia Neta $G(t)$', color='green')
        ax.axhline(0, color='red', linestyle='--', linewidth=0.8, label='Punto de Equilibrio ($G=0$)')
        
        if tiempo_equilibrio_min > 0 and tiempo_equilibrio_min < df_plot['Tiempo_min'].max():
            ax.axvline(tiempo_equilibrio_min, color='red', linestyle=':', linewidth=0.8)
            ax.plot(tiempo_equilibrio_min, 0, 'ro', label=f'Rentable a {tiempo_equilibrio_min:.1f} min')

        ax.set_title('Ganancia Neta Obtenida vs. Tiempo de Operación', fontsize=12)
        ax.set_xlabel('Tiempo de Operación ($t$) [min]', fontsize=10)
        ax.set_ylabel('Ganancia Neta $G$ [$ARG]', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        st.pyplot(fig)
        
        if tiempo_equilibrio_min > 0 and tiempo_equilibrio_min < df_plot['Tiempo_min'].max():
            # El valor de 31.1 min es el reportado por el documento 
            # Si el cálculo lo reproduce (debe ser con los parámetros base), se muestra ese.
            # Sino, se muestra el valor calculado.
            st.success(f"El proceso comienza a ser rentable a partir de los **{tiempo_equilibrio_min:.1f} minutos**.")
        else:
            st.warning("El tiempo de operación actual es insuficiente para cubrir los Costos Fijos.")
    else:
        st.warning("El tiempo de operación es demasiado corto para generar una curva de ganancia significativa (mínimo 1 minuto).")
else:
     st.warning("Aumenta el tiempo de operación (mínimo 60 min) para ver el gráfico de Ganancia Neta.")
