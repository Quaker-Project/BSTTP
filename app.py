import streamlit as st
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point
from datetime import datetime
from scipy.stats import gaussian_kde
import rasterio
from rasterio.transform import from_bounds

# Añadir paquete local
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from bstpp.main import Hawkes_Model
import numpyro.distributions as dist

# --- Variables globales ---
rutas = {"eventos": None, "vial": None}
modelo_entrenado = None
gdf_vial_global = None
gdf_train_global = None

st.title("Hawkes App - Simulator")

# --- Carga de archivos ---
eventos_file = st.file_uploader("Shapefile de eventos (.shp)", type=["shp"])
vial_file = st.file_uploader("Shapefile de red vial (.shp)", type=["shp"])

if eventos_file:
    rutas["eventos"] = eventos_file
if vial_file:
    rutas["vial"] = vial_file

# --- Entradas de fechas ---
st.subheader("Fechas de entrenamiento y simulación")
train_start = st.text_input("Entrenamiento - Fecha inicio (dd/mm/yyyy)")
train_end = st.text_input("Entrenamiento - Fecha fin (dd/mm/yyyy)")
sim_start = st.text_input("Simulación - Fecha inicio (dd/mm/yyyy)")
sim_end = st.text_input("Simulación - Fecha fin (dd/mm/yyyy)")

# --- Parámetros del modelo ---
st.subheader("Parámetros del modelo")
lr = st.number_input("Learning rate (lr)", min_value=0.0001, value=0.02, step=0.001)
num_steps = st.number_input("Número de pasos (num_steps)", min_value=1, value=2000, step=100)
n_sim = st.number_input("Número de simulaciones", min_value=1, value=1, step=1)

# --- Funciones auxiliares ---
def validar_fecha(texto):
    try:
        return datetime.strptime(texto, "%d/%m/%Y")
    except:
        return None

def ejecutar_modelo():
    global modelo_entrenado, gdf_vial_global, gdf_train_global

    if rutas["eventos"] is None or rutas["vial"] is None:
        st.error("Debes cargar ambos shapefiles antes de ejecutar el modelo.")
        return

    try:
        gdf_events = gpd.read_file(rutas["eventos"]).to_crs("EPSG:4326")
        gdf_vial = gpd.read_file(rutas["vial"]).to_crs("EPSG:4326")
        gdf_vial_global = gdf_vial.copy()

        gdf_events["Fecha"] = pd.to_datetime(gdf_events["Fecha"], format="%d/%m/%Y", errors="coerce")
        t0 = gdf_events["Fecha"].min()
        gdf_events["t"] = (gdf_events["Fecha"] - t0).dt.total_seconds() / 86400
        gdf_events = gdf_events.sort_values("t").reset_index(drop=True)

        # Validar fechas
        t_start = validar_fecha(train_start)
        t_end = validar_fecha(train_end)
        s_start = validar_fecha(sim_start)
        s_end = validar_fecha(sim_end)

        if None in [t_start, t_end, s_start, s_end]:
            st.error("Por favor, ingresa fechas válidas en formato dd/mm/yyyy.")
            return

        gdf_train = gdf_events[(gdf_events["Fecha"] >= t_start) & (gdf_events["Fecha"] <= t_end)]
        gdf_test = gdf_events[(gdf_events["Fecha"] >= s_start) & (gdf_events["Fecha"] <= s_end)]

        if gdf_train.empty:
            st.error("No hay datos en el periodo de entrenamiento seleccionado.")
            return

        # Buffer vial
        gdf_buffered = gdf_vial.copy()
        gdf_buffered["geometry"] = gdf_buffered.buffer(0.00015)

        data_model = gdf_train[["t","Long","Lat"]].rename(columns={"t":"T","Long":"X","Lat":"Y"})

        model = Hawkes_Model(
            data=data_model,
            A=gdf_buffered,
            T=gdf_train["t"].max(),
            cox_background=True,
            a_0=dist.Normal(1,10),
            alpha=dist.Beta(20,60),
            beta=dist.HalfNormal(2.0),
            sigmax_2=dist.HalfNormal(0.25)
        )

        model.run_svi(lr=lr, num_steps=num_steps)

        gdf_train_global = gdf_train
        modelo_entrenado = model

        # Resultados
        data_test = gdf_test[["t","Long","Lat"]].rename(columns={"t":"T","Long":"X","Lat":"Y"})
        st.success(f"Log Expected Likelihood: {model.log_expected_likelihood(data_test):.2f}")
        st.success(f"Expected AIC: {model.expected_AIC():.2f}")

        # Gráficas
        fig1 = model.plot_spatial(include_cov=False)
        st.pyplot(fig1)

        fig2 = model.plot_prop_excitation()
        st.pyplot(fig2)

        fig3 = model.plot_temporal()
        st.pyplot(fig3)

    except Exception as e:
        st.error(f"Ocurrió un error: {str(e)}")

# --- Botón para ejecutar ---
if st.button("▶️ Ejecutar Modelo"):
    ejecutar_modelo()
