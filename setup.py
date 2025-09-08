import streamlit as st
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from shapely.geometry import Point
from scipy.stats import gaussian_kde
import numpyro.distributions as dist
from bstpp.main import Hawkes_Model
import rasterio
from rasterio.transform import from_bounds
import io

# --- Variables globales ---
modelo_entrenado = None
gdf_vial_global = None
gdf_train_global = None

st.set_page_config(page_title="Hawkes Simulator", layout="wide")
st.title("Hawkes App - Simulator")

# --- Upload de archivos ---
st.sidebar.header("Archivos de entrada")
uploaded_eventos = st.sidebar.file_uploader("Shapefile de eventos (.shp + .dbf + .shx)", type=["shp"], key="eventos")
uploaded_vial = st.sidebar.file_uploader("Shapefile de red vial (.shp + .dbf + .shx)", type=["shp"], key="vial")

# Parámetros de fechas y modelo
st.sidebar.header("Parámetros")
train_start = st.sidebar.date_input("Entrenamiento - Fecha inicio")
train_end = st.sidebar.date_input("Entrenamiento - Fecha fin")
sim_start = st.sidebar.date_input("Simulación - Fecha inicio")
sim_end = st.sidebar.date_input("Simulación - Fecha fin")

lr = st.sidebar.number_input("Learning rate", value=0.02, min_value=0.0, step=0.001)
num_steps = st.sidebar.number_input("Número de pasos (num_steps)", value=2000, min_value=1, step=1)
n_sim = st.sidebar.number_input("Número de simulaciones", value=1, min_value=1, step=1)

# --- Función para leer shapefile desde Streamlit ---
def read_shapefile(uploaded_file):
    if uploaded_file is None:
        return None
    with io.BytesIO(uploaded_file.read()) as f:
        gdf = gpd.read_file(f)
    return gdf

def ejecutar_modelo():
    global modelo_entrenado, gdf_vial_global, gdf_train_global
    if uploaded_eventos is None or uploaded_vial is None:
        st.error("Debe cargar ambos shapefiles.")
        return

    gdf_events = gpd.read_file(uploaded_eventos).to_crs("EPSG:4326")
    gdf_vial = gpd.read_file(uploaded_vial).to_crs("EPSG:4326")
    gdf_vial_global = gdf_vial.copy()

    for col in ["Fecha", "Long", "Lat"]:
        if col not in gdf_events.columns:
            st.error(f"El shapefile de eventos debe contener columna '{col}'.")
            return

    gdf_events["Fecha"] = pd.to_datetime(gdf_events["Fecha"], format="%d/%m/%Y", errors="coerce")

    gdf_train = gdf_events[(gdf_events["Fecha"] >= pd.to_datetime(train_start)) &
                           (gdf_events["Fecha"] <= pd.to_datetime(train_end))]
    gdf_test = gdf_events[(gdf_events["Fecha"] >= pd.to_datetime(sim_start)) &
                          (gdf_events["Fecha"] <= pd.to_datetime(sim_end))]

    if gdf_train.empty:
        st.error("No hay datos en el periodo de entrenamiento seleccionado.")
        return

    t0 = gdf_train["Fecha"].min()
    gdf_train["t"] = (gdf_train["Fecha"] - t0).dt.total_seconds() / 86400
    gdf_train = gdf_train.sort_values("t").reset_index(drop=True)

    gdf_test["t"] = (gdf_test["Fecha"] - t0).dt.total_seconds() / 86400
    gdf_test = gdf_test.sort_values("t").reset_index(drop=True)

    gdf_buffered = gdf_vial.copy()
    gdf_buffered["geometry"] = gdf_buffered.buffer(0.00015)

    data_model = gdf_train[["t", "Long", "Lat"]].rename(columns={"t": "T", "Long": "X", "Lat": "Y"})

    model = Hawkes_Model(
        data=data_model,
        A=gdf_buffered,
        T=gdf_train["t"].max(),
        cox_background=True,
        a_0=dist.Normal(1, 10),
        alpha=dist.Beta(20, 60),
        beta=dist.HalfNormal(2.0),
        sigmax_2=dist.HalfNormal(0.25)
    )

    with st.spinner("Entrenando modelo... esto puede tardar unos minutos"):
        model.run_svi(lr=lr, num_steps=num_steps)

    data_test = gdf_test[["t", "Long", "Lat"]].rename(columns={"t": "T", "Long": "X", "Lat": "Y"})
    resultados = f"**Log Expected Likelihood:** {model.log_expected_likelihood(data_test):.2f}\n\n"
    resultados += f"**Expected AIC:** {model.expected_AIC():.2f}"

    st.success("Modelo entrenado correctamente")
    st.markdown(resultados)

    modelo_entrenado = model
    gdf_train_global = gdf_train

    st.subheader("Gráficas de diagnóstico")
    fig = model.plot_spatial(include_cov=False)
    st.pyplot(fig)
    fig = model.plot_prop_excitation()
    st.pyplot(fig)
    fig = model.plot_temporal()
    st.pyplot(fig)

def ejecutar_simulacion():
    global modelo_entrenado, gdf_vial_global, gdf_train_global
    if modelo_entrenado is None:
        st.error("Primero debes entrenar un modelo.")
        return

    all_sims = []
    for i in range(n_sim):
        sim_sample = modelo_entrenado.simulate(parameters=None)
        sim_sample['X_orig'] = sim_sample['X']
        sim_sample['Y_orig'] = sim_sample['Y']
        T_passed = gdf_train_global['t'].max()
        time_scale = T_passed / 50.0
        sim_sample['t_days'] = sim_sample['T'] * time_scale
        t0 = gdf_train_global['Fecha'].min()
        sim_sample['Fecha_sim'] = sim_sample['t_days'].apply(lambda x: t0 + pd.Timedelta(days=float(x)))
        sim_sample['geometry'] = sim_sample.apply(lambda r: Point(r['X_orig'], r['Y_orig']), axis=1)
        sim_sample['sim_id'] = i + 1
        all_sims.append(sim_sample)

    sim_gdf = gpd.GeoDataFrame(pd.concat(all_sims, ignore_index=True), geometry='geometry', crs="EPSG:4326")

    st.subheader("Simulación de puntos sobre la red vial")
    fig, ax = plt.subplots(figsize=(8,8))
    gdf_vial_global.plot(ax=ax, edgecolor='gray', alpha=0.7)
    sim_gdf.plot(ax=ax, markersize=4, color='red', alpha=0.3)
    st.pyplot(fig)

    # Exportación Shapefile
    st.subheader("Exportar resultados")
    shp_bytes = io.BytesIO()
    sim_gdf.to_file("/tmp/temp_sim.shp")  # Generamos temporalmente
    with open("/tmp/temp_sim.shp", "rb") as f:
        shp_bytes.write(f.read())
    st.download_button("📥 Descargar shapefile de simulación", shp_bytes.getvalue(), file_name="simulacion.shp")

    # Exportación GeoTIFF (heatmap)
    if n_sim > 1:
        xy = np.vstack([sim_gdf['X_orig'], sim_gdf['Y_orig']])
        kde = gaussian_kde(xy)
        minx, miny, maxx, maxy = gdf_vial_global.total_bounds
        nx, ny = 300, 300
        xx, yy = np.mgrid[minx:maxx:nx*1j, miny:maxy:ny*1j]
        coords = np.vstack([xx.ravel(), yy.ravel()])
        zz = kde(coords).reshape(xx.shape)

        transform = from_bounds(minx, miny, maxx, maxy, nx, ny)
        raster_path = "/tmp/heatmap.tif"
        with rasterio.open(
            raster_path, 'w',
            driver='GTiff',
            height=ny,
            width=nx,
            count=1,
            dtype=zz.dtype,
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(zz.T, 1)

        with open(raster_path, "rb") as f:
            st.download_button("📥 Descargar heatmap GeoTIFF", f.read(), file_name="heatmap.tif")

# --- Botones ---
if st.button("▶️ Ejecutar Modelo"):
    ejecutar_modelo()

if st.button("▶️ Simular"):
    ejecutar_simulacion()
