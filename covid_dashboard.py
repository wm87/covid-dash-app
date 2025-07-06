# 📦 Imports
from concurrent.futures import ThreadPoolExecutor

import folium
import geopandas as gpd
import numpy as np
import plotly.express as px
import polars as pl
import streamlit as st
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# 🌐 Datenquellen (CSSE GitHub)
BASE_URL = (
    "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
    "csse_covid_19_data/csse_covid_19_time_series/"
)
URLS = {
    "Confirmed": BASE_URL + "time_series_covid19_confirmed_global.csv",
    "Deaths": BASE_URL + "time_series_covid19_deaths_global.csv",
    "Recovered": BASE_URL + "time_series_covid19_recovered_global.csv",
}

# 🔧 Länder-Namen vereinheitlichen
country_name_corrections = {
    "United States of America": "US",
    "UK": "United Kingdom",
    "Great Britain": "United Kingdom",
    "England": "United Kingdom",
    "Russian Federation": "Russia",
    "South Korea": "Republic of Korea",
    "North Korea": "Democratic People's Republic of Korea",
    "Iran": "Iran (Islamic Republic of)",
    "Vietnam": "Viet Nam",
    "Syria": "Syrian Arab Republic",
    "Laos": "Lao People's Democratic Republic",
    "Bolivia": "Bolivia (Plurinational State of)",
    "Tanzania": "United Republic of Tanzania",
    "Venezuela": "Venezuela (Bolivarian Republic of)",
    "Brunei": "Brunei Darussalam",
    "Moldova": "Republic of Moldova",
    "Palestine": "Palestine, State of",
    "Czechia": "Czech Republic",
    "Ivory Coast": "Côte d'Ivoire",
    "Cape Verde": "Cabo Verde",
    "Swaziland": "Eswatini",
    "Burma": "Myanmar",
    "Republic of Ireland": "Ireland",
    "Micronesia": "Micronesia (Federated States of)",
    "Macau": "Macao",
    "Saint Martin": "Saint Martin (French part)",
    "Curaçao": "Curacao",
    "French Guiana": "French Guiana",
    "Guinea-Bissau": "Guinea Bissau",
    "Hong Kong": "Hong Kong",
    "Puerto Rico": "Puerto Rico",
    "Guam": "Guam",
    "American Samoa": "American Samoa",
    "Faroe Islands": "Faroe Islands",
    "Türkiye": "Turkey",
    "AU": "Australia",
    "AUS": "Australia",
    "Saudi Arabia": "Saudi Arabia",
    "Saudi-Arabien": "Saudi Arabia",
    "Kingdom of Saudi Arabia": "Saudi Arabia",
}

# 🌍 Shapefile für Ländergrenzen
NE_SHAPEFILE_PATH = "ne_10m_admin_0_countries.shp"
country_col = "ADMIN"

# ---------------------------------------------------
# 📥 Datenladen und Vorverarbeitung
# ---------------------------------------------------

@st.cache_data
def load_covid_csvs() -> dict[str, pl.DataFrame]:
    return {k: pl.read_csv(url) for k, url in URLS.items()}

@st.cache_data
def load_shapefile(path: str) -> gpd.GeoDataFrame:
    world = gpd.read_file(path)
    world_proj = world.to_crs(epsg=3857)
    world_proj['centroid'] = world_proj.geometry.centroid
    world['centroid'] = world_proj['centroid'].to_crs(epsg=4326)
    world['Longitude'] = world['centroid'].x
    world['Latitude'] = world['centroid'].y
    return world

def standardize_country_names(raw_list: list[str], shapefile_names: set[str]) -> list[str]:
    name_map = {v: k for k, v in country_name_corrections.items()}
    result = []
    for name in raw_list:
        if name in shapefile_names:
            result.append(name)
        elif name in country_name_corrections:
            result.append(country_name_corrections[name])
        elif name in name_map:
            result.append(name_map[name])
        else:
            result.append(name)
    return sorted(set(result))

@st.cache_data
def build_country_coords(_world_df: gpd.GeoDataFrame, countries: list[str]) -> dict[str, tuple[float, float]]:
    coords_df = _world_df[[country_col, "Longitude", "Latitude"]].copy()
    coords_df.rename(columns={country_col: "Country"}, inplace=True)

    def get_coords(name: str):
        corrected = country_name_corrections.get(name, name)
        row = coords_df[coords_df["Country"] == corrected]
        if not row.empty:
            return name, (row.iloc[0]["Latitude"], row.iloc[0]["Longitude"])
        return name, (np.nan, np.nan)

    with ThreadPoolExecutor() as executor:
        return dict(executor.map(get_coords, countries))

@st.cache_data(ttl=300)
def prepare_data(countries: list[str], metric: str, avg: str, per_100k: bool,
                 dfs: dict, coords: dict, population: dict, continent_map: dict) -> pl.DataFrame:
    dfs_list = []
    for country in countries:
        df_c = dfs["Confirmed"].filter(pl.col("Country/Region") == country)
        df_d = dfs["Deaths"].filter(pl.col("Country/Region") == country)
        df_r = dfs["Recovered"].filter(pl.col("Country/Region") == country)

        sum_c = df_c.drop(["Province/State", "Lat", "Long", "Country/Region"]).sum()
        sum_d = df_d.drop(["Province/State", "Lat", "Long", "Country/Region"]).sum()
        sum_r = df_r.drop(["Province/State", "Lat", "Long", "Country/Region"]).sum()

        dates = sum_c.columns
        df = pl.DataFrame({
            "Date": dates,
            "Confirmed": sum_c.row(0),
            "Deaths": sum_d.row(0),
            "Recovered": sum_r.row(0),
        }).with_columns(pl.col("Date").str.to_date("%m/%d/%y")).sort("Date")

        df = df.with_columns([
            pl.col("Confirmed").diff().fill_null(0).alias("New Confirmed"),
            (pl.col("Confirmed") - pl.col("Deaths") - pl.col("Recovered")).alias("Active")
        ])

        if avg != "None":
            df = df.with_columns(
                pl.col(metric if metric != "New Confirmed" else "New Confirmed")
                .rolling_mean(window_size=int(avg), min_samples=1)
                .alias(metric)
            )

        if per_100k:
            pop = population.get(country, 1_000_000)
            for col in ["Confirmed", "Deaths", "Recovered", "Active", "New Confirmed"]:
                df = df.with_columns((pl.col(col) / pop * 100_000).alias(col))

        lat, lon = coords.get(country, (np.nan, np.nan))
        df = df.with_columns([
            pl.lit(country).alias("Country"),
            pl.lit(continent_map.get(country, "Other")).alias("Continent"),
            pl.lit(lat).alias("Lat"),
            pl.lit(lon).alias("Long"),
        ])

        dfs_list.append(df)

    return pl.concat(dfs_list).sort(["Country", "Date"])

# ✂️ Ab hier: Alles gleich wie in deinem bisherigen Code...

# ---------------------------------------------------
# 🌐 Initialisierung & UI
# ---------------------------------------------------

dfs = load_covid_csvs()
world = load_shapefile(NE_SHAPEFILE_PATH)

shapefile_countries = set(world[country_col].unique())
raw_countries = dfs["Confirmed"]["Country/Region"].unique().to_list()
all_countries = standardize_country_names(raw_countries, shapefile_countries)
country_coords = build_country_coords(world, all_countries)

# Bevölkerungen (vereinfachend: gleichmäßig)
country_population = {c: 1_000_000 for c in all_countries}

# Kontinente zuordnen
country_to_continent = {}
for _, row in world.iterrows():
    name = row[country_col]
    country_to_continent[name] = row["CONTINENT"]
    for k, v in country_name_corrections.items():
        if v == name:
            country_to_continent[k] = row["CONTINENT"]

# 🎛️ Streamlit Konfiguration
st.set_page_config(layout="wide")
st.title("🦠 COVID-19 Ultimate Dashboard")

# Sidebar – Erst Auswahl (ohne Metrik)
with st.sidebar:
    view = st.radio("Anzeige nach:", ["Countries", "Continents"], index=0)
    continents = sorted(set(world['CONTINENT'].dropna().unique()))

    if view == "Countries":
        # Standardmäßig ausgewählte Länder, nur wenn sie existieren
        default_countries = [c for c in ["US", "India", "Brazil", "Germany", "Turkey", "UK", "France", "China"] if c in all_countries]
        countries = st.multiselect("Länder auswählen", sorted(all_countries), default=default_countries)
    else:
        # Standardmäßig Europa und Asien
        sel_cont = st.multiselect("Kontinente auswählen", continents, default=["Europe", "Asia"])
        countries = [c for c in all_countries if country_to_continent.get(c) in sel_cont]

# 📊 Tabs
tab_names = ["📈 Verlauf", "🌡️ Heatmap", "🌍 Karte", "🔥 Folium", "📥 CSV", "📸 Export"]

if "active_tab" not in st.session_state:
    st.session_state.active_tab = tab_names[0]

def update_tab():
    st.session_state.active_tab = st.session_state.selected_tab

selected_tab = st.selectbox("Ansicht auswählen:", tab_names,
                            index=tab_names.index(st.session_state.active_tab),
                            key="selected_tab", on_change=update_tab)

# ⛔ Kein Land gewählt?
if not countries:
    st.warning("Bitte mindestens ein Land auswählen.")
    st.stop()

# Metrik-Steuerung NUR für visuelle Tabs
show_metric_controls = selected_tab not in ["📥 CSV", "📸 Export"]

if show_metric_controls:
    with st.sidebar:
        metric = st.selectbox("Metrik", ["Confirmed", "Deaths", "Recovered", "New Confirmed"])
        per_100k = st.checkbox("pro 100.000 Einwohner")
        avg_choice = st.selectbox("Gleitender Durchschnitt", ["None", "7", "14", "30"], index=1)
else:
    # Platzhalter
    metric = "Confirmed"
    per_100k = False
    avg_choice = "None"

# 📊 Daten vorbereiten
combined_df = prepare_data(countries, metric, avg_choice, per_100k, dfs,
                           country_coords, country_population, country_to_continent)

# 📆 Zeitraumfilter
min_date = combined_df["Date"].min()
max_date = combined_df["Date"].max()
date_range = st.slider("Zeitraum wählen:", min_value=min_date, max_value=max_date,
                       value=(min_date, max_date), format="DD.MM.YYYY")

filtered_df = combined_df.filter((pl.col("Date") >= date_range[0]) & (pl.col("Date") <= date_range[1]))
plot_df = filtered_df.to_pandas()

# 📈 Verlauf
if selected_tab == "📈 Verlauf":
    st.subheader("Verlauf der Fälle")
    if view == "Countries":
        fig = px.line(plot_df, x="Date", y=metric, color="Country", title="Zeitverlauf")
    else:
        grp = plot_df.groupby(["Date", "Continent"]).sum(numeric_only=True).reset_index()
        fig = px.line(grp, x="Date", y=metric, color="Continent", title="Zeitverlauf")
    st.plotly_chart(fig, use_container_width=True)

# 🌡️ Heatmap
elif selected_tab == "🌡️ Heatmap":
    st.subheader("Heatmap der Fälle nach Datum/Land")
    cmap = st.selectbox("Farbschema", ["Viridis", "Plasma", "Cividis", "Inferno"])
    dfgrp = plot_df.groupby(["Date", "Country"])[metric].sum().reset_index()
    hm = dfgrp.pivot(index="Country", columns="Date", values=metric).fillna(0)
    logscale = st.checkbox("Logarithmische Skala", True)
    vals = np.where(hm.values < 0, 0, hm.values)
    vals = np.log1p(vals) if logscale else vals
    fig2 = px.imshow(vals, x=hm.columns.strftime("%Y-%m-%d"), y=hm.index,
                     labels=dict(color="log(1 + Fälle)" if logscale else "Fälle"),
                     color_continuous_scale=cmap, aspect="auto")
    fig2.update_layout(xaxis_nticks=20, yaxis={'autorange': 'reversed'})
    st.plotly_chart(fig2, use_container_width=True)

# 🌍 Karte
elif selected_tab == "🌍 Karte":
    st.subheader("Weltkarte mit Animation")

    sel = metric  # für Karte wird dieselbe Metrik genutzt

    # Daten vorbereiten
    anim = plot_df.copy()
    anim["YearMonth"] = anim["Date"].dt.to_period("M").dt.to_timestamp()

    # Aggregation
    agg = anim.groupby(["Country", "YearMonth", "Continent", "Lat", "Long"], as_index=False)[sel]
    agg = agg.sum() if sel == "New Confirmed" else agg.max()
    agg = agg.dropna(subset=["Lat", "Long"])

    # Größe für Punktgröße auf Karte berechnen
    maxv = agg[sel].max() or 1
    agg["size_scaled"] = agg[sel].apply(lambda x: max(3, np.sqrt(max(0, x / maxv)) * 30))

    # Spalten für Tooltip explizit festlegen (keine Lat, Long, size_scaled)
    hover_data = {sel: True, "Country": True, "Continent": (view == "Continents")}

    # Weltkarte mit Animation erzeugen
    fig3 = px.scatter_map(
        agg, lat="Lat", lon="Long", size="size_scaled",
        color="Continent" if view == "Continents" else sel,
        hover_name="Country",
        hover_data={
            sel: True,
            "Country": True,
            "Continent": True,
            "Lat": False,
            "Long": False,
            "size_scaled": False
        },
        animation_frame=agg["YearMonth"].dt.strftime('%Y-%m'),
        size_max=30, zoom=1, height=700,
        color_continuous_scale="YlOrRd",
        title=f"COVID‑19 {sel} – monatlich ({'nach Kontinent' if view == 'Continents' else 'nach Land'})"
    )

    fig3.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig3, use_container_width=True, config={"scrollZoom": True})


# 🔥 Folium
elif selected_tab == "🔥 Folium":
    st.subheader("Statische Heatmap mit Folium")
    sel = metric
    anim = plot_df.copy()
    anim["YearMonth"] = anim["Date"].dt.to_period("M").dt.to_timestamp()
    agg = anim.groupby(["Country", "YearMonth", "Continent", "Lat", "Long"], as_index=False)[sel]
    agg = agg.sum() if sel == "New Confirmed" else agg.max()
    agg = agg.dropna(subset=["Lat", "Long"])
    unique_months = agg["YearMonth"].sort_values().unique()
    selected_month = st.select_slider(
        "Monat auswählen",
        options=unique_months,
        value=unique_months[-1],
        format_func=lambda d: d.strftime("%Y-%m")
    )
    hmap = agg[agg["YearMonth"] == selected_month][["Lat", "Long", sel]].dropna()
    if not hmap.empty:
        hmap[sel] = hmap[sel] / hmap[sel].max()
        m = folium.Map(location=[20, 0], zoom_start=1, tiles="OpenStreetMap")
        HeatMap(hmap.values.tolist(), radius=25, blur=15).add_to(m)
        st_folium(m, width=700, height=500)
    else:
        st.warning("Keine Daten für den ausgewählten Monat vorhanden.")

# 📥 CSV
elif selected_tab == "📥 CSV":
    st.subheader("CSV herunterladen")
    st.download_button("CSV herunterladen", plot_df.to_csv(index=False).encode("utf-8"),
                       file_name="covid_data.csv", mime="text/csv")

# 📸 Export
elif selected_tab == "📸 Export":
    st.subheader("Export Hinweis")
    st.info("📌 In den Plotly-Diagrammen das Download-Icon oder Rechtsklick zum Exportieren nutzen.")
