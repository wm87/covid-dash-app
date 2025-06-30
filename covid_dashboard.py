import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import polars as pl
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# Auto-Refresh alle 5 Minuten
st_autorefresh(interval=5 * 60 * 1000, key="datarefresh")

# CSV-URLs
BASE_URL = (
    "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
    "csse_covid_19_data/csse_covid_19_time_series/"
)
URLS = {
    "Confirmed": BASE_URL + "time_series_covid19_confirmed_global.csv",
    "Deaths": BASE_URL + "time_series_covid19_deaths_global.csv",
    "Recovered": BASE_URL + "time_series_covid19_recovered_global.csv",
}

country_name_corrections = {
    # USA Varianten
    "United States of America": "US",

    # UK Varianten
    "UK": "United Kingdom",
    "Great Britain": "United Kingdom",
    "England": "United Kingdom",

    # Andere häufige Abweichungen
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

# --- Natural Earth Daten laden ---
NE_SHAPEFILE_PATH = "ne_10m_admin_0_countries.shp"  # Pfad anpassen

# Lokale Shapefile laden
world = gpd.read_file(NE_SHAPEFILE_PATH)

print(world.columns)

country_col = None
for col in ["ADMIN"]:
    if col in world.columns:
        country_col = col
        break
if country_col is None:
    raise ValueError("Keine passende Spalte für Ländernamen gefunden")

# world ist dein GeoDataFrame mit Geometrien in EPSG:4326 (lat/lon)
world_proj = world.to_crs(epsg=3857)  # projiziert auf Web Mercator (Meter)
world_proj['centroid'] = world_proj.geometry.centroid  # jetzt genaue Centroid-Berechnung

# zurück projizieren auf lat/lon
world['centroid'] = world_proj['centroid'].to_crs(epsg=4326)

# Lat/Lon extrahieren
world['Longitude'] = world['centroid'].x
world['Latitude'] = world['centroid'].y

coords_df = world[[country_col, "Longitude", "Latitude"]].copy()
coords_df.rename(columns={country_col: "Country"}, inplace=True)


# Beispiel: CSVs laden und alle Länder
dfs = {k: pl.read_csv(url) for k, url in URLS.items()}
all_countries = dfs["Confirmed"]["Country/Region"].unique().to_list()
all_countries.sort()

#print(sorted(all_countries))


def get_coords(country_name):
    # Korrigierten Ländernamen suchen
    corrected_name = country_name_corrections.get(country_name, country_name)
    row = coords_df[coords_df['Country'] == corrected_name]
    if not row.empty:
        return row.iloc[0]['Latitude'], row.iloc[0]['Longitude']
    return np.nan, np.nan


country_coords = {c: get_coords(c) for c in all_countries}

# Dummy-Populationen
country_population = {c: 1_000_000 for c in all_countries}

# Set Streamlit Page Config
st.set_page_config(layout="wide")
st.title("🦠 COVID-19 Ultimate Dashboard")

date_columns = [col for col in dfs["Confirmed"].columns if col not in (
    "Province/State", "Country/Region", "Lat", "Long")]
date_columns.sort()

with st.sidebar:
    st.header("⚙️ Einstellungen")

    view = st.radio(
        "Anzeige nach:",
        ["Countries", "Continents"],
        index=0
    )

    # Automatisch Kontinente aus Natural Earth (über Länder extrahieren)
    continents = sorted(set(world['CONTINENT'].dropna().unique()))

    # Für Länder-Continent Mapping (Natural Earth 'CONTINENT' Spalte nutzen)
    # Mapping dict
    country_to_continent = {}
    for idx, row in world.iterrows():
        admin_name = row['ADMIN']
        cont_name = row['CONTINENT']
        # Korrekturen übernehmen (nur wenn admin_name in deinem Landelist auftaucht)
        if admin_name in all_countries:
            country_to_continent[admin_name] = cont_name

    if view == "Countries":
        desired_defaults = [
            "USA", "India", "Brazil", "Russia", "China", "Poland",
            "Ireland", "Turkey", "Japan", "Chile", "Canada", "Australia",
            "Saudi Arabia", "Namibia", "Mexico", "Sweden", "Greenland",
            "Germany", "United Kingdom", "France", "Italy", "Spain"
        ]
        available_defaults = [c for c in desired_defaults if c in all_countries]

        countries = st.multiselect(
            "Länder auswählen",
            sorted(all_countries),
            default=available_defaults
        )
    else:
        selected_continents = st.multiselect(
            "Kontinente auswählen",
            continents,
            default=["Europe", "Asia"]
        )
        countries = [
            c for c in all_countries if country_to_continent.get(c) in selected_continents
        ]

    metric = st.selectbox(
        "Metrik",
        ["Confirmed", "Deaths", "Recovered", "New Confirmed"]
    )

    per_100k = st.checkbox("pro 100.000 Einwohner")
    avg_choice = st.selectbox(
        "Moving Average",
        ["None", "7", "14", "30"],
        index=1
    )

if not countries:
    st.warning("Bitte mindestens ein Land oder Kontinent auswählen.")
    st.stop()

# --- Datenvorbereitung ---
long_dfs = []
for country in countries:
    df_confirmed = dfs["Confirmed"].filter(pl.col("Country/Region") == country)
    df_deaths = dfs["Deaths"].filter(pl.col("Country/Region") == country)
    df_recovered = dfs["Recovered"].filter(pl.col("Country/Region") == country)

    sum_confirmed = df_confirmed.drop(
        ["Province/State", "Lat", "Long", "Country/Region"]
    ).sum()
    dates = sum_confirmed.columns
    confirmed_cases = sum_confirmed.row(0)

    sum_deaths = df_deaths.drop(
        ["Province/State", "Lat", "Long", "Country/Region"]
    ).sum()
    deaths_cases = sum_deaths.row(0)

    sum_recovered = df_recovered.drop(
        ["Province/State", "Lat", "Long", "Country/Region"]
    ).sum()
    recovered_cases = sum_recovered.row(0)

    pl_df = pl.DataFrame({
        "Date": dates,
        "Confirmed": confirmed_cases,
        "Deaths": deaths_cases,
        "Recovered": recovered_cases,
    }).with_columns(
        pl.col("Date").str.to_date("%m/%d/%y")
    ).sort("Date")

    pl_df = pl_df.with_columns(
        pl.col("Confirmed").diff().fill_null(0).alias("New Confirmed")
    )

    display_metric = "New Confirmed" if metric == "New Confirmed" else metric

    if avg_choice != "None":
        pl_df = pl_df.with_columns(
            pl.col(display_metric).rolling_mean(window_size=int(avg_choice), min_samples=1)
            .alias(display_metric)
        )

    pl_df = pl_df.with_columns(
        (pl.col("Confirmed") - pl.col("Deaths") - pl.col("Recovered")).alias("Active")
    )

    if per_100k:
        pop = country_population.get(country)
        if pop:
            for colname in ["Confirmed", "Deaths", "Recovered", "Active", "New Confirmed"]:
                pl_df = pl_df.with_columns(
                    (pl.col(colname) / pop * 100000).alias(colname)
                )
        else:
            st.warning(f"⚠️ Keine Bevölkerungszahl für {country}")

    lat, lon = country_coords.get(country, (np.nan, np.nan))
    pl_df = pl_df.with_columns(
        pl.lit(country).alias("Country"),
        pl.lit(country_to_continent.get(country, "Other")).alias("Continent"),
        pl.lit(lat).alias("Lat"),
        pl.lit(lon).alias("Long"),
    )

    long_dfs.append(pl_df)

combined_df = pl.concat(long_dfs).sort(["Country", "Date"])
min_date = combined_df["Date"].min()
max_date = combined_df["Date"].max()

date_range = st.slider(
    "Zeitraum:",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="DD.MM.YYYY"
)

filtered_df = combined_df.filter(
    (pl.col("Date") >= date_range[0]) & (pl.col("Date") <= date_range[1])
)

plot_df = filtered_df.to_pandas()

# --- Tabs ---

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Verlauf", "🌡️ Heatmap", "🌍 Weltkarte", "📥 CSV", "📸 Export Hinweis"
])

# Verlauf
with tab1:
    st.subheader("Zeitverlauf")
    if view == "Countries":
        fig_line = px.line(
            plot_df,
            x="Date",
            y=metric if metric != "New Confirmed" else "New Confirmed",
            color="Country",
            title="Verlauf nach Land"
        )
    else:
        df_group = plot_df.groupby(["Date", "Continent"]).sum(numeric_only=True).reset_index()
        fig_line = px.line(
            df_group,
            x="Date",
            y=metric if metric != "New Confirmed" else "New Confirmed",
            color="Continent",
            title="Verlauf nach Kontinent"
        )
    st.plotly_chart(fig_line, use_container_width=True)

# Heatmap
with tab2:
    st.subheader("Heatmap der Fälle")

    color_scale = st.selectbox(
        "Farbschema für die Heatmap",
        ["Viridis", "Plasma", "Cividis", "Inferno"],
        index=0
    )

    df_group = plot_df.groupby(["Date", "Country"])[
        metric if metric != "New Confirmed" else "New Confirmed"].sum().reset_index()
    heatmap_df = df_group.pivot(index="Country", columns="Date",
                                values=metric if metric != "New Confirmed" else "New Confirmed").fillna(0)
    apply_log = st.checkbox("Logarithmische Farbskala", value=True, key="heatmap_log")
    if apply_log:
        heatmap_vals = np.log1p(heatmap_df.values)
        colorbar_title = "log(1 + Fälle)"
    else:
        heatmap_vals = heatmap_df.values
        colorbar_title = "Fälle"
    fig_heat = px.imshow(
        heatmap_vals,
        labels=dict(x="Datum", y="Land", color=colorbar_title),
        x=[d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d) for d in heatmap_df.columns],
        y=heatmap_df.index,
        color_continuous_scale=color_scale,
        aspect="auto",
        title="Heatmap der COVID-19 Fälle"
    )
    fig_heat.update_layout(
        xaxis_nticks=20,
        yaxis={'autorange': 'reversed'},
        margin=dict(l=100, r=20, t=50, b=100)
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# Weltkarte
with tab3:
    st.subheader("Interaktive Weltkarte mit Plotly Express")

    selected_metric = st.selectbox(
        "Metrik auswählen:",
        ["Confirmed", "Deaths", "Recovered", "New Confirmed"],
        index=0,
        key="metric_map"
    )

    anim_df = combined_df.to_pandas()
    anim_df["Date"] = pd.to_datetime(anim_df["Date"])

    anim_df = anim_df[
        (anim_df["Date"] >= pd.Timestamp(date_range[0])) &
        (anim_df["Date"] <= pd.Timestamp(date_range[1]))
        ]

    anim_df["YearMonth"] = anim_df["Date"].dt.to_period("M").dt.to_timestamp()

    # Bereit mit NE-Koordinaten, also keine zusätzliche Koordinatenquelle nötig
    # Aggregation je nach Metrik
    if selected_metric == "New Confirmed":
        aggregated_df = (
            anim_df
            .groupby(["Country", "YearMonth", "Continent", "Lat", "Long"], as_index=False)
            .agg({selected_metric: "sum"})
        )
    else:
        aggregated_df = (
            anim_df
            .groupby(["Country", "YearMonth", "Continent", "Lat", "Long"], as_index=False)
            .agg({selected_metric: "max"})
        )

    unique_cats = aggregated_df["Continent"].unique()
    palette = px.colors.qualitative.Safe
    if len(unique_cats) > len(palette):
        palette = (palette * ((len(unique_cats) // len(palette)) + 1))[:len(unique_cats)]
    color_map = dict(zip(unique_cats, palette))

    min_radius = 3
    max_radius = 30
    max_val = aggregated_df[selected_metric].max()
    if max_val == 0 or pd.isna(max_val):
        max_val = 1

    aggregated_df["size_scaled"] = aggregated_df[selected_metric].apply(
        lambda x: max(min_radius, np.sqrt(x / max_val) * max_radius)
    )

    hover_data = {
        selected_metric: ":,.0f",
        "YearMonth": True,
        "Lat": False,
        "Long": False,
        "size_scaled": False
    }

    fig_anim = px.scatter_geo(
        aggregated_df,
        lat="Lat",
        lon="Long",
        size="size_scaled",
        color="Continent",
        hover_name="Country",
        hover_data=hover_data,
        animation_frame=aggregated_df["YearMonth"].dt.strftime('%Y-%m'),
        projection="natural earth",
        title=f"Animierte COVID-19 {selected_metric} weltweit (aggregiert pro Monat und Land)",
        size_max=max_radius,
        template="plotly_white",
        color_discrete_map=color_map,
    )

    fig_anim.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        legend_title_text="Kontinent",
        geo=dict(
            showland=True,
            landcolor="rgb(240,240,240)",
            oceancolor="rgb(230,230,255)",
            showocean=True,
            lakecolor="rgb(200,200,255)",
            showcountries=True,
            countrycolor="gray",
            showframe=False,
            projection_type="natural earth",
            resolution=110
        ),
        dragmode="zoom"
    )

    # Animation Geschwindigkeit erhöhen
    if fig_anim.layout.updatemenus:
        fig_anim.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 700
        fig_anim.layout.updatemenus[0].buttons[1].args[1]['frame']['duration'] = 0

    st.plotly_chart(fig_anim, use_container_width=True)

# CSV Download
with tab4:
    st.subheader("CSV herunterladen")
    csv = plot_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "CSV herunterladen",
        csv,
        file_name="covid_data.csv",
        mime="text/csv"
    )

# Export Hinweis
with tab5:
    st.subheader("Exportieren")
    st.info(
        "📌 Bitte nutze die interaktiven Plotly-Charts oben, um per Rechtsklick oder über das 'Download'-Symbol Grafiken als PNG oder SVG zu speichern.\n\n"
        "Automatischer Bild-Export per Knopfdruck ist ohne Chrome/Chromium nicht möglich."
    )
