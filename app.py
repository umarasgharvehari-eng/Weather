import os
from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from openai import OpenAI


st.set_page_config(
    page_title="AI Weather Pro App",
    page_icon="🌦️",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
.main-title {
    background: linear-gradient(135deg, #0f172a, #1e3a8a, #0284c7);
    padding: 24px;
    border-radius: 20px;
    color: white;
    margin-bottom: 20px;
}
.main-title h1 {
    margin: 0;
    font-size: 2.2rem;
}
.main-title p {
    margin-top: 8px;
    font-size: 1rem;
    opacity: 0.95;
}
.metric-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 16px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.05);
    text-align: center;
}
.metric-label {
    color: #64748b;
    font-size: 14px;
    margin-bottom: 6px;
}
.metric-value {
    color: #0f172a;
    font-size: 22px;
    font-weight: 700;
}
.info-box {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-title">
    <h1>🌦️ AI Weather Pro App</h1>
    <p>Get accurate live weather, AI insights, 7-day forecast, monthly trends, and yearly analytics.</p>
</div>
""", unsafe_allow_html=True)

# =========================
# API CLIENT
# =========================
api_key = None
try:
    api_key = st.secrets["Weather"]
except Exception:
    api_key = os.getenv("Weather")

client = None
if api_key:
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )

# =========================
# HELPERS
# =========================
def get_coordinates(city_name: str):
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {
        "name": city_name,
        "count": 1,
        "language": "en",
        "format": "json"
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    if "results" in data and data["results"]:
        place = data["results"][0]
        return {
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "name": place.get("name", city_name),
            "country": place.get("country", ""),
            "admin1": place.get("admin1", ""),
            "timezone": place.get("timezone", "auto"),
        }
    return None


def get_current_forecast(lat: float, lon: float, timezone: str):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": timezone,
        "forecast_days": 7,
        "current": ",".join([
            "temperature_2m",
            "apparent_temperature",
            "relative_humidity_2m",
            "precipitation",
            "weather_code",
            "cloud_cover",
            "wind_speed_10m",
            "wind_gusts_10m",
            "wind_direction_10m",
            "pressure_msl"
        ]),
        "daily": ",".join([
            "weather_code",
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "wind_speed_10m_max"
        ])
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def get_historical_daily(lat: float, lon: float, start_date: str, end_date: str, timezone: str):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": timezone,
        "daily": ",".join([
            "weather_code",
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "precipitation_sum",
            "wind_speed_10m_max"
        ])
    }
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def weather_code_to_text(code):
    mapping = {
        0: "Clear Sky",
        1: "Mainly Clear",
        2: "Partly Cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Rime Fog",
        51: "Light Drizzle",
        53: "Moderate Drizzle",
        55: "Dense Drizzle",
        56: "Light Freezing Drizzle",
        57: "Dense Freezing Drizzle",
        61: "Slight Rain",
        63: "Moderate Rain",
        65: "Heavy Rain",
        66: "Light Freezing Rain",
        67: "Heavy Freezing Rain",
        71: "Slight Snow",
        73: "Moderate Snow",
        75: "Heavy Snow",
        77: "Snow Grains",
        80: "Slight Rain Showers",
        81: "Moderate Rain Showers",
        82: "Violent Rain Showers",
        85: "Slight Snow Showers",
        86: "Heavy Snow Showers",
        95: "Thunderstorm",
        96: "Thunderstorm with Slight Hail",
        99: "Thunderstorm with Heavy Hail",
    }
    return mapping.get(code, f"Code {code}")


def get_weather_emoji(code):
    if code in [0, 1]:
        return "☀️"
    if code in [2]:
        return "⛅"
    if code in [3]:
        return "☁️"
    if code in [45, 48]:
        return "🌫️"
    if code in [51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82]:
        return "🌧️"
    if code in [71, 73, 75, 77, 85, 86]:
        return "❄️"
    if code in [95, 96, 99]:
        return "⛈️"
    return "🌦️"


def get_ai_summary(place, forecast_data):
    if not client:
        return "Groq API key not found. Add Streamlit secret named 'Weather'."

    current = forecast_data.get("current", {})
    daily = forecast_data.get("daily", {})

    prompt = f"""
You are a professional weather assistant.

Give a helpful and simple weather explanation for an everyday user.

Location: {place['name']}, {place['country']}
Region: {place.get('admin1', '')}
Current temperature: {current.get('temperature_2m', 'N/A')} °C
Feels like: {current.get('apparent_temperature', 'N/A')} °C
Humidity: {current.get('relative_humidity_2m', 'N/A')}%
Weather code: {current.get('weather_code', 'N/A')}
Wind speed: {current.get('wind_speed_10m', 'N/A')} km/h
Wind gusts: {current.get('wind_gusts_10m', 'N/A')} km/h
Cloud cover: {current.get('cloud_cover', 'N/A')}%
Pressure: {current.get('pressure_msl', 'N/A')} hPa
Today's max: {daily.get('temperature_2m_max', ['N/A'])[0] if daily.get('temperature_2m_max') else 'N/A'} °C
Today's min: {daily.get('temperature_2m_min', ['N/A'])[0] if daily.get('temperature_2m_min') else 'N/A'} °C
Today's precipitation: {daily.get('precipitation_sum', ['N/A'])[0] if daily.get('precipitation_sum') else 'N/A'} mm

Return:
1. Short weather summary
2. Clothing advice
3. Outdoor advice
4. Rain/wind risk
5. One practical tip
"""

    try:
        response = client.responses.create(
            model="llama-3.3-70b-versatile",
            input=prompt
        )
        return response.output_text
    except Exception as e:
        return f"AI explanation unavailable:\n{str(e)}"


def make_recent_daily_chart(forecast_data: dict):
    daily = forecast_data.get("daily", {})
    if not daily or "time" not in daily:
        return None

    df = pd.DataFrame({
        "Date": daily.get("time", []),
        "Max Temperature": daily.get("temperature_2m_max", []),
        "Min Temperature": daily.get("temperature_2m_min", []),
    })

    fig = px.line(
        df,
        x="Date",
        y=["Max Temperature", "Min Temperature"],
        markers=True,
        title="7-Day Temperature Forecast"
    )
    fig.update_layout(template="plotly_white", height=420, title_x=0.5)
    return fig


def make_monthly_chart(hist_df: pd.DataFrame):
    monthly = hist_df.copy()
    monthly["time"] = pd.to_datetime(monthly["time"])
    monthly["month"] = monthly["time"].dt.strftime("%Y-%m")

    grouped = monthly.groupby("month", as_index=False).agg({
        "temperature_2m_mean": "mean",
        "precipitation_sum": "sum",
    })

    fig = px.line(
        grouped,
        x="month",
        y="temperature_2m_mean",
        markers=True,
        title="Monthly Average Temperature (Last 12 Months)"
    )
    fig.update_layout(template="plotly_white", height=420, title_x=0.5)
    return fig


def make_yearly_chart(hist_df: pd.DataFrame):
    yearly = hist_df.copy()
    yearly["time"] = pd.to_datetime(yearly["time"])
    yearly["year"] = yearly["time"].dt.year

    grouped = yearly.groupby("year", as_index=False).agg({
        "temperature_2m_mean": "mean",
        "precipitation_sum": "sum",
    })

    fig = px.bar(
        grouped,
        x="year",
        y="precipitation_sum",
        title="Yearly Total Precipitation (Last 5 Years)"
    )
    fig.update_layout(template="plotly_white", height=420, title_x=0.5)
    return fig


def metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================
# UI
# =========================
city = st.text_input("Enter City Name", placeholder="e.g. Vehari, Lahore, Karachi")

if st.button("Get Weather", use_container_width=True):
    if not city.strip():
        st.warning("Please enter a city name.")
    else:
        try:
            place = get_coordinates(city)
            if not place:
                st.error("City not found. Please try another city.")
            else:
                forecast_data = get_current_forecast(
                    place["latitude"],
                    place["longitude"],
                    place["timezone"]
                )

                end_dt = date.today()
                start_12m = end_dt - timedelta(days=365)
                start_5y = end_dt - timedelta(days=365 * 5)

                hist_12m = get_historical_daily(
                    place["latitude"],
                    place["longitude"],
                    start_12m.isoformat(),
                    end_dt.isoformat(),
                    place["timezone"]
                )

                hist_5y = get_historical_daily(
                    place["latitude"],
                    place["longitude"],
                    start_5y.isoformat(),
                    end_dt.isoformat(),
                    place["timezone"]
                )

                df_12m = pd.DataFrame(hist_12m.get("daily", {}))
                df_5y = pd.DataFrame(hist_5y.get("daily", {}))

                current = forecast_data.get("current", {})
                daily = forecast_data.get("daily", {})
                code = current.get("weather_code", "N/A")
                condition = weather_code_to_text(code) if code != "N/A" else "N/A"
                emoji = get_weather_emoji(code) if code != "N/A" else "🌦️"

                st.markdown(
                    f"""
                    <div class="info-box">
                        <h2>{emoji} {place['name']}, {place['country']}</h2>
                        <p>{place.get('admin1', '')} • {place.get('timezone', 'N/A')}</p>
                        <p><b>Condition:</b> {condition}</p>
                        <p><b>Updated:</b> {current.get('time', 'N/A')}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    metric_card("Temperature", f"{current.get('temperature_2m', 'N/A')} °C")
                with c2:
                    metric_card("Feels Like", f"{current.get('apparent_temperature', 'N/A')} °C")
                with c3:
                    metric_card("Humidity", f"{current.get('relative_humidity_2m', 'N/A')}%")
                with c4:
                    metric_card("Wind Speed", f"{current.get('wind_speed_10m', 'N/A')} km/h")

                c5, c6, c7, c8 = st.columns(4)
                with c5:
                    metric_card("Wind Gusts", f"{current.get('wind_gusts_10m', 'N/A')} km/h")
                with c6:
                    metric_card("Pressure", f"{current.get('pressure_msl', 'N/A')} hPa")
                with c7:
                    metric_card("Cloud Cover", f"{current.get('cloud_cover', 'N/A')}%")
                with c8:
                    metric_card("Current Rain", f"{current.get('precipitation', 'N/A')} mm")

                c9, c10, c11, c12 = st.columns(4)
                with c9:
                    metric_card("Today's Max", f"{daily.get('temperature_2m_max', ['N/A'])[0] if daily.get('temperature_2m_max') else 'N/A'} °C")
                with c10:
                    metric_card("Today's Min", f"{daily.get('temperature_2m_min', ['N/A'])[0] if daily.get('temperature_2m_min') else 'N/A'} °C")
                with c11:
                    metric_card("Today's Rain", f"{daily.get('precipitation_sum', ['N/A'])[0] if daily.get('precipitation_sum') else 'N/A'} mm")
                with c12:
                    metric_card("Wind Direction", f"{current.get('wind_direction_10m', 'N/A')}°")

                tab1, tab2, tab3, tab4 = st.tabs([
                    "AI Explanation",
                    "7-Day Forecast",
                    "Monthly Graph",
                    "Yearly Graph"
                ])

                with tab1:
                    st.text_area("AI Weather Insight", get_ai_summary(place, forecast_data), height=260)

                with tab2:
                    fig = make_recent_daily_chart(forecast_data)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

                with tab3:
                    if not df_12m.empty and "time" in df_12m.columns:
                        st.plotly_chart(make_monthly_chart(df_12m), use_container_width=True)

                with tab4:
                    if not df_5y.empty and "time" in df_5y.columns:
                        st.plotly_chart(make_yearly_chart(df_5y), use_container_width=True)

        except requests.exceptions.RequestException as e:
            st.error(f"Weather service error: {e}")
        except Exception as e:
            st.error(f"Something went wrong: {e}")
