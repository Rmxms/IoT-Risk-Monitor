# 📡 IoT Risk Monitor

A real-time IoT environmental monitoring dashboard built with Streamlit. Supports three data sources — uploaded sensor CSV, live simulation, and a physical Tello drone. Classifies sensor readings into Safe, Warning, or Critical risk levels using a full ML pipeline of 8 models plus a stacking classifier.

---

## Features

- **Upload CSV** — trains 8 ML classifiers (Decision Tree, Logistic Regression, KNN, SVM, Random Forest, XGBoost, LightGBM, Stacking) on your sensor data and presents full model evaluation including confusion matrices, ROC curves, balanced accuracy, macro F1 and feature importance
- **Live Simulation** — auto-advancing synthetic sensor feed with MQTT message flow visualisation, event log and real-time trend charts
- **Tello Drone** — connects to a DJI Tello via WiFi, polls live telemetry (battery, height, speed, flight time), and supports takeoff, land, directional navigation, rotation and camera snapshot
- **Predictor tab** — set sensor values with sliders and get the current risk status instantly with condition chips showing exactly which sensor is triggering it
- **Zone analysis** — rolling risk timelines per zone, zone summary table and class transition matrix
- **Adjustable thresholds** — all classification boundaries are configurable live from the sidebar with a reset-to-defaults button

---

## Installation

```bash
# clone the repo
git clone https://github.com/your-username/iot-risk-monitor.git
cd iot-risk-monitor

# create and activate virtual environment
python3 -m venv iot_env
source iot_env/bin/activate        # mac / linux
iot_env\Scripts\activate           # windows

# install dependencies
pip install -r requirements.txt
```

---

## Running the Dashboard

```bash
streamlit run IoT_Risk_Monitor.py
```

Opens at `http://localhost:8501`

---

## Usage

### Upload CSV Mode
1. Select **Upload CSV** in the sidebar
2. Upload your sensor CSV file (must contain: `timestamp`, `temperature`, `humidity`, `battery`, `sound`, `motion`, `light`, `pressure`, `zone`)
3. Click **▶ Train Models** — training takes 1–3 minutes
4. Browse all 5 tabs — Live Monitor, Analysis, Models, Predictor, Zones
5. Use the sidebar slider to scrub through individual readings

### Live Simulation Mode
1. Select **Live Simulation**
2. Click **▶ Start** — the feed advances automatically every 1.2 seconds
3. Click **⏹ Stop** to pause, **⏭ Next Reading** to step manually
4. Adjust threshold sliders to see how classifications change in real time

### Tello Drone Mode
1. Power on the Tello drone and wait for the LED to blink yellow
2. Connect your computer WiFi to the Tello network (`TELLO-XXXXXX`)
3. Select **Tello Drone** in the sidebar
4. Click **🔗 Connect**, wait 3–4 seconds, then click **📥 Poll**
5. Use flight controls for takeoff, landing, navigation and camera

> **Note:** Real sensors from Tello are battery, height, speed and flight time. Temperature, humidity and smoke/gas are simulated since the Tello EDU does not have those sensors.

---

## CSV Format

Your CSV should have these columns:

| Column | Description |
|---|---|
| `timestamp` | datetime string |
| `device_id` | sensor device ID |
| `zone` | location label (e.g. Room A, Outside) |
| `temperature` | °C |
| `humidity` | % |
| `battery` | % |
| `sound` | raw sound level (used to derive smoke_gas) |
| `motion` | 0 or 1 |
| `light` | lux |
| `pressure` | hPa |

---

## ML Pipeline

The v5 pipeline trains the following models using `SMOTE` inside each CV fold to prevent data leakage:

| Model | Type |
|---|---|
| Decision Tree | Base classifier |
| Logistic Regression | Base classifier |
| KNN | Base classifier |
| SVM (RBF) | Base classifier |
| Random Forest | Base classifier (tuned) |
| XGBoost | Base classifier (tuned) |
| LightGBM | Base classifier (tuned) |
| Stacking (LGB + XGB + RF → LR) | Meta-learner ensemble |

Features include raw sensor values, lag features (t-1, t-2), rolling mean and std (window=5), and rate-of-change deltas for temperature, smoke/gas and battery. The target is the **next timestep's risk label**.

---

## Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Dashboard framework |
| `streamlit-autorefresh` | Non-blocking auto-refresh for live simulation |
| `plotly` | Interactive charts |
| `scikit-learn` | ML models and metrics |
| `imbalanced-learn` | SMOTE oversampling |
| `xgboost` | XGBoost classifier |
| `lightgbm` | LightGBM classifier |
| `djitellopy` | Tello drone SDK |
| `opencv-python` | Drone camera feed |

---

## Project Structure

```
iot-risk-monitor/
├── MQTT.py              # main dashboard application
├── requirements.txt     # python dependencies
├── README.md            # this file
└── real_time_data.csv   # sample sensor dataset (optional)
```

---

## Alert Thresholds (Defaults)

| Sensor | Warning | Critical |
|---|---|---|
| Temperature | > 26°C | > 27.5°C |
| Battery | < 78% | < 74% |
| Smoke/Gas | > 360 | > 400 |
| Humidity | > 62% or < 45% | — |

All thresholds are adjustable from the sidebar.
