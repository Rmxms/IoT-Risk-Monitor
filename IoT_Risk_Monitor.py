import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json, random, io, warnings
from datetime import datetime, timedelta
from collections import Counter
from streamlit_autorefresh import st_autorefresh

#sklearn imports for all models and metrics
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, StackingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, roc_curve, auc, mean_absolute_error, mean_squared_error)

#imbalanced-learn for SMOTE oversampling to handle class imbalance
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

#gradient boosting libraries
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore')
np.random.seed(42)

#try importing tello drone library, mark as unavailable if not installed
try:
    from djitellopy import Tello
    TELLO_AVAILABLE = True
except ImportError:
    TELLO_AVAILABLE = False

#page config wide layout for dashboard feel
st.set_page_config(page_title="IoT Risk Monitor", page_icon="📡",
                   layout="wide", initial_sidebar_state="expanded")

#custom css for dark theme, fonts, status banners, mqtt box and condition chips
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.main {background-color: #080d18; }
.status-banner {
    padding: 16px 24px; border-radius: 14px; text-align: center;
    margin: 10px 0; font-family: 'Syne', sans-serif;
    font-weight: 800; font-size: 1.3rem; letter-spacing: 2px;
}
.safe-banner {background:#061309; border:2px solid #00e676; color:#00e676; }
.warn-banner {background:#130f00; border:2px solid #ffd600; color:#ffd600; }
.crit-banner {background:#130404; border:2px solid #ff1744; color:#ff1744; }
.mqtt-box {
    background: #050a12; border: 1px solid #162035; border-radius: 10px;
    padding: 14px; font-family: 'JetBrains Mono', monospace;
    font-size: 11.5px; height: 240px; overflow-y: auto;
}
.mqtt-pub {color: #42a5f5; }
.mqtt-recv {color: #66bb6a; }
.mqtt-upd {color: #ba68c8; }
.mqtt-ts {color: #455a64; }
.section-header {
    font-family: 'Syne', sans-serif; font-size: 0.8rem; font-weight: 700;
    color: #42a5f5; text-transform: uppercase; letter-spacing: 4px;
    border-bottom: 1px solid #1a2e4a; padding-bottom: 6px; margin: 22px 0 14px 0;
}
.live-dot {
    display: inline-block; width: 9px; height: 9px; border-radius: 50%;
    background: #ff1744; animation: blink 1s infinite; margin-right: 6px;
}
@keyframes blink { 0%,100%{opacity:1;} 50%{opacity:0.15;} }
.cond-chip {
    display: inline-block; padding: 4px 14px; border-radius: 999px;
    font-size: 0.75rem; font-weight: 700; margin: 3px; font-family: 'Syne', sans-serif;
}
.chip-c {background:#2a050533; color:#ff5252; border:1px solid #ff525255; }
.chip-w {background:#2a1e0033; color:#ffd740; border:1px solid #ffd74055; }
.chip-s {background:#002a1533; color:#69f0ae; border:1px solid #69f0ae55; }
</style>
""", unsafe_allow_html=True)

#label mappings and color constants used across all tabs
LABEL_NAMES = ['Safe', 'Warning', 'Critical']
LABEL_MAP = {'Safe': 0, 'Warning': 1, 'Critical': 2}
INV_MAP = {v: k for k, v in LABEL_MAP.items()}
STATUS_COLOR = {'Safe': '#00e676', 'Warning': '#ffd600', 'Critical': '#ff1744'}
STATUS_BG = {'Safe': '#061309',  'Warning': '#130f00',  'Critical': '#130404'}
STATUS_CSS = {'Safe': 'safe-banner','Warning': 'warn-banner','Critical': 'crit-banner'}
STATUS_ICON = {'Safe': '🟢', 'Warning': '🟡', 'Critical': '🔴'}
BASE_SENSORS = ['temperature','humidity','battery','smoke_gas','motion','light','pressure']
FUTURE_MET = ['temperature','humidity','battery','smoke_gas','light','pressure']

#initialise session state with defaults so the app doesnt crash on first load
defaults = dict(mode='Upload CSV', pipeline=None, sim_df=None, idx=10, running=False, mqtt_logs=[], event_log=[], tello=None, tello_connected=False, live_readings=[], thresh={'temp_warn':26.0,'temp_crit':27.5,'battery_warn':78.0, 'battery_crit':74.0,'gas_warn':360.0,'gas_crit':400.0, 'hum_high':62.0,'hum_low':45.0})
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

#autorefresh must be called before rendering anything — fires every 1.2s in live simulation mode
if st.session_state.get('running', False) and st.session_state.get('mode','') == 'Live Simulation':
    st_autorefresh(interval=1200, key="live_refresh")

#rule-based classifier — checks each sensor against physical danger thresholds
def classify_row(row, T=None):
    if T is None: T = st.session_state.thresh
    t,b,g,h = row['temperature'],row['battery'],row['smoke_gas'],row['humidity']
    if t>T['temp_crit'] or b<T['battery_crit'] or g>T['gas_crit']: return 'Critical'
    if t>T['temp_warn'] or b<T['battery_warn'] or g>T['gas_warn'] or h>T['hum_high'] or h<T['hum_low']: return 'Warning'
    return 'Safe'

#returns which specific sensors are triggering the current risk level
def get_conditions(row, T=None):
    if T is None: T = st.session_state.thresh
    t,b,g,h = row['temperature'],row['battery'],row['smoke_gas'],row['humidity']
    c = []
    if  t>T['temp_crit']: c.append(('High Temp','c'))
    elif t>T['temp_warn']:  c.append(('High Temp','w'))
    if  b<T['battery_crit']: c.append(('Low Battery','c'))
    elif b<T['battery_warn']: c.append(('Low Battery','w'))
    if  g>T['gas_crit']:  c.append(('High Smoke/Gas','c'))
    elif g>T['gas_warn']: c.append(('High Smoke/Gas','w'))
    if  h>T['hum_high']: c.append(('High Humidity','w'))
    elif h<T['hum_low']:  c.append(('Low Humidity','w'))
    return c if c else [('Normal','s')]

#builds the alert string shown in the status banner
def build_alert(row):
    conds = get_conditions(row)
    if conds[0][0] == 'Normal': return 'All systems nominal'
    icons = {'c':'🔴','w':'🟡','s':'🟢'}
    return '  │  '.join(f"{icons[s]} {n}" for n,s in conds)

#pipeline factory — wraps any model with SMOTE so oversampling happens inside each cv fold (no leakage)
def make_pipe(model, scale=False):
    steps = [('smote', SMOTE(random_state=42, k_neighbors=5))]
    if scale: steps.append(('scaler', StandardScaler()))
    steps.append(('model', model))
    return ImbPipeline(steps)

#adds gaussian jitter, random spikes and oscillating drift to simulate real sensor hardware noise
def add_noise(df):
    df_n, n = df.copy(), len(df)
    t = np.arange(n)
    for col in ['temperature','humidity','battery','light','pressure']:
        if col not in df_n.columns: continue
        r = df_n[col].max() - df_n[col].min()
        sp = np.zeros(n)
        idx2 = np.random.choice(n, size=int(n*.02), replace=False)
        sp[idx2] = np.random.choice([-1,1],size=len(idx2))*np.random.uniform(2,4,len(idx2))
        df_n[col] = df_n[col]+np.random.normal(0,.3,n)+sp+.00002*r*np.sin(2*np.pi*t/(n*.6))
    if 'sound' in df_n.columns:
        df_n['sound'] = df_n['sound']+np.random.normal(0,.2,n)
        df_n['smoke_gas'] = df_n['sound']*3.5
    return df_n

#v5 pipeline — trains 8 classifiers + stacking on uploaded csv
#cached so retraining only happens when a new file is uploaded
@st.cache_data(show_spinner=False)
def run_pipeline(file_bytes):
    df = pd.read_csv(io.BytesIO(file_bytes))
    df.columns = df.columns.str.lower().str.strip()
    df = df.rename(columns={'location':'zone'})
    df['smoke_gas'] = df['sound']*3.5
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = add_noise(df)

    #use physical thresholds for labelling (not percentile-based)
    T = {'temp_warn':26.0,'temp_crit':27.5,'battery_warn':78.0, 'battery_crit':74.0,'gas_warn':360.0,'gas_crit':400.0,'hum_high':62.0,'hum_low':45.0}
    df['status'] = df.apply(lambda r: classify_row(r,T), axis=1)

    #add 10% label noise to simulate real-world sensor measurement uncertainty
    nm = np.random.rand(len(df))<.10
    df.loc[nm,'status'] = np.random.choice(['Safe','Warning','Critical'],size=nm.sum())
    df['label'] = df['status'].map(LABEL_MAP)

    #create next-timestep targets for classification and regression
    df['next_label'] = df['label'].shift(-1)
    for col in FUTURE_MET: df[f'next_{col}'] = df[col].shift(-1)

    #lag features (t-1, t-2) give the model temporal context
    for col in BASE_SENSORS:
        df[f'{col}_lag1'] = df[col].shift(1); df[f'{col}_lag2'] = df[col].shift(2)
        df[f'{col}_roll_mean'] = df[col].rolling(5).mean()
        df[f'{col}_roll_std']  = df[col].rolling(5).std()

    #rate of change for the three most predictive sensors
    for col in ['temperature','smoke_gas','battery']: df[f'{col}_delta'] = df[col].diff()

    dm = df.dropna().copy(); dm['next_label'] = dm['next_label'].astype(int)

    FEAT = (BASE_SENSORS +[f'{c}_lag1' for c in BASE_SENSORS]+[f'{c}_lag2' for c in BASE_SENSORS] +[f'{c}_roll_mean' for c in BASE_SENSORS]+[f'{c}_roll_std' for c in BASE_SENSORS] +['temperature_delta','smoke_gas_delta','battery_delta'])

    #stratified split preserves class balance in both train and test sets
    Xtr,Xte,ytr,yte,Ytrm,Ytem,zt,_ = train_test_split(
        dm[FEAT],dm['next_label'],dm[[f'next_{c}' for c in FUTURE_MET]],dm['zone'],
        test_size=.20,random_state=42,stratify=dm['next_label'])

    SKF = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

    #tuned hyperparameters from v5 randomised search with stronger regularisation
    lp = make_pipe(LGBMClassifier(class_weight='balanced',random_state=42,verbose=-1, subsample=0.6,reg_lambda=1.0,reg_alpha=0.1,num_leaves=31,n_estimators=100, min_split_gain=0.1,min_child_samples=20,max_depth=-1,learning_rate=0.03,colsample_bytree=0.6))
    xp = make_pipe(XGBClassifier(eval_metric='mlogloss',random_state=42,verbosity=0, subsample=0.8,reg_lambda=1.0,reg_alpha=0.05,n_estimators=100,min_child_weight=3, max_depth=3,max_delta_step=1,learning_rate=0.05,gamma=0.05,colsample_bytree=0.6))
    rp = make_pipe(RandomForestClassifier(class_weight='balanced',random_state=42,n_jobs=-1, n_estimators=400,min_samples_split=15,min_samples_leaf=8,max_samples=0.8, max_features='sqrt',max_depth=None))

    #all 7 base classifiers
    pipes = {
        'Decision Tree': make_pipe(DecisionTreeClassifier(max_depth=8,min_samples_leaf=5,class_weight='balanced',random_state=42)),
        'Logistic Regression': make_pipe(LogisticRegression(max_iter=3000,C=0.5,class_weight='balanced',random_state=42),scale=True),
        'KNN':make_pipe(KNeighborsClassifier(n_neighbors=9,weights='distance'),scale=True),
        'SVM': make_pipe(SVC(kernel='rbf',C=2.0,gamma='scale',probability=True,class_weight='balanced',random_state=42),scale=True),
        'Random Forest':rp, 'XGBoost': xp, 'LightGBM': lp,
    }
    results = {}
    for name, pipe in pipes.items():
        pipe.fit(Xtr,ytr); pr=pipe.predict(Xte); pb=pipe.predict_proba(Xte)
        results[name] = dict(pipe=pipe,preds=pr,proba=pb, accuracy=accuracy_score(yte,pr), bal_acc=balanced_accuracy_score(yte,pr), f1_macro=f1_score(yte,pr,average='macro',zero_division=0))

    #stacking classifier — meta-learner combines base model probabilities
    #uses out-of-fold predictions so base models never see meta-training data
    stk = StackingClassifier(
        estimators=[('lgbm',lp.named_steps['model']),('xgb',xp.named_steps['model']),('rf',rp.named_steps['model'])], final_estimator=LogisticRegression(C=1.0,max_iter=1000,class_weight='balanced',random_state=42), cv=SKF,stack_method='predict_proba',n_jobs=-1)
    sp2 = ImbPipeline([('smote',SMOTE(random_state=42,k_neighbors=5)),('model',stk)])
    sp2.fit(Xtr,ytr); spr=sp2.predict(Xte); spb=sp2.predict_proba(Xte)
    results['Stacking'] = dict(pipe=sp2,preds=spr,proba=spb,
        accuracy=accuracy_score(yte,spr), bal_acc=balanced_accuracy_score(yte,spr), f1_macro=f1_score(yte,spr,average='macro',zero_division=0))

    #random forest regressor for predicting actual next sensor values
    rfr = RandomForestRegressor(n_estimators=200,max_depth=12,random_state=42,n_jobs=-1)
    rfr.fit(Xtr,Ytrm)

    return dict(df=df,df_model=dm,X_train=Xtr,X_test=Xte,y_train=ytr,y_test=yte, Y_test_m=Ytem,zone_test=zt,results=results,FEATURES=FEAT, rfr=rfr,rfr_preds=rfr.predict(Xte))

#synthetic simulation data — balanced risk distribution driven by temperature and smoke spikes
#battery stays healthy so it doesnt dominate and cause everything to be critical
@st.cache_data
def generate_sim(n=500):
    np.random.seed(42); random.seed(42)
    zones=['Zone A','Zone B','Zone C','Zone D']; base_ts=datetime(2025,1,1,8,0,0); rows=[]
    for i in range(n):
        zone=zones[i%4]
        temp = 24 + 3*np.sin(i/40) + np.random.normal(0,0.8)
        if random.random() < 0.08: temp += random.uniform(3, 8)
        hum = np.clip(52 + 12*np.sin(i/35+1) + np.random.normal(0,3), 35, 75)
        bat = np.clip(88 + 6*np.sin(i/60) + np.random.normal(0,1.5), 75, 100)
        gas = max(0, 280 + np.random.normal(0,40) + (random.uniform(100,250) if random.random()<0.07 else 0))
        rows.append({'timestamp':base_ts+timedelta(seconds=i*3),'device_id':'Drone_01','zone':zone,
                     'temperature':round(temp,1),'humidity':round(hum,1),'battery':round(bat,1),
                     'sound':round(gas/3.5,1),'smoke_gas':round(gas,1),
                     'motion':int(random.random()<0.22),
                     'light':round(max(0,380+np.random.normal(0,55)),1),
                     'pressure':round(1013+np.random.normal(0,3.5),1)})
    return pd.DataFrame(rows)

#sidebar — mode selection, dataset controls, tello controls and threshold sliders
with st.sidebar:
    st.markdown("## 📡 IoT Risk Monitor")
    st.markdown("---")
    mode = st.radio("**Mode**",['Upload CSV','Live Simulation','Tello Drone'])
    st.session_state.mode = mode
    st.markdown("---")

    if mode == 'Upload CSV':
        st.markdown("**Dataset**")
        uploaded = st.file_uploader("CSV file",type=['csv'],label_visibility="collapsed")
        if uploaded and st.button("▶  Train Models",use_container_width=True,type="primary"):
            with st.spinner("Training 8 models + stacking…"):
                st.session_state.pipeline = run_pipeline(uploaded.read())
                st.session_state.csv_idx  = 0
            st.success("✅ Training complete")
        if st.session_state.pipeline:
            P=st.session_state.pipeline
            best=max({k:v for k,v in P['results'].items()},key=lambda k:P['results'][k]['bal_acc'])
            st.caption(f"Best: **{best}** — {P['results'][best]['bal_acc']*100:.1f}% BalAcc")
            st.markdown("**Browse Readings**")
            total_rows = len(P['df'])
            if 'csv_idx' not in st.session_state: st.session_state.csv_idx = total_rows - 1
            #slider lets user scrub through all readings in the dataset
            st.session_state.csv_idx = st.slider(
                "Reading", 0, total_rows-1, st.session_state.csv_idx, 1,
                label_visibility="collapsed")
            nav1, nav2, nav3 = st.columns(3)
            with nav1:
                if st.button("⏮ First"): st.session_state.csv_idx = 0
            with nav2:
                if st.button("◀ Prev") and st.session_state.csv_idx > 0:
                    st.session_state.csv_idx -= 1
            with nav3:
                if st.button("Next ▶") and st.session_state.csv_idx < total_rows-1:
                    st.session_state.csv_idx += 1

    elif mode == 'Live Simulation':
        #version flag forces regeneration when simulation parameters change
        SIM_VERSION = "v3_balanced"
        if st.session_state.sim_df is None or st.session_state.get('sim_version') != SIM_VERSION:
            st.session_state.sim_df = generate_sim(500)
            st.session_state.idx = 10
            st.session_state.sim_version = SIM_VERSION
        c1,c2=st.columns(2)
        with c1:
            if st.button("▶ Start",use_container_width=True,type="primary"): st.session_state.running=True
        with c2:
            if st.button("⏹ Stop",use_container_width=True): st.session_state.running=False
        if st.button("⏭ Next Reading",use_container_width=True):
            if st.session_state.idx < len(st.session_state.sim_df)-1:
                st.session_state.idx+=1

    else:
        st.markdown("**Tello Drone**")
        drone_ip = st.text_input("Drone IP","192.168.10.1")
        c1,c2=st.columns(2)
        with c1:
            if st.button("🔗 Connect",use_container_width=True,type="primary"):
                if TELLO_AVAILABLE:
                    try:
                        t=Tello(host=drone_ip); t.connect()
                        st.session_state.tello=t; st.session_state.tello_connected=True
                        st.success("Connected!")
                    except Exception as e: st.error(f"Failed: {e}")
                else: st.warning("pip install djitellopy")
        with c2:
            if st.button("📥 Poll",use_container_width=True):
                if st.session_state.tello_connected:
                    t=st.session_state.tello
                    try:
                        import time as _t
                        state=None
                        #tello state stream takes a few seconds to start after connect
                        for _ in range(10):
                            try:
                                state=t.get_current_state()
                                if state: break
                            except: pass
                            _t.sleep(0.3)
                        if not state:
                            st.warning("State not ready. Wait 2s and retry.")
                        else:
                            ht=float(t.get_height())
                            #real sensors: battery, height, speed, flight time
                            #simulated: humidity, smoke/gas, temperature (tello has no gas or humidity sensor)
                            r={'timestamp':datetime.now(),
                               'battery':float(t.get_battery()),
                               'height':ht,
                               'flight_time':float(t.get_flight_time()),
                               'speed_x':float(t.get_speed_x()),
                               'speed_y':float(t.get_speed_y()),
                               'speed_z':float(t.get_speed_z()),
                               'humidity':float(50+np.random.normal(0,5)),
                               'smoke_gas':float(max(0,np.random.normal(300,60))),
                               'temperature':float(25+np.random.normal(0,1)),
                               'light':float(max(0,ht*3.5)),
                               'pressure':float(1013+np.random.normal(0,2)),
                               'motion':int(abs(t.get_speed_x())>20 or abs(t.get_speed_y())>20),
                               'zone':'Tello'}
                            st.session_state.live_readings.append(r)
                            st.session_state.live_readings=st.session_state.live_readings[-200:]
                            st.success(f"✅ Battery {r['battery']:.0f}%  Height {r['height']} cm")
                    except Exception as e:
                        st.error(f"Poll failed: {e}")

        if st.session_state.tello_connected:
            st.markdown("**✈️ Flight Controls**")
            #check battery before allowing takeoff
            try:
                batt_now = st.session_state.tello.get_battery()
                if batt_now < 20:
                    st.error(f"⛔ Battery too low ({batt_now}%) — charge before flying")
                else:
                    st.caption(f"🔋 Battery: {batt_now}%  ✅ Ready")
            except: pass

            fa,fb=st.columns(2)
            with fa:
                if st.button("🛫 Takeoff",use_container_width=True):
                    try:
                        import time as _tf
                        t2=st.session_state.tello
                        #send sdk command first to confirm communication before takeoff
                        t2.send_command_with_return("command")
                        _tf.sleep(0.5)
                        t2.takeoff()
                        st.success("🛫 Airborne!")
                    except Exception as e: st.error(f"Takeoff failed: {e}")
            with fb:
                if st.button("🛬 Land",use_container_width=True):
                    try: st.session_state.tello.land(); st.success("🛬 Landed.")
                    except Exception as e: st.error(str(e))
            st.markdown("**🕹️ Navigate**")
            dist=st.slider("Distance (cm)",20,150,50,10)
            na,nb,nc,nd=st.columns(4)
            with na:
                if st.button("⬆️",use_container_width=True):
                    try: st.session_state.tello.move_forward(dist)
                    except Exception as e: st.error(str(e))
            with nb:
                if st.button("⬇️",use_container_width=True):
                    try: st.session_state.tello.move_back(dist)
                    except Exception as e: st.error(str(e))
            with nc:
                if st.button("⬅️",use_container_width=True):
                    try: st.session_state.tello.move_left(dist)
                    except Exception as e: st.error(str(e))
            with nd:
                if st.button("➡️",use_container_width=True):
                    try: st.session_state.tello.move_right(dist)
                    except Exception as e: st.error(str(e))
            re1,re2=st.columns(2)
            with re1:
                if st.button("↺ Rotate L",use_container_width=True):
                    try: st.session_state.tello.rotate_counter_clockwise(45)
                    except Exception as e: st.error(str(e))
            with re2:
                if st.button("↻ Rotate R",use_container_width=True):
                    try: st.session_state.tello.rotate_clockwise(45)
                    except Exception as e: st.error(str(e))
            st.markdown("**📸 Camera**")
            p1,p2=st.columns(2)
            with p1:
                if st.button("📷 Snapshot",use_container_width=True):
                    try:
                        import time as _t2, cv2
                        st.session_state.tello.streamon(); _t2.sleep(1)
                        frame=st.session_state.tello.get_frame_read().frame
                        fname=f"tello_{datetime.now().strftime('%H%M%S')}.jpg"
                        cv2.imwrite(fname,frame)
                        st.session_state.tello.streamoff()
                        st.success(f"Saved: {fname}")
                    except Exception as e: st.error(f"Camera: {e}  (pip install opencv-python)")
            with p2:
                stream_on=st.session_state.get('tello_stream',False)
                if st.button("🎥 Stream ON" if not stream_on else "🎥 Stream OFF",use_container_width=True):
                    st.session_state['tello_stream']=not stream_on
                    try:
                        if not stream_on: st.session_state.tello.streamon()
                        else: st.session_state.tello.streamoff()
                    except Exception as e: st.error(str(e))

        if not TELLO_AVAILABLE:
            st.info("📦 pip install djitellopy")
        st.caption("Real: battery, height, speed, flight time.  Simulated: temp, humidity, smoke/gas.")

    st.markdown("---")
    st.markdown("**⚙️ Alert Thresholds**")
    st.markdown("""<div style='font-size:0.72rem;color:#546e7a;line-height:1.7;margin-bottom:8px;'>
    🌡️ <b>Temperature</b> — higher = more dangerous<br>
    🔋 <b>Battery</b> — lower = more dangerous<br>
    💨 <b>Smoke/Gas</b> — higher = more dangerous
    </div>""", unsafe_allow_html=True)

    DEFAULTS = {'temp_warn':26.0,'temp_crit':27.5,'battery_warn':78.0,
                'battery_crit':74.0,'gas_warn':360.0,'gas_crit':400.0,
                'hum_high':62.0,'hum_low':45.0}

    if st.button("↺ Reset to Defaults", use_container_width=True):
        st.session_state.thresh = DEFAULTS.copy()
        st.rerun()

    #sliders update thresholds live — all classifications re-run instantly on change
    T = st.session_state.thresh
    T['temp_warn']= st.slider("🌡️ Temp Warn °C  (above = Warning)",    22.0,35.0,T['temp_warn'],  0.5)
    T['temp_crit'] = st.slider("🌡️ Temp Crit °C  (above = Critical)",   24.0,40.0,T['temp_crit'],  0.5)
    T['battery_warn'] = st.slider("🔋 Batt Warn %   (below = Warning)",    50.0,90.0,T['battery_warn'],1.0)
    T['battery_crit'] = st.slider("🔋 Batt Crit %   (below = Critical)",   40.0,80.0,T['battery_crit'],1.0)
    T['gas_warn'] = st.slider("💨 Gas Warn      (above = Warning)",   200.0,500.0,T['gas_warn'], 10.0)
    st.session_state.thresh = T
    st.markdown("---")
    st.markdown("**📡 MQTT Broker**")
    st.code("host: localhost\nport: 1883\ntopic: iot/env/#",language="yaml")

#resolve which data source is active based on current mode
T = st.session_state.thresh

if mode == 'Upload CSV':
    if not st.session_state.pipeline:
        st.markdown("""
        <div style='text-align:center;padding:80px 20px;'>
            <div style='font-size:3.5rem;'>📡</div>
            <div style='font-size:1.8rem;font-weight:800;letter-spacing:-0.02em;margin:10px 0 6px;'>IoT Risk Monitor</div>
            <div style='font-size:0.95rem;color:#455a64;margin-bottom:24px;'>Upload a sensor CSV and click <b>Train Models</b> to begin</div>
            <div style='display:flex;gap:28px;justify-content:center;'>
                <div style='text-align:center;'><div style='font-size:2rem;'>🔬</div><div style='font-size:0.78rem;color:#455a64;margin-top:4px;'>8 ML Models</div></div>
                <div style='text-align:center;'><div style='font-size:2rem;'>📊</div><div style='font-size:0.78rem;color:#455a64;margin-top:4px;'>Live Analysis</div></div>
                <div style='text-align:center;'><div style='font-size:2rem;'>🎯</div><div style='font-size:0.78rem;color:#455a64;margin-top:4px;'>Risk Predictor</div></div>
                <div style='text-align:center;'><div style='font-size:2rem;'>🚁</div><div style='font-size:0.78rem;color:#455a64;margin-top:4px;'>Tello Drone</div></div>
            </div>
        </div>""", unsafe_allow_html=True)
        st.stop()
    P = st.session_state.pipeline
    df_live = P['df'].copy()
    #re-apply thresholds in case user changed the sliders
    df_live['status'] = df_live.apply(lambda r: classify_row(r,T), axis=1)
    if 'csv_idx' not in st.session_state: st.session_state.csv_idx = len(df_live)-1
    idx = st.session_state.csv_idx
    cur = df_live.iloc[idx]

elif mode == 'Live Simulation':
    sim_df = st.session_state.sim_df
    sim_df = sim_df.copy()
    sim_df['status'] = sim_df.apply(lambda r: classify_row(r,T), axis=1)
    #advance the index on each autorefresh cycle
    if st.session_state.running:
        if st.session_state.idx < len(sim_df)-1: st.session_state.idx += 1
        else: st.session_state.running = False
    idx = st.session_state.idx
    cur = sim_df.iloc[idx]; df_live = sim_df
    topic = f"iot/env/{cur['zone'].lower().replace(' ','')}"
    ts_str = cur['timestamp'].strftime('%H:%M:%S')
    #only append a new log entry when the timestamp changes to avoid duplicates
    if not st.session_state.mqtt_logs or st.session_state.mqtt_logs[-1].get('ts') != ts_str:
        st.session_state.mqtt_logs += [
            {'ts':ts_str,'msg':f'[PUBLISH]  → {topic}','cls':'mqtt-pub'},
            {'ts':ts_str,'msg':'[BROKER]   message queued','cls':'mqtt-recv'},
            {'ts':ts_str,'msg':'[GATEWAY]  payload received','cls':'mqtt-recv'},
            {'ts':ts_str,'msg':'[PROCESS]  thresholds evaluated','cls':'mqtt-upd'},
            {'ts':ts_str,'msg':f'[STATUS]   → {cur["status"]}','cls':'mqtt-upd'},
            {'ts':ts_str,'msg':'[DASHBOARD] view refreshed','cls':'mqtt-pub'},
        ]
        st.session_state.mqtt_logs = st.session_state.mqtt_logs[-40:]
    ev = {'Time':ts_str,'Zone':cur['zone'],'Temp°C':cur['temperature'],
          'Hum%':cur['humidity'],'Batt%':cur['battery'],'Smoke':int(cur['smoke_gas']),
          'Motion':'✅' if cur['motion'] else '—',
          'Status':f"{STATUS_ICON[cur['status']]} {cur['status']}"}
    if not st.session_state.event_log or st.session_state.event_log[-1]['Time']!=ts_str:
        st.session_state.event_log.append(ev)
        st.session_state.event_log = st.session_state.event_log[-80:]
else:
    if not st.session_state.live_readings:
        st.info("Connect your Tello drone and click **Poll** to start receiving data.")
        st.stop()
    df_live = pd.DataFrame(st.session_state.live_readings)
    df_live['status'] = df_live.apply(lambda r: classify_row(r,T), axis=1)
    cur = df_live.iloc[-1]; idx = len(df_live)-1

status = cur['status']
alert  = build_alert(cur)

#header — shows mode, current reading index and timestamp
h1,h2 = st.columns([4,1])
with h1:
    if st.session_state.running: st.markdown('<span class="live-dot"></span>', unsafe_allow_html=True)
    st.title("📡 IoT Risk Monitor")
    st.caption(f"{'🔴 Live Simulation' if mode=='Live Simulation' else '📂 Analysis' if mode=='Upload CSV' else '🚁 Tello Live'}  │  "
               f"Reading **{idx+1}** / {len(df_live)}  │  "
               f"{cur['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if hasattr(cur['timestamp'],'strftime') else cur['timestamp']}")
with h2:
    if st.session_state.running: st.success("🔴 LIVE")
    else: st.info(f"⏸ {mode}")
st.markdown("---")

TABS = st.tabs(["📡 Live Monitor","📊 Analysis","🤖 Models","🎯 Predictor","🗺️ Zones"])

#tab 1 — live sensor readings, status banner, trend charts, event log and mqtt feed
with TABS[0]:
    prev = df_live.iloc[max(0,idx-1)]
    c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
    c1.metric("🌡️ Temp",  f"{cur['temperature']:.1f}°C", f"{cur['temperature']-prev['temperature']:+.1f}")
    c2.metric("💧 Humidity",  f"{cur['humidity']:.1f}%", f"{cur['humidity']-prev['humidity']:+.1f}")
    c3.metric("🔋 Battery",  f"{cur['battery']:.1f}%",   f"{cur['battery']-prev['battery']:+.1f}")
    c4.metric("💨 Smoke/Gas",  f"{cur['smoke_gas']:.0f}",  f"{cur['smoke_gas']-prev['smoke_gas']:+.0f}")
    c5.metric("🚶 Motion",  "Yes" if cur['motion'] else "No")
    c6.metric("💡 Light",  f"{cur['light']:.0f} lx")
    c7.metric("📍 Zone",  cur.get('zone','—'))

    st.markdown(
        f'<div class="status-banner {STATUS_CSS[status]}">'
        f'{STATUS_ICON[status]}  STATUS: {status.upper()}<br>'
        f'<span style="font-size:.8rem;font-weight:400;opacity:.8">{alert}</span></div>',
        unsafe_allow_html=True)

    st.markdown('<div class="section-header">📈 Sensor Trends</div>', unsafe_allow_html=True)
    window = df_live.iloc[max(0,idx-100):idx+1]
    #colour each marker by its risk status
    mk_col = window['status'].map(STATUS_COLOR)

    def trend_fig(col, clr, label):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=window['timestamp'],y=window[col],mode='lines+markers',
            line=dict(color=clr,width=2),marker=dict(color=list(mk_col),size=5),
            hovertemplate=f'%{{x|%H:%M:%S}}<br>{label}: %{{y}}<extra></extra>'))
        fig.update_layout(template='plotly_dark',height=230,
            margin=dict(l=40,r=10,t=20,b=30),
            paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(8,13,24,1)')
        return fig

    t1,t2,t3,t4 = st.tabs(["🌡️ Temp","💧 Humidity","🔋 Battery","💨 Smoke"])
    with t1: st.plotly_chart(trend_fig('temperature','#ef5350','Temp °C'),use_container_width=True)
    with t2: st.plotly_chart(trend_fig('humidity','#26c6da','Humidity %'),use_container_width=True)
    with t3: st.plotly_chart(trend_fig('battery','#ffa726','Battery %'),use_container_width=True)
    with t4: st.plotly_chart(trend_fig('smoke_gas','#ba68c8','Smoke/Gas'),use_container_width=True)

    #event log and mqtt feed only shown in live modes — not relevant for static csv analysis
    if mode != 'Upload CSV':
        st.markdown('<div class="section-header">📋 Event Log  &amp;  MQTT Feed</div>', unsafe_allow_html=True)
        ev_col, mq_col = st.columns([3,2])
        with ev_col:
            if st.session_state.event_log:
                st.dataframe(pd.DataFrame(st.session_state.event_log[::-1]),
                             use_container_width=True, height=260)
            else:
                st.info("No events yet — press ▶ Start or ⏭ Next Reading")
        with mq_col:
            if st.session_state.mqtt_logs:
                html = '<div class="mqtt-box">'
                for item in reversed(st.session_state.mqtt_logs[-24:]):
                    html += (f'<div><span class="mqtt-ts">{item["ts"]}  </span>'
                             f'<span class="{item["cls"]}">{item["msg"]}</span></div>')
                html += '</div>'
                st.markdown(html, unsafe_allow_html=True)
            else:
                payload = {"zone":cur.get('zone','—'),"temperature":cur['temperature'], "humidity":cur['humidity'],"battery":cur['battery'], "smoke_gas":round(float(cur['smoke_gas']),1), "motion":int(cur['motion']),"status":status, "timestamp":str(cur['timestamp'])}
                st.code(json.dumps(payload,indent=2), language="json")

    if mode=='Tello Drone' and st.session_state.tello_connected:
        st.markdown('<div class="section-header">🚁 Tello Live Telemetry</div>', unsafe_allow_html=True)
        try:
            t=st.session_state.tello
            ta,tb,tc,td,te = st.columns(5)
            ta.metric("🔋 Battery",f"{t.get_battery()}%")
            tb.metric("📏 Height",  f"{t.get_height()} cm")
            tc.metric("⏱ Flight Time", f"{t.get_flight_time()} s")
            td.metric("💨 Speed X", f"{t.get_speed_x()} cm/s")
            te.metric("💨 Speed Y",  f"{t.get_speed_y()} cm/s")
        except: st.warning("Could not read telemetry.")

        if st.session_state.get('tello_stream', False):
            st.markdown('<div class="section-header">📹 Live Camera Feed</div>', unsafe_allow_html=True)
            try:
                import cv2
                frame_placeholder = st.empty()
                frame = t.get_frame_read().frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, caption="Tello Camera", use_column_width=True)
            except Exception as e:
                st.warning(f"Stream unavailable: {e}")

        if st.session_state.live_readings:
            st.markdown('<div class="section-header">📋 Flight Log</div>', unsafe_allow_html=True)
            log_cols = ['timestamp','battery','height','flight_time','speed_x','speed_y','motion','status']
            log_df = pd.DataFrame(st.session_state.live_readings)
            show_cols = [c for c in log_cols if c in log_df.columns]
            st.dataframe(log_df[show_cols].iloc[::-1], use_container_width=True, height=240)

#tab 2 — distribution charts, sensor violin plots, correlations and spike detection
with TABS[1]:
    st.markdown('<div class="section-header">🧮 Status Distribution</div>', unsafe_allow_html=True)
    counts = df_live['status'].value_counts()
    ca,cb = st.columns([1,2])
    with ca:
        fig_pie = go.Figure(go.Pie(
            labels=list(counts.index), values=list(counts.values),
            marker_colors=[STATUS_COLOR.get(s,'#888') for s in counts.index],
            hole=.45, textinfo='label+percent'))
        fig_pie.update_layout(template='plotly_dark',height=280,showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig_pie, use_container_width=True)
    with cb:
        fig_bar = go.Figure()
        for s,c in STATUS_COLOR.items():
            fig_bar.add_trace(go.Bar(name=s,x=[s],y=[int(counts.get(s,0))],
                marker_color=c,text=[int(counts.get(s,0))],textposition='auto',width=0.4))
        fig_bar.update_layout(template='plotly_dark',height=280,showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(8,13,24,1)',
            margin=dict(l=20,r=10,t=20,b=20))
        st.plotly_chart(fig_bar, use_container_width=True)

    #use df_model when in csv mode for cleaner engineered-feature data
    src = st.session_state.pipeline['df_model'] if (mode=='Upload CSV' and st.session_state.pipeline) else df_live
    st.markdown('<div class="section-header">📊 Sensor Distributions per Class</div>', unsafe_allow_html=True)
    dist_sel = st.selectbox("Sensor",['temperature','humidity','battery','smoke_gas','light','pressure'])

    #helper to convert hex to rgba since plotly doesnt accept 8-digit hex
    def hex_to_rgba(hex_color, alpha=0.15):
        h = hex_color.lstrip('#')
        r,g,b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
        return f'rgba({r},{g},{b},{alpha})'

    fig_vio = go.Figure()
    for s,c in STATUS_COLOR.items():
        vals = src[src['status']==s][dist_sel].dropna()
        if len(vals):
            fig_vio.add_trace(go.Violin(y=vals,name=s,line_color=c,
                fillcolor=hex_to_rgba(c,0.15),box_visible=True,meanline_visible=True))
    fig_vio.update_layout(template='plotly_dark',height=320,
        paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(8,13,24,1)',
        margin=dict(l=40,r=10,t=20,b=20),violinmode='group')
    st.plotly_chart(fig_vio, use_container_width=True)

    st.markdown('<div class="section-header">🔗 Correlations</div>', unsafe_allow_html=True)
    avail = [c for c in ['temperature','humidity','battery','smoke_gas','light','pressure','motion'] if c in df_live.columns]
    corr = df_live[avail].corr()
    fig_corr = go.Figure(go.Heatmap(z=corr.values,x=corr.columns,y=corr.columns,
        colorscale='RdBu',zmid=0,text=np.round(corr.values,2),
        texttemplate='%{text}',textfont_size=9))
    fig_corr.update_layout(template='plotly_dark',height=380,
        paper_bgcolor='rgba(0,0,0,0)',margin=dict(l=60,r=10,t=20,b=60))
    st.plotly_chart(fig_corr, use_container_width=True)

    #spike detection — flags readings more than 2.5 standard deviations from the mean
    st.markdown('<div class="section-header">⚡ Spike Detection</div>', unsafe_allow_html=True)
    sp_sel = st.selectbox("Sensor",['temperature','humidity','battery','smoke_gas'],key='sp_sel')
    vals = df_live[sp_sel].values; mu,sigma = vals.mean(),vals.std()
    up=np.where(vals>mu+2.5*sigma)[0]; dn=np.where(vals<mu-2.5*sigma)[0]
    fig_sp = go.Figure()
    fig_sp.add_trace(go.Scatter(y=vals,mode='lines',line=dict(color='#42a5f5',width=1.2),name='Reading'))
    if len(up): fig_sp.add_trace(go.Scatter(x=list(up),y=vals[up],mode='markers',
        marker=dict(color='#ff1744',size=7),name=f'High ({len(up)})'))
    if len(dn): fig_sp.add_trace(go.Scatter(x=list(dn),y=vals[dn],mode='markers',
        marker=dict(color='#ffd600',size=7),name=f'Low ({len(dn)})'))
    fig_sp.add_hline(y=mu+2.5*sigma,line_dash='dash',line_color='rgba(255,23,68,0.55)',annotation_text='+2.5σ')
    fig_sp.add_hline(y=mu-2.5*sigma,line_dash='dash',line_color='rgba(255,214,0,0.55)',annotation_text='-2.5σ')
    fig_sp.update_layout(template='plotly_dark',height=280,
        paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(8,13,24,1)',
        margin=dict(l=40,r=10,t=20,b=20))
    st.plotly_chart(fig_sp, use_container_width=True)

#tab 3 — model performance, confusion matrices, roc curves and feature importance
#persists even if user switches to simulation mode after training
with TABS[2]:
    active_pipeline = st.session_state.pipeline
    if not active_pipeline:
        st.info("Upload a CSV and train models to see full ML results here.")
    else:
        P = active_pipeline
        results = P['results']; y_test = P['y_test']

        st.markdown('<div class="section-header">📊 Performance Summary</div>', unsafe_allow_html=True)
        rows = [{'Model':k,'Accuracy':f"{v['accuracy']*100:.1f}%",
                 'Balanced Acc':f"{v['bal_acc']*100:.1f}%",'Macro F1':f"{v['f1_macro']*100:.1f}%"}
                for k,v in results.items()]
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

        col_ba,col_f1 = st.columns(2)
        with col_ba:
            df_m=pd.DataFrame([{'M':k,'V':v['bal_acc']*100} for k,v in results.items()]).sort_values('V')
            n=len(df_m)
            #darker red = higher score (better models at top)
            red_palette=[f'hsl(0,{int(60+40*i/(n-1))}%,{int(65-30*i/(n-1))}%)' for i in range(n)]
            fig=go.Figure(go.Bar(x=df_m['V'],y=df_m['M'],orientation='h',
                marker_color=red_palette,
                text=[f"{v:.1f}%" for v in df_m['V']],textposition='auto'))
            fig.update_layout(template='plotly_dark',height=320,xaxis_range=[0,105],
                title='Balanced Accuracy',paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(8,13,24,1)',margin=dict(l=140,r=10,t=40,b=20))
            st.plotly_chart(fig,use_container_width=True)
        with col_f1:
            df_m2=pd.DataFrame([{'M':k,'V':v['f1_macro']*100} for k,v in results.items()]).sort_values('V')
            n2=len(df_m2)
            #darker blue = higher f1 score
            blue_palette=[f'hsl(210,{int(60+30*i/(n2-1))}%,{int(70-30*i/(n2-1))}%)' for i in range(n2)]
            fig2=go.Figure(go.Bar(x=df_m2['V'],y=df_m2['M'],orientation='h',
                marker_color=blue_palette,
                text=[f"{v:.1f}%" for v in df_m2['V']],textposition='auto'))
            fig2.update_layout(template='plotly_dark',height=320,xaxis_range=[0,105],
                title='Macro F1',paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(8,13,24,1)',margin=dict(l=140,r=10,t=40,b=20))
            st.plotly_chart(fig2,use_container_width=True)

        #show confusion matrices for top 4 models ranked by balanced accuracy
        st.markdown('<div class="section-header">🔲 Confusion Matrices — Top 4</div>', unsafe_allow_html=True)
        top4=sorted(results.items(),key=lambda x:x[1]['bal_acc'],reverse=True)[:4]
        cm_cols=st.columns(4)
        for col,(name,res) in zip(cm_cols,top4):
            with col:
                cm=confusion_matrix(y_test,res['preds'],labels=[0,1,2])
                fig_cm=go.Figure(go.Heatmap(z=cm,x=LABEL_NAMES,y=LABEL_NAMES,
                    colorscale='Blues',text=cm,texttemplate='%{text}',
                    textfont_size=13,showscale=False))
                fig_cm.update_layout(template='plotly_dark',height=260,
                    title=dict(text=f"{name} {res['bal_acc']*100:.0f}%",font_size=10),
                    xaxis_title='Predicted',yaxis_title='Actual',
                    paper_bgcolor='rgba(0,0,0,0)',margin=dict(l=50,r=10,t=50,b=50))
                st.plotly_chart(fig_cm,use_container_width=True)

        #roc curves plotted one-vs-rest for each class
        st.markdown('<div class="section-header">📉 ROC Curves — Top 4</div>', unsafe_allow_html=True)
        y_bin=label_binarize(y_test,classes=[0,1,2])
        roc_cols=st.columns(4)
        for col,(name,res) in zip(roc_cols,top4):
            with col:
                fig_r=go.Figure()
                fig_r.add_shape(type='line',line=dict(dash='dash',color='#455a64'),x0=0,x1=1,y0=0,y1=1)
                all_fpr=np.linspace(0,1,200); tprs=[]; aucs=[]
                for i,cls in enumerate(LABEL_NAMES):
                    fpr,tpr,_=roc_curve(y_bin[:,i],res['proba'][:,i])
                    ra=auc(fpr,tpr)
                    fig_r.add_trace(go.Scatter(x=fpr,y=tpr,mode='lines',
                        line=dict(color=list(STATUS_COLOR.values())[i],width=1.8),
                        name=f'{cls} {ra:.2f}'))
                    tprs.append(np.interp(all_fpr,fpr,tpr)); aucs.append(ra)
                fig_r.add_trace(go.Scatter(x=all_fpr,y=np.mean(tprs,axis=0),mode='lines',
                    line=dict(color='white',width=1.5,dash='dot'),name=f'Avg {np.mean(aucs):.2f}'))
                fig_r.update_layout(template='plotly_dark',height=260,
                    title=dict(text=name,font_size=10),
                    xaxis_title='FPR',yaxis_title='TPR',
                    legend=dict(font_size=7,bgcolor='rgba(0,0,0,0)'),
                    paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(8,13,24,1)',
                    margin=dict(l=40,r=10,t=40,b=40))
                st.plotly_chart(fig_r,use_container_width=True)

        #feature importance from xgboost — smoke_gas and battery dominate because they
        #have the strongest separation between safe and critical readings
        st.markdown('<div class="section-header">🔍 Feature Importance — XGBoost</div>', unsafe_allow_html=True)
        if 'XGBoost' in results:
            ms=results['XGBoost']['pipe'].named_steps['model']
            imp=pd.Series(ms.feature_importances_,index=P['FEATURES']).sort_values(ascending=False).head(15)
            fig_imp=go.Figure(go.Bar(x=imp.values[::-1],y=imp.index[::-1],orientation='h',
                marker=dict(color=imp.values[::-1],colorscale='Plasma'),
                text=[f"{v:.3f}" for v in imp.values[::-1]],textposition='auto'))
            fig_imp.update_layout(template='plotly_dark',height=380,
                paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(8,13,24,1)',
                margin=dict(l=180,r=20,t=20,b=20))
            st.plotly_chart(fig_imp,use_container_width=True)

#tab 4 — rule-based predictor with live sliders
#ml model not used here since it was trained for next-step time-series prediction not single-point queries
with TABS[3]:
    st.markdown('<div style="font-size:.85rem;color:#546e7a;margin-bottom:14px;">Set sensor values — get the <b>current risk status</b> and which sensor is triggering it.</div>', unsafe_allow_html=True)
    col_sl,col_res = st.columns([2,1])
    ranges = {'temperature':(15.0,40.0,24.0,0.1,"°C"),'humidity':(20.0,90.0,54.0,1.0,"%"),
              'battery':(10.0,100.0,85.0,1.0,"%"),'smoke_gas':(50.0,700.0,300.0,5.0,""),
              'light':(200.0,600.0,380.0,5.0,"lx"),'pressure':(990.0,1050.0,1013.0,0.5,"hPa")}
    manual = {}
    with col_sl:
        st.markdown('<div class="section-header">⚙️ Sensor Inputs</div>', unsafe_allow_html=True)
        for sensor,(lo,hi,default,step,unit) in ranges.items():
            label=f"{sensor.replace('_',' ').title()} {'('+unit+')' if unit else ''}"
            manual[sensor] = st.slider(label,lo,hi,default,step)
        manual['motion'] = float(st.radio("🚶 Motion",[0,1],horizontal=True))

    with col_res:
        st.markdown('<div class="section-header">🎯 Result</div>', unsafe_allow_html=True)
        rule_status = classify_row(pd.Series(manual),T)
        sc = STATUS_COLOR[rule_status]

        st.markdown(f"""
        <div style="background:{STATUS_BG[rule_status]};border:2px solid {sc};
                    border-radius:16px;padding:24px;text-align:center;margin-bottom:14px;">
            <div style="color:#546e7a;font-size:0.68rem;text-transform:uppercase;
                        letter-spacing:.12em;margin-bottom:6px;">STATUS</div>
            <div style="color:{sc};font-size:2.6rem;font-weight:800;line-height:1.1;">
                {STATUS_ICON[rule_status]} {rule_status}</div>
        </div>""", unsafe_allow_html=True)

        #show which specific sensors are causing the current risk level
        conds=get_conditions(pd.Series(manual),T)
        chips="".join(f"<span class='cond-chip chip-{s}'>{n}</span>" for n,s in conds)
        st.markdown(f"<div style='margin-top:14px;'>{chips}</div>", unsafe_allow_html=True)

        #gauge coloured by current risk status zone
        fig_g=go.Figure(go.Indicator(mode='gauge+number',value=manual['temperature'],
            number={'suffix':'°C','valueformat':'.1f'},
            gauge={'axis':{'range':[0,45],'tickcolor':'#546e7a'},
                   'steps':[{'range':[0,T['temp_warn']],'color':'#061309'},
                             {'range':[T['temp_warn'],T['temp_crit']],'color':'#130f00'},
                             {'range':[T['temp_crit'],45],'color':'#130404'}],
                   'bar':{'color':STATUS_COLOR[rule_status]}},
            title={'text':'Temperature','font':{'color':'#90caf9','size':12}}))
        fig_g.update_layout(template='plotly_dark',height=200,
            margin=dict(l=20,r=20,t=50,b=10),paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_g, use_container_width=True)

#tab 5 — zone timelines, summary table and class transition matrix
with TABS[4]:
    src2 = st.session_state.pipeline['df_model'] if (mode=='Upload CSV' and st.session_state.pipeline) else df_live
    zones = sorted(src2['zone'].unique()) if 'zone' in src2.columns else []
    if not zones:
        st.info("No zone data available.")
    else:
        st.markdown('<div class="section-header">🗺️ Risk Timeline per Zone</div>', unsafe_allow_html=True)
        zone_sel=st.multiselect("Zones",zones,default=zones[:4] if len(zones)>4 else zones)
        win=max(10,len(src2)//(50*max(len(zone_sel),1)))
        for zone in zone_sel:
            zdf=src2[src2['zone']==zone].sort_index().reset_index(drop=True)
            fig_z=go.Figure()
            for s,c in STATUS_COLOR.items():
                rolled=(zdf['status']==s).astype(float).rolling(win,center=True).mean()
                #convert hex to rgb for rgba fill since plotly rejects 8-digit hex
                r2,g2,b2 = int(c[1:3],16),int(c[3:5],16),int(c[5:7],16)
                fig_z.add_trace(go.Scatter(y=rolled,mode='lines',name=s,
                    line=dict(color=c,width=2),fill='tozeroy',
                    fillcolor=f'rgba({r2},{g2},{b2},0.18)'))
            fig_z.update_layout(template='plotly_dark',height=200,yaxis=dict(range=[0,1]),
                title=dict(text=f"Zone: {zone}  ({len(zdf)} readings)",font_size=11),
                paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(8,13,24,1)',
                margin=dict(l=50,r=10,t=40,b=20),
                legend=dict(orientation='h',y=1.2,font_size=9))
            st.plotly_chart(fig_z,use_container_width=True)

        st.markdown('<div class="section-header">📊 Zone Summary</div>', unsafe_allow_html=True)
        summary=[]
        for zone in zones:
            zdf=src2[src2['zone']==zone]
            dom=zdf['status'].value_counts().idxmax() if len(zdf) else '—'
            summary.append({'Zone':zone,'Readings':len(zdf),
                'Safe %':f"{(zdf['status']=='Safe').mean()*100:.1f}%",
                'Warning %':f"{(zdf['status']=='Warning').mean()*100:.1f}%",
                'Critical %':f"{(zdf['status']=='Critical').mean()*100:.1f}%",
                'Dominant':f"{STATUS_ICON.get(dom,'?')} {dom}"})
        st.dataframe(pd.DataFrame(summary).set_index('Zone'),use_container_width=True)

        #transition matrix shows how often risk status changes between readings
        #high diagonal values explain why the persistence baseline is hard to beat
        st.markdown('<div class="section-header">🔄 Class Transition Matrix</div>', unsafe_allow_html=True)
        ls=src2['label'].values if 'label' in src2.columns else src2['status'].map(LABEL_MAP).values
        trans=np.zeros((3,3),dtype=int)
        for i in range(len(ls)-1): trans[ls[i],ls[i+1]]+=1
        tp=trans.astype(float); rs=tp.sum(axis=1,keepdims=True)
        tp=np.divide(tp,rs,where=rs>0)
        tc1,tc2=st.columns(2)
        with tc1:
            fig_tp=go.Figure(go.Heatmap(z=tp,x=LABEL_NAMES,y=LABEL_NAMES,colorscale='YlOrRd',
                text=np.round(tp,2),texttemplate='%{text:.2f}',textfont_size=12,showscale=True,zmin=0,zmax=1))
            fig_tp.update_layout(template='plotly_dark',height=300,title='Transition Probabilities',
                xaxis_title='Next',yaxis_title='Current',
                paper_bgcolor='rgba(0,0,0,0)',margin=dict(l=60,r=10,t=50,b=50))
            st.plotly_chart(fig_tp,use_container_width=True)
        with tc2:
            fig_tc=go.Figure(go.Heatmap(z=trans,x=LABEL_NAMES,y=LABEL_NAMES,colorscale='Blues',
                text=trans,texttemplate='%{text}',textfont_size=12,showscale=True))
            fig_tc.update_layout(template='plotly_dark',height=300,title='Transition Counts',
                xaxis_title='Next',yaxis_title='Current',
                paper_bgcolor='rgba(0,0,0,0)',margin=dict(l=60,r=10,t=50,b=50))
            st.plotly_chart(fig_tc,use_container_width=True)
        st.caption(f"Avg self-transition: **{np.diag(tp).mean()*100:.1f}%** — explains why baseline persistence scores ~{np.diag(tp).mean()*100:.0f}%")
