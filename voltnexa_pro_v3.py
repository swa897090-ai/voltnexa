"""
VOLTNEXA - FIND EV CHARGING STATIONS ANYWHERE (version 3)
New in this version:
  * REGISTER page: first name, last name, phone number, email, OTP sent to the phone number, strong password + confirm password
  * LOGIN page: phone number OR email + password
  * TIME CHECK: the time box shows the live current time; a time that has already passed today (for example a morning
    time when it is already afternoon) shows an error and the search is blocked (checked in the browser AND on the server)
  * MOBILE: fully responsive, phone bottom-menu bar, big touch buttons
  * INSTALLABLE WEB APP (PWA): on a phone, Chrome menu > "Add to Home screen" / "Install app" opens Voltnexa like an app
OPENING THE SITE: a visitor who is not logged in lands on the REGISTER page; if this browser already has an account it lands on the LOGIN page.\nFiles needed in the SAME folder: ev_charging_sessions.csv and stations.csv  (users are saved in voltnexa_users.db, created automatically)
Run on your computer:   pip install flask pandas numpy scikit-learn matplotlib xgboost pillow
                        python voltnexa_pro_v3.py        then open  http://127.0.0.1:5000  in Chrome
Open on your phone:     phone and computer on the SAME Wi-Fi, then open the "Phone" address printed in the terminal.
OTP SMS:                set TWILIO_SID, TWILIO_TOKEN and TWILIO_FROM to send real SMS. Without them the OTP is printed in the
                        terminal and shown on the register page (demo mode, VOLTNEXA_DEMO_OTP=0 turns the on-screen OTP off).
"""
import base64, contextlib, hashlib, hmac, io, json, math, os, re, secrets, socket, sqlite3, sys, time, urllib.parse, urllib.request
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, Response, abort, jsonify, redirect, render_template, request, send_from_directory, session
from werkzeug.security import check_password_hash, generate_password_hash
from jinja2 import DictLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
try:
    from xgboost import XGBRegressor
except ImportError:
    sys.exit("XGBoost is not installed. Run:  pip install xgboost")

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()   # works in Colab / Jupyter too
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"]})

# ============================================================ SAMPLE STATIONS AND PLACES (approximate coordinates)
STATIONS = pd.DataFrame([
    ("ST01", "Chikkabanavara EV Hub", 13.0786, 77.5071, "home", 4, 22, "AC", 13),
    ("ST02", "Abbigere Main Road Charger", 13.0668, 77.5126, "shop", 6, 30, "DC", 17),
    ("ST03", "Bagalkunte Cross Charging Point", 13.0610, 77.5190, "shop", 6, 30, "DC", 18),
    ("ST04", "T Dasarahalli Metro Charger", 13.0480, 77.5160, "shop", 6, 30, "DC", 17),
    ("ST05", "Peenya Industrial Area Hub", 13.0285, 77.5200, "office", 10, 50, "DC", 16),
    ("ST06", "Jalahalli Cross Fast Charger", 13.0488, 77.5385, "hwy", 8, 60, "DC", 19),
    ("ST07", "Nagasandra Metro Parking", 13.0480, 77.5000, "home", 4, 22, "AC", 13),
    ("ST08", "Mathikere Charging Point", 13.0330, 77.5640, "home", 4, 22, "AC", 14),
    ("ST09", "Vidyaranyapura Mall Charger", 13.0780, 77.5580, "mall", 8, 50, "DC", 18),
    ("ST10", "Yeshwanthpur Metro Charger", 13.0285, 77.5409, "shop", 6, 30, "DC", 17),
    ("ST11", "Nelamangala NH48 Fast Charger", 13.0957, 77.3938, "hwy", 8, 60, "DC", 20),
    ("ST12", "Dobbaspet Highway Charger", 13.2240, 77.3260, "hwy", 6, 60, "DC", 21),
    ("ST13", "Tumakuru Highway Hub", 13.3392, 77.1140, "hwy", 10, 120, "DC", 22),
    ("ST14", "Doddaballapura Town Charger", 13.2957, 77.5376, "shop", 6, 30, "DC", 18),
    ("ST15", "Devanahalli Airport Road Hub", 13.2473, 77.7120, "hwy", 12, 120, "DC", 22),
    ("ST16", "Hoskote Junction Charger", 13.0707, 77.7982, "hwy", 8, 60, "DC", 20),
    ("ST17", "Electronic City Phase 1 Hub", 12.8452, 77.6602, "office", 8, 60, "DC", 17),
    ("ST18", "Kengeri Mysuru Road Charger", 12.9171, 77.4838, "hwy", 8, 60, "DC", 20),
    ("ST19", "Ramanagara Highway Hub", 12.7157, 77.2810, "hwy", 8, 120, "DC", 22),
    ("ST20", "Yelahanka New Town Hub", 13.1007, 77.5963, "mall", 6, 50, "DC", 18),
], columns=["Station_ID", "Name", "Lat", "Lng", "Type", "Chargers", "Power_kW", "Connector", "Price"])

STATIONS_PATH = os.path.join(HERE, "stations.csv")
CSV_PATH = os.path.join(HERE, "ev_charging_sessions.csv")
if os.path.exists(STATIONS_PATH):
    STATIONS = pd.read_csv(STATIONS_PATH)       # columns: Station_ID, Name, Lat, Lng, Type, Chargers, Power_kW, Connector, Price

TYPES = ["office", "mall", "hwy", "shop", "home"]
FEATURES = ["hour", "day_of_week", "month", "is_weekend", "station_code", "type_code", "chargers", "power_kw",
            "lag_1h", "lag_2h", "lag_24h", "rolling_3h", "rolling_24h"]
TARGET = "Demand_kWh"


def peak(h, c, w):
    d = min(abs(h - c), 24 - abs(h - c))
    return math.exp(-d * d / (2 * w * w))


def profile(t, h, we):          # typical daily pattern per station type (only used to create the sample data)
    if t == "office": return 0.12 if we else .25 + .6 * peak(h, 10, 1.6) + .5 * peak(h, 18, 1.6)
    if t == "mall": return .15 + (.7 if we else .45) * peak(h, 18, 2.4) + .25 * peak(h, 13, 2)
    if t == "hwy": return .2 + .45 * peak(h, 9, 2) + .5 * peak(h, 18, 2.4) + (.15 if we else 0)
    if t == "shop": return .15 + .5 * peak(h, 12, 2.5) + .5 * peak(h, 19, 2)
    return .12 + .7 * peak(h, 21, 2.6) + .2 * peak(h, 7, 1.5) + (.1 if we else 0)


def km(a, b, c, d):
    r = math.pi / 180
    return 6371 * math.hypot((c - a) * r, (d - b) * r * math.cos(a * r))


# ============================================================ MACHINE LEARNING PIPELINE (steps 0-7)
def split(df, frac=0.2):        # chronological split: earlier hours = training, latest hours = testing
    cut = np.sort(df.Hour.unique())[int(df.Hour.nunique() * (1 - frac))]
    return df[df.Hour < cut], df[df.Hour >= cut]


def build_pipeline():
    print("STEP 0  Loading the dataset:", CSV_PATH)
    if not os.path.exists(CSV_PATH):
        sys.exit("Dataset not found. Put ev_charging_sessions.csv (and stations.csv) in the folder: " + HERE)
    df = pd.read_csv(CSV_PATH, parse_dates=["Connection_Time", "Disconnection_Time"])
    df = df[df["Station_ID"].isin(STATIONS["Station_ID"])]
    print(f"        {len(df)} charging sessions, {df['Station_ID'].nunique()} stations")

    print("STEP 1  Data preprocessing")
    df["Energy_kWh"] = df["Energy_kWh"].fillna(df.groupby("Station_ID")["Energy_kWh"].transform("median"))
    df = df.drop_duplicates()
    df["Duration_min"] = (df["Disconnection_Time"] - df["Connection_Time"]).dt.total_seconds() / 60
    df = df[(df["Energy_kWh"] > 0) & (df["Duration_min"] > 0)]
    df["Hour"] = df["Connection_Time"].dt.floor("h")
    full = pd.MultiIndex.from_product([STATIONS.Station_ID, pd.date_range(df.Hour.min(), df.Hour.max(), freq="h")],
                                      names=["Station_ID", "Hour"])
    hourly = df.groupby(["Station_ID", "Hour"])["Energy_kWh"].sum().reindex(full, fill_value=0).rename(TARGET).reset_index()

    print("STEP 2  Feature engineering")
    f = hourly.sort_values(["Station_ID", "Hour"]).merge(STATIONS[["Station_ID", "Type", "Chargers", "Power_kW"]], on="Station_ID")
    f["hour"], f["day_of_week"], f["month"] = f.Hour.dt.hour, f.Hour.dt.dayofweek, f.Hour.dt.month
    f["is_weekend"] = (f.day_of_week >= 5).astype(int)
    f["station_code"] = f.Station_ID.astype("category").cat.codes
    f["type_code"] = f.Type.map({t: i for i, t in enumerate(TYPES)})
    f = f.rename(columns={"Chargers": "chargers", "Power_kW": "power_kw"})
    g = f.groupby("Station_ID")[TARGET]
    f["lag_1h"], f["lag_2h"], f["lag_24h"] = g.shift(1), g.shift(2), g.shift(24)
    f["rolling_3h"] = g.transform(lambda s: s.shift(1).rolling(3).mean())
    f["rolling_24h"] = g.transform(lambda s: s.shift(1).rolling(24).mean())
    f = f.dropna(subset=["lag_24h", "rolling_24h"]).reset_index(drop=True)
    train, test = split(f)

    print("STEP 3  Feature selection (XGBoost importance)")
    m0 = XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42).fit(train[FEATURES], train[TARGET])
    imp0 = pd.Series(m0.feature_importances_, index=FEATURES).sort_values(ascending=False)
    sel = [x for x in imp0.index if imp0[x] >= 0.02] or list(imp0.index[:5])
    print("        selected:", sel)

    print("STEP 4  XGBoost model training")
    model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8,
                         objective="reg:squarederror", random_state=42).fit(train[sel], train[TARGET])

    print("STEP 5  Model evaluation")
    p = np.clip(model.predict(test[sel]), 0, None)
    out = test[["Station_ID", "Hour"]].assign(Actual=test[TARGET].values, Predicted=p)
    metrics = {"MAE": round(mean_absolute_error(test[TARGET], p), 2),
               "RMSE": round(math.sqrt(mean_squared_error(test[TARGET], p)), 2),
               "R2": round(r2_score(test[TARGET], p), 3)}
    print("       ", metrics)

    print("STEP 6  Output figures (saved in the figures folder)")
    make_figures(f, out, model, sel)

    print("STEP 7  Predicting demand for every station, day type and hour")
    avg = f.groupby(["Station_ID", "is_weekend", "hour"])[TARGET].mean()
    ref = f.groupby("Station_ID")[TARGET].quantile(.95)               # a station's "very busy" level
    codes = {s: i for i, s in enumerate(sorted(f.Station_ID.unique()))}
    rows = []
    for s in STATIONS.itertuples():
        for we in (0, 1):
            a = lambda k: float(avg.get((s.Station_ID, we, k % 24), 0))
            dm = float(np.mean([a(k) for k in range(24)]))
            for h in range(24):
                rows.append(dict(sid=s.Station_ID, we=we, h=h, hour=h, day_of_week=5 if we else 2, month=int(f.month.iloc[-1]),
                                 is_weekend=we, station_code=codes[s.Station_ID], type_code=TYPES.index(s.Type),
                                 chargers=s.Chargers, power_kw=s.Power_kW, lag_1h=a(h - 1), lag_2h=a(h - 2), lag_24h=a(h),
                                 rolling_3h=(a(h - 1) + a(h - 2) + a(h - 3)) / 3, rolling_24h=dm))
    t = pd.DataFrame(rows)
    t["kwh"] = np.clip(model.predict(t[sel]), 0, None)
    t["u"] = (t["kwh"] / t["sid"].map(ref)).clip(.03, .98)
    return {(r.sid, r.we, r.h): (float(r.kwh), float(r.u)) for r in t.itertuples()}, metrics


def make_figures(f, out, model, sel):
    def save(fig, name):
        fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, name + ".png"), dpi=150, facecolor="white"); plt.close(fig)
    h = f.groupby("hour")[TARGET].mean()
    fig, ax = plt.subplots(figsize=(8, 3.6)); ax.plot(h.index, h.values, marker="o", color="#7c3aed", lw=2)
    ax.set(xlabel="Hour of day", ylabel="Average demand (kWh)", title="Hourly Charging Demand"); ax.grid(alpha=.3); save(fig, "hourly_demand")
    sid = out.groupby("Station_ID").Actual.sum().idxmax(); s = out[out.Station_ID == sid].head(168)
    fig, ax = plt.subplots(figsize=(8, 3.6)); ax.plot(s.Hour, s.Actual, label="Actual", color="#1d4ed8")
    ax.plot(s.Hour, s.Predicted, "--", label="Predicted (XGBoost)", color="#dc2626")
    ax.set(xlabel="Time", ylabel="Demand (kWh)", title="Actual vs Predicted Demand"); ax.legend(); ax.grid(alpha=.3); fig.autofmt_xdate(); save(fig, "actual_vs_predicted")
    imp = pd.Series(model.feature_importances_, index=sel).sort_values()
    fig, ax = plt.subplots(figsize=(8, 3.8)); ax.barh(imp.index, imp.values, color="#16a34a", edgecolor="black")
    ax.set(xlabel="Importance score", title="XGBoost Feature Importance"); save(fig, "feature_importance")
    st = out.groupby("Station_ID")[["Actual", "Predicted"]].mean(); x = np.arange(len(st))
    fig, ax = plt.subplots(figsize=(8, 3.6)); ax.bar(x - .2, st.Actual, .4, label="Actual", edgecolor="black")
    ax.bar(x + .2, st.Predicted, .4, label="Predicted", edgecolor="black")
    ax.set(xticks=x, xticklabels=st.index, ylabel="Average kWh per hour", title="Station-wise Predicted Demand")
    ax.tick_params(axis="x", labelsize=7); ax.legend(); save(fig, "station_wise_demand")


# ============================================================ RECOMMENDATION LOGIC
DEMO_NOTICE = True        # set to False when you use real station data


def level(u):
    return ("lo", "Quiet") if u < .4 else ("me", "Moderate") if u < .7 else ("hi", "Busy")


def card(s, d, we, hour, u_kwh, extra=None):
    kwh, u = u_kwh
    c, txt = level(u)
    sc = round((.4 * max(0, 1 - d / 30) + .4 * (1 - u) + .2 * s.Power_kW / 120) * 100)
    r = dict(name=s.Name, dist=round(d, 1), kwh=round(kwh, 1), busy=round(u * 100), css=c, level=txt, wait=round(u ** 3 * 45),
             chargers=int(s.Chargers), kw=int(s.Power_kW), conn=s.Connector, price=float(s.Price), score=sc,
             lat=float(s.Lat), lng=float(s.Lng))
    r.update(extra or {})
    return r


def nav_url(origin, dest, via=None):
    """Google Maps driving directions. No origin = the phone's live GPS position; via = a charging stop on the way."""
    u = "https://www.google.com/maps/dir/?api=1&travelmode=driving&hl=en&destination=%.6f,%.6f" % (dest[0], dest[1])
    if origin: u += "&origin=%.6f,%.6f" % (origin[0], origin[1])
    if via: u += "&waypoints=%.6f,%.6f" % (via[0], via[1])
    return u


def nearby(lat, lng, hour, we, conn, top=6, origin=None):
    rows = [card(s, km(lat, lng, s.Lat, s.Lng), we, hour, LOOK[(s.Station_ID, we, hour)], dict(nav=nav_url(origin, (s.Lat, s.Lng))))
            for s in STATIONS.itertuples() if conn == "any" or s.Connector == conn]
    return sorted(rows, key=lambda r: -r["score"])[:top]


def road_route(a, b):
    """Real driving route from the free OpenStreetMap (OSRM) server: (km, minutes, [[lat, lng], ...]), or None if unavailable."""
    try:
        url = f"https://router.project-osrm.org/route/v1/driving/{a[1]},{a[0]};{b[1]},{b[0]}?overview=full&geometries=geojson"
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Voltnexa/1.0"}), timeout=8) as r:
            rt = json.load(r)["routes"][0]
        return rt["distance"] / 1000, rt["duration"] / 60, [[c[1], c[0]] for c in rt["geometry"]["coordinates"]]
    except Exception:
        return None


def cum_len(pts):
    c = [0.0]
    for i in range(1, len(pts)):
        c.append(c[-1] + km(pts[i - 1][0], pts[i - 1][1], pts[i][0], pts[i][1]))
    return c


def along(pts, cum, la, ln):
    """Closest point of the route to a station: (distance in km from the route, fraction 0-1 along the route)."""
    k = math.cos(math.radians(la)) * 111.2
    best = (1e9, 0.0)
    for i in range(len(pts) - 1):
        x0, y0 = pts[i][1], pts[i][0]
        x1, y1 = (pts[i + 1][1] - x0) * k, (pts[i + 1][0] - y0) * 111.2
        px, py = (ln - x0) * k, (la - y0) * 111.2
        L2 = x1 * x1 + y1 * y1
        t = max(0.0, min(1.0, (px * x1 + py * y1) / L2)) if L2 else 0.0
        d = math.hypot(px - t * x1, py - t * y1)
        if d < best[0]:
            best = (d, cum[i] + t * (cum[i + 1] - cum[i]))
    return best[0], best[1] / (cum[-1] or 1)


def clock(minutes):
    m = int(minutes) % 1440
    return f"{(m // 60) % 12 or 12}:{m % 60:02d} {'AM' if m < 720 else 'PM'}"


def plan_trip(a, b, range_km, charge, depart_min, we, conn, gps_start=False):
    route = road_route(a, b)                                    # real road route when the internet is available
    live = route is not None
    if live:
        road, drive_min, pts = route
    else:                                                       # fallback: straight line x 1.25, average speed 50 km/h
        road = km(a[0], a[1], b[0], b[1]) * 1.25
        drive_min, pts = road / 50 * 60, [list(a), list(b)]
    cum = cum_len(pts)
    usable = range_km * charge / 100 * 0.9                      # keep 10% battery in reserve
    stops = []
    for s in STATIONS.itertuples():
        if conn != "any" and s.Connector != conn:
            continue
        off, frac = along(pts, cum, s.Lat, s.Lng)
        if off <= 10:                                           # stations within 10 km of the route
            arrive = depart_min + frac * drive_min
            stops.append(card(s, off, we, 0, LOOK[(s.Station_ID, we, int(arrive // 60) % 24)],
                              dict(prog=round(frac * road), arrive=clock(arrive), off=round(off, 1),
                                   nav=nav_url(None if gps_start else a, b, (s.Lat, s.Lng)))))
    stops.sort(key=lambda r: r["prog"])
    reach = [x for x in stops if x["prog"] <= usable]
    best = None
    hrs = f"{int(drive_min // 60)} h {int(drive_min % 60)} min" if drive_min >= 60 else f"{int(drive_min)} min"
    head = f"The driving route is about {road:.0f} km ({hrs}{'' if live else ', estimated'}). "
    if road <= usable:
        msg = head + f"Your battery should cover it (about {usable:.0f} km usable). Optional stops are listed below."
    elif reach:
        quiet = [x for x in reach if x["busy"] < 70]
        best = max(quiet, key=lambda x: x["prog"]) if quiet else min(reach, key=lambda x: x["busy"])
        msg = head + (f"Your battery covers about {usable:.0f} km. Recommended charging stop: {best['name']} at km {best['prog']}, "
                      f"arriving around {best['arrive']} ({best['busy']}% busy).")
    else:
        msg = head + f"Your battery covers about {usable:.0f} km and no listed station is reachable on this route, so charge before you leave."
    for x in stops:
        x["best"] = bool(best and x["name"] == best["name"])
    step = max(1, len(pts) // 600)                              # fewer points for the map line
    return dict(road=round(road), usable=round(usable), msg=msg, stops=stops, live=live, pts=pts[::step] + [pts[-1]])


# ============================================================ WEBSITE (Flask)
CSS = """
:root{--bg:#f6f9f7;--card:#fff;--tx:#10201a;--mu:#5a6b62;--ac:#0f9d58;--ac2:#e6f5ec;--bd:#dbe6df;--warn:#d9822b;--bad:#c8433a;--h1:#0b3d2a;--h2:#0f7a4d}
@media (prefers-color-scheme:dark){:root{--bg:#0d1511;--card:#15211a;--tx:#e8f2ec;--mu:#98ad9f;--ac:#3fcf8e;--ac2:#1a3326;--bd:#26382d;--warn:#f0a050;--bad:#ef6f66;--h1:#08281b;--h2:#0b5a38}}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--tx);font:16px/1.65 "Segoe UI",system-ui,-apple-system,sans-serif}
.w{max-width:1040px;margin:0 auto;padding:0 20px}
nav{position:sticky;top:0;z-index:1000;background:var(--card);border-bottom:1px solid var(--bd)}
nav .w{display:flex;align-items:center;justify-content:space-between;height:64px;gap:10px}
.brand{display:flex;align-items:center;gap:10px;text-decoration:none;color:var(--tx)}.bn{font-size:23px;font-weight:800;letter-spacing:-.4px}.bn em{font-style:normal;color:var(--ac)}
.links{display:flex;align-items:center;gap:4px}.links a{color:var(--mu);text-decoration:none;font-weight:600;padding:8px 13px;border-radius:99px;font-size:15px}.links a.on{color:var(--ac);background:var(--ac2)}
.links a.cta{background:var(--ac);color:#04150c;margin-left:6px}
h1{font-size:clamp(30px,5.2vw,52px);line-height:1.1;margin:0 0 14px;letter-spacing:-1px}h2{font-size:clamp(23px,3vw,30px);margin:0 0 8px;letter-spacing:-.4px}h3{margin:0 0 4px;font-size:17px}
.lead{font-size:18px;color:var(--mu);max-width:660px;margin:0 0 22px}.m{color:var(--mu);font-size:14px}
.btn,button{background:var(--ac);color:#04150c;border:0;border-radius:11px;padding:12px 22px;font-size:15px;font-weight:700;cursor:pointer;text-decoration:none;display:inline-block;font-family:inherit}
.btn.o{background:transparent;color:var(--ac);border:1.5px solid var(--ac)}
.hero{background:linear-gradient(135deg,var(--h1),var(--h2));color:#fff;padding:68px 0 74px}
.hero .lead{color:#cfe9db}.eyebrow{display:inline-block;background:rgba(255,255,255,.14);padding:5px 14px;border-radius:99px;font-size:13px;font-weight:600;margin-bottom:18px}
.hform{background:var(--card);color:var(--tx);border-radius:16px;padding:14px;max-width:720px;box-shadow:0 12px 40px rgba(0,0,0,.25)}
.hrow{display:flex;gap:10px}.hrow .loc-box{flex:1}.link{background:none;color:var(--ac);padding:10px 4px 2px;font-size:14px}
section{padding:46px 0 6px}.g3{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:16px;margin-top:20px}
.card{background:var(--card);border:1px solid var(--bd);border-radius:16px;padding:22px}.ico{font-size:26px;margin-bottom:6px}
.step{counter-increment:s}.step:before{content:counter(s);display:grid;place-items:center;width:34px;height:34px;border-radius:50%;background:var(--ac);color:#04150c;font-weight:800;margin-bottom:10px}
.steps{counter-reset:s}.band{background:var(--ac2);border:1px solid var(--bd);border-radius:18px;padding:30px;display:flex;justify-content:space-between;align-items:center;gap:16px;flex-wrap:wrap;margin-top:40px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px;margin-top:-34px;position:relative}.stats .card{text-align:center;padding:18px}.stats b{display:block;font-size:26px;color:var(--ac)}
.tabs{display:flex;gap:8px;margin:0 0 16px}.tabs a{padding:9px 20px;border-radius:99px;text-decoration:none;font-weight:700;color:var(--mu);border:1px solid var(--bd);background:var(--card)}.tabs a.on{background:var(--ac);color:#04150c;border-color:var(--ac)}
.f{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px}.f .wide{grid-column:1/-1}
label{font-size:13px;color:var(--mu);display:block;margin-bottom:5px;font-weight:600}
select,input{width:100%;padding:11px;border-radius:10px;border:1px solid var(--bd);background:var(--bg);color:var(--tx);font-size:15px;font-family:inherit}
.loc-box{position:relative}.sug{position:absolute;left:0;right:0;top:100%;z-index:1100;background:var(--card);border:1px solid var(--bd);border-radius:10px;box-shadow:0 10px 30px rgba(0,0,0,.18);overflow:hidden}
.sug:empty{display:none}.sug button{display:block;width:100%;text-align:left;background:var(--card);color:var(--tx);border-radius:0;font-weight:500;font-size:14px;padding:10px 12px;border-bottom:1px solid var(--bd)}.sug button:hover{background:var(--ac2)}
.mini{margin-top:8px;padding:7px 13px;font-size:13px;background:var(--ac2);color:var(--ac)}.go{margin-top:18px}
#map{height:380px;border-radius:16px;border:1px solid var(--bd);margin-top:18px;z-index:1}
.res{display:grid;gap:12px;margin-top:18px}.row{display:grid;grid-template-columns:46px 1fr auto;gap:16px;align-items:center}
.rk{width:46px;height:46px;border-radius:50%;background:var(--ac2);color:var(--ac);display:grid;place-items:center;font-weight:800;font-size:15px;text-align:center;line-height:1.1}
.pill{display:inline-block;font-size:12px;padding:2px 10px;border-radius:99px;margin-right:6px;font-weight:700}
.lo{background:var(--ac2);color:var(--ac)}.me{background:rgba(217,130,43,.18);color:var(--warn)}.hi{background:rgba(200,67,58,.18);color:var(--bad)}
.sc{text-align:right}.sc b{font-size:24px;color:var(--ac)}.best{border:2px solid var(--ac)}.go2{font-size:13px;padding:7px 14px;margin-top:8px}
.note{background:rgba(217,130,43,.12);border:1px solid var(--warn);border-radius:12px;padding:12px 16px;margin-top:16px;font-size:14px}
.ch{display:flex;align-items:flex-end;gap:3px;height:120px;margin-top:10px}.ch div{flex:1;background:var(--ac);border-radius:3px 3px 0 0;opacity:.4;min-height:3px}.ch div.on{opacity:1}
.ax{display:flex;justify-content:space-between;font-size:12px;color:var(--mu);margin-top:4px}
ul.o{margin:6px 0 0;padding-left:20px;color:var(--mu)}img.fig{max-width:100%;border:1px solid var(--bd);border-radius:14px;background:#fff;margin-top:12px}
details{border-bottom:1px solid var(--bd);padding:14px 0}summary{cursor:pointer;font-weight:700}details p{margin:8px 0 0;color:var(--mu)}
footer{padding:30px 0 50px;color:var(--mu);font-size:13.5px;border-top:1px solid var(--bd);margin-top:50px}
.auth{display:flex;align-items:center;gap:8px}.auth form{margin:0}.hi{font-weight:600;color:var(--mu);font-size:14px;margin-right:4px}
.btn.sm,button.sm{padding:8px 16px;font-size:14px}.btn.o.sm{background:transparent}
.tabbar{display:none}
.card.auth-card{max-width:520px;margin:0 auto}.f1{display:grid;gap:14px}.f2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.ph{display:flex;gap:8px}.ph .pre{display:grid;place-items:center;padding:0 12px;border:1px solid var(--bd);border-radius:10px;background:var(--bg);color:var(--mu);font-weight:700}
.ph input{flex:1;min-width:0}.ph button{white-space:nowrap}
.hint{font-size:13px;margin-top:6px;min-height:19px;color:var(--mu)}.hint.ok{color:var(--ac)}.hint.bad{color:var(--bad)}
.pw{position:relative}.pw input{padding-right:64px}.pw button{position:absolute;right:6px;top:50%;transform:translateY(-50%);padding:6px 10px;font-size:12px;background:var(--ac2);color:var(--ac)}
.meter{height:7px;border-radius:99px;background:var(--bd);margin-top:8px;overflow:hidden}.meter i{display:block;height:100%;width:0;background:var(--bad);transition:width .25s,background .25s}
.rules{list-style:none;margin:8px 0 0;padding:0;font-size:13px;color:var(--mu);display:grid;grid-template-columns:1fr 1fr;gap:2px 12px}.rules li:before{content:"○ "}.rules li.ok{color:var(--ac)}.rules li.ok:before{content:"✓ "}
.err{background:rgba(200,67,58,.12);border:1px solid var(--bad);color:var(--bad);border-radius:12px;padding:10px 14px;font-size:14px;margin-top:10px}.err[hidden]{display:none}
.okbox{background:var(--ac2);border:1px solid var(--ac);color:var(--ac);border-radius:12px;padding:10px 14px;font-size:14px;margin-top:10px}
.clock{display:inline-block;background:var(--ac2);color:var(--ac);border-radius:99px;padding:3px 12px;font-size:13px;font-weight:700;margin-left:6px}
.inv{border-color:var(--bad)!important}button:disabled{opacity:.5;cursor:not-allowed}
@media(max-width:700px){
 body.app{padding-bottom:calc(66px + env(safe-area-inset-bottom))}.w{padding:0 16px}
 nav .w{height:56px}.links,.hi{display:none}.bn{font-size:20px}
 .hero{padding:38px 0 52px}.lead{font-size:16px}.hform{padding:10px}.stats{margin-top:-26px}.stats b{font-size:22px}
 section{padding:32px 0 4px}.band{padding:22px}.band .btn{width:100%;text-align:center}
 .ph{gap:6px}.ph .pre{padding:0 9px}.ph button{padding:10px 12px;font-size:13px}
 .hrow{flex-direction:column}.hrow .btn{width:100%}.f{grid-template-columns:1fr}.f2{grid-template-columns:1fr}.rules{grid-template-columns:1fr}
 input,select,textarea{font-size:16px;min-height:46px}.btn,button{min-height:44px}.go{width:100%}
 #map{height:300px}.row{grid-template-columns:42px 1fr}.sc{grid-column:2;text-align:left}.card{padding:18px}
 .tabbar{display:flex;position:fixed;left:0;right:0;bottom:0;z-index:1200;background:var(--card);border-top:1px solid var(--bd);padding:6px 4px calc(6px + env(safe-area-inset-bottom))}
 .tabbar a,.tabbar button{flex:1;display:flex;flex-direction:column;align-items:center;gap:2px;background:none;border:0;border-radius:10px;color:var(--mu);font-size:11.5px;font-weight:700;text-decoration:none;padding:4px 0;min-height:48px;justify-content:center}
 .tabbar span{font-size:20px;line-height:1}.tabbar .on{color:var(--ac)}.tabbar form{flex:1;display:flex;margin:0}
 footer{margin-top:30px}
}
@media(display-mode:standalone){body{-webkit-tap-highlight-color:transparent}}
"""

JS = """
var NOM="https://nominatim.openstreetmap.org/";
function setLoc(box,name,la,ln){box.querySelector(".q").value=name;box.querySelector(".lat").value=la;box.querySelector(".lng").value=ln;var m=box.querySelector(".sug");if(m)m.innerHTML=""}
function debounce(f,ms){var t;return function(){clearTimeout(t);t=setTimeout(f,ms)}}
function geocode(q,n){return fetch(NOM+"search?format=jsonv2&accept-language=en&limit="+n+"&countrycodes=in&q="+encodeURIComponent(q)).then(function(r){return r.json()})}
document.querySelectorAll(".loc-box").forEach(function(box){
 var q=box.querySelector(".q"),sug=box.querySelector(".sug");
 q.addEventListener("input",debounce(function(){box.querySelector(".lat").value="";box.querySelector(".lng").value="";var v=q.value.trim();sug.innerHTML="";if(v.length<3)return;
  geocode(v,5).then(function(d){sug.innerHTML="";d.forEach(function(x){var b=document.createElement("button");b.type="button";b.textContent=x.display_name;b.onclick=function(){setLoc(box,x.display_name,x.lat,x.lon)};sug.appendChild(b)})}).catch(function(){})},450))});
function submitForm(f){if(f.requestSubmit)f.requestSubmit();else if(timeOk(f))f.submit()}
function pad(n){return (n<10?"0":"")+n}
function nowInfo(){var d=new Date();return {date:d.getFullYear()+"-"+pad(d.getMonth()+1)+"-"+pad(d.getDate()),time:pad(d.getHours())+":"+pad(d.getMinutes()),mins:d.getHours()*60+d.getMinutes()}}
function t12(t){var h=+t.slice(0,2),m=t.slice(3,5);return ((h%12)||12)+":"+m+" "+(h<12?"AM":"PM")}
function timeOk(f){                       // false (and shows an error) when the chosen date/time has already passed
 if(!f||!f.date||!f.time)return true;
 var n=nowInfo(),box=document.getElementById("timeerr"),msg="";
 if(f.cnow)f.cnow.value=n.date+" "+n.time;
 if(!f.date.value)msg="Please choose a date.";
 else if(!f.time.value)msg="Please choose a time.";
 else if(f.date.value<n.date)msg="That date has already passed. Please choose today or a later date.";
 else if(f.date.value===n.date){var p=f.time.value.split(":");if(+p[0]*60+ +p[1]<n.mins-1)msg="That time has already passed today. It is now "+t12(n.time)+", so please choose "+t12(n.time)+" or a later time.";}
 f.date.classList.toggle("inv",!!msg&&/date/i.test(msg));f.time.classList.toggle("inv",!!msg&&!/date/i.test(msg));
 if(box){box.textContent=msg;box.hidden=!msg}
 return !msg}
document.querySelectorAll("form.geo").forEach(function(f){f.addEventListener("submit",function(e){
 if(f.dataset.live==="1"&&f.time)syncNow(f);
 if(!timeOk(f)){e.preventDefault();var b=document.getElementById("timeerr");if(b&&b.scrollIntoView)b.scrollIntoView({block:"center",behavior:"smooth"});return}
 var need=[].filter.call(f.querySelectorAll(".loc-box"),function(b){return !b.querySelector(".lat").value&&b.querySelector(".q").value.trim()});
 if(!need.length)return;e.preventDefault();
 Promise.all(need.map(function(b){return geocode(b.querySelector(".q").value.trim(),1).then(function(d){if(!d.length)return false;setLoc(b,d[0].display_name,d[0].lat,d[0].lon);return true}).catch(function(){return false})}))
 .then(function(r){if(r.every(Boolean))f.submit();else alert("We could not find that place. Try a nearby town or a landmark.")})})});
document.querySelectorAll(".useme").forEach(function(b){b.onclick=function(){var f=b.form,box=f.querySelector(".loc-box");
 if(!navigator.geolocation){alert("Location is not supported on this device.");return}
 var t=b.textContent;b.textContent="Locating…";
 navigator.geolocation.getCurrentPosition(function(p){setLoc(box,"My current location",p.coords.latitude,p.coords.longitude);if(f.dataset.auto)submitForm(f);else b.textContent="✓ Current location set"},
 function(){b.textContent=t;alert("Location access is blocked. Allow it in your browser, or type a place.")})}});
function syncNow(f){var n=nowInfo();f.date.value=n.date;f.time.value=n.time}
(function(){                              // date + time boxes show the real current date/time and keep updating until you change them
 var f=document.getElementById("mainform");if(!f||!f.date||!f.time)return;
 var lab=document.getElementById("clock"),live=f.dataset.live==="1";
 function tick(){var n=nowInfo();f.date.min=n.date;if(live)syncNow(f);if(lab)lab.textContent="Now "+t12(n.time);if(!live)timeOk(f)}
 tick();setInterval(tick,1000);
 function setLive(v){live=v;f.dataset.live=v?"1":"0"}
 setLive(live);
 [f.date,f.time].forEach(function(x){["input","change"].forEach(function(ev){x.addEventListener(ev,function(){setLive(false);timeOk(f)})})});
 document.querySelectorAll(".nowbtn").forEach(function(b){b.onclick=function(){setLive(true);syncNow(f);timeOk(f)}});
})();
/* register the service worker (installable app) and the Install button */
if("serviceWorker" in navigator)window.addEventListener("load",function(){navigator.serviceWorker.register("/sw.js").catch(function(){})});
var dip=null,ib=document.getElementById("inst"),std=window.matchMedia("(display-mode: standalone)").matches||navigator.standalone;
window.addEventListener("beforeinstallprompt",function(e){e.preventDefault();dip=e;if(ib&&!std)ib.hidden=false});
if(ib){ib.onclick=function(){if(dip){dip.prompt();dip=null;ib.hidden=true}else alert("To install: open the browser menu and choose Add to Home screen (on iPhone: Share > Add to Home Screen).")};
 if(/iphone|ipad|ipod/i.test(navigator.userAgent)&&!std)ib.hidden=false}
"""

T = {}
T["macros.html"] = """{% macro locbox(p, q, lat, lng, ph) %}<div class="loc-box"><input class="q" name="{{ p }}q" value="{{ q }}" placeholder="{{ ph }}" autocomplete="off">
<input type="hidden" class="lat" name="{{ p }}lat" value="{{ lat }}"><input type="hidden" class="lng" name="{{ p }}lng" value="{{ lng }}"><div class="sug"></div></div>{% endmacro %}"""

T["base.html"] = """<!DOCTYPE html><html lang="en" translate="no"><head><meta charset="utf-8"><meta name="google" content="notranslate">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#0f9d58"><link rel="manifest" href="/manifest.webmanifest"><link rel="icon" href="/icon-192.png">
<link rel="apple-touch-icon" href="/icon-192.png"><meta name="mobile-web-app-capable" content="yes"><meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-title" content="Voltnexa"><meta name="apple-mobile-web-app-status-bar-style" content="default">
<title>Voltnexa – Find EV charging stations anywhere, before you run low</title><style>""" + CSS + """</style>{% block head %}{% endblock %}</head><body class="{{ 'app' if user }}">
<nav><div class="w"><a class="brand" href="/"><svg width="36" height="36" viewBox="0 0 48 48"><rect width="48" height="48" rx="12" fill="var(--ac)"/><path d="M27 7 13 27h9l-3 14 16-22h-9z" fill="var(--bg)"/><circle cx="38" cy="10" r="3" fill="var(--bg)"/></svg><span class="bn">Volt<em>nexa</em></span></a>
<div class="links">{% if user %}<a href="/" class="{{ 'on' if request.endpoint == 'home' }}">Home</a><a href="/about" class="{{ 'on' if request.endpoint == 'about' }}">About</a><a href="/find" class="cta">Find a charger</a>{% endif %}</div>
<div class="auth">{% if user %}<span class="hi">Hi, {{ user.first_name }}</span><form method="post" action="/logout"><input type="hidden" name="csrf" value="{{ csrf() }}"><button class="btn o sm" type="submit">Logout</button></form>
{% else %}<a class="btn o sm" href="/login">Login</a><a class="btn sm" href="/register">Register</a>{% endif %}</div></div></nav>
{% block body %}{% endblock %}
{% if user %}<div class="tabbar"><a href="/" class="{{ 'on' if request.endpoint == 'home' }}"><span>🏠</span>Home</a><a href="/find" class="{{ 'on' if request.endpoint == 'find' }}"><span>⚡</span>Find</a><a href="/about" class="{{ 'on' if request.endpoint == 'about' }}"><span>ℹ️</span>About</a>
<form method="post" action="/logout"><input type="hidden" name="csrf" value="{{ csrf() }}"><button type="submit"><span>👤</span>Logout</button></form></div>{% endif %}
<div class="w"><footer><b>Voltnexa</b> · Smarter EV charging for every journey.<br><button type="button" id="inst" class="btn o sm" hidden style="margin:8px 0">📲 Install the Voltnexa app</button><br>Crowd levels and waiting times are estimates from historical charging patterns, not live availability. Please confirm with the station operator.{% if demo %}<br>Demo data: station details and demand shown here are simulated.{% endif %}</footer></div>
<script>""" + JS + """</script>{% block scripts %}{% endblock %}</body></html>"""

T["home.html"] = """{% extends "base.html" %}{% from "macros.html" import locbox %}{% block body %}
<div class="hero"><div class="w">
<span class="eyebrow">⚡ EV charging, planned for the road</span>
<h1>Never get stuck searching<br>for a charger again.</h1>
<p class="lead">Voltnexa shows which charging stations will be quiet when you arrive, near you or along your outstation route, so you can drive with confidence.</p>
<form class="hform geo" data-auto="1" data-live="1" action="/find" method="get"><input type="hidden" name="date"><input type="hidden" name="time"><input type="hidden" name="cnow">
<div class="hrow">{{ locbox("", "", "", "", "Where are you? Search any place, address or landmark") }}<button class="btn" type="submit">Find chargers</button></div>
<button type="button" class="link useme">📍 Use my current location</button></form>
</div></div>
<div class="w"><div class="stats">
<div class="card"><b>{{ n }}</b><span class="m">charging stations</span></div>
<div class="card"><b>24 / 7</b><span class="m">hourly crowd forecasts</span></div>
<div class="card"><b>Anywhere</b><span class="m">search any place in India</span></div></div>

<section><h2>Built for drivers on the move</h2><p class="m">Whether it is your daily commute or a long outstation drive, find the right charger faster.</p>
<div class="g3">
<div class="card"><div class="ico">📊</div><h3>Know the crowd before you go</h3><p class="m">See whether a station is quiet, moderate or busy at the hour you will arrive, and the expected waiting time.</p></div>
<div class="card"><div class="ico">🛣️</div><h3>Plan long drives with confidence</h3><p class="m">Enter your route and battery level. Voltnexa suggests a charging stop along the way with the least crowd.</p></div>
<div class="card"><div class="ico">📍</div><h3>Search anywhere, or use your location</h3><p class="m">Type any town, address or landmark or use your current location with one tap.</p></div></div></section>

<section><h2>How it works</h2><div class="g3 steps">
<div class="card step"><h3>Choose where</h3><p class="m">Search a place or use your current location, and pick when you want to charge.</p></div>
<div class="card step"><h3>See what to expect</h3><p class="m">Stations are ranked by distance, predicted crowd and charging speed.</p></div>
<div class="card step"><h3>Navigate</h3><p class="m">Open directions in your maps app and drive straight to a charger that suits you.</p></div></div></section>

<div class="band"><div><h2 style="margin:0">Planning a long drive?</h2><p class="m" style="margin:4px 0 0">Get a charging stop along your route, timed for when you will actually arrive.</p></div><a class="btn" href="/find?mode=trip">Plan my route</a></div>
</div>{% endblock %}"""

T["find.html"] = """{% extends "base.html" %}{% from "macros.html" import locbox %}
{% block body %}<div class="w"><section>
<h2>Find a charging station</h2><p class="m">Search any place, or plan a route. Results show the predicted crowd at your time of travel.</p>
<div class="tabs" style="margin-top:16px"><a href="/find" class="{{ 'on' if mode == 'nearby' }}">Near a place</a><a href="/find?mode=trip" class="{{ 'on' if mode == 'trip' }}">Route planner</a></div>
<div class="card"><form id="mainform" class="geo" data-auto="{{ '1' if mode == 'nearby' }}" data-live="{{ '1' if live }}" action="/find" method="get" novalidate>
<input type="hidden" name="mode" value="{{ mode }}"><input type="hidden" name="go" value="1"><input type="hidden" name="cnow" value="">
<div class="f">
<div class="wide"><label>{{ 'Starting point' if mode == 'trip' else 'Where do you want to charge?' }}</label>{{ locbox("", q, lat, lng, "Search any place, address or landmark") }}<button class="mini useme" type="button">📍 Use my current location</button></div>
{% if mode == 'trip' %}<div class="wide"><label>Destination</label>{{ locbox("d_", dq, dlat, dlng, "Search your destination") }}</div>
<div><label>Battery range at 100% (km)</label><input type="number" name="rng" min="50" max="600" value="{{ rng }}"></div>
<div><label>Current charge (%)</label><input type="number" name="chg" min="5" max="100" value="{{ chg }}"></div>{% endif %}
<div><label>{{ 'Departure date' if mode == 'trip' else 'Date' }}</label><input type="date" name="date" value="{{ day }}" min="{{ today }}"></div>
<div><label>{{ 'Departure time' if mode == 'trip' else 'Time' }}<span class="clock" id="clock">Now</span></label><input type="time" name="time" value="{{ tm }}"><button class="mini nowbtn" type="button">Now</button></div>
<div class="wide err" id="timeerr" hidden></div>
<div><label>Charger type</label><select name="conn"><option value="any" {{ 'selected' if conn == 'any' }}>Any</option><option value="DC" {{ 'selected' if conn == 'DC' }}>DC fast</option><option value="AC" {{ 'selected' if conn == 'AC' }}>AC</option></select></div>
</div><button class="go" type="submit">{{ 'Plan my route' if mode == 'trip' else 'Find chargers' }}</button></form>
{% if error %}<div class="note">{{ error }}</div>{% endif %}</div>

{% if mode == 'nearby' and results %}
<p class="m" style="margin-top:16px">{{ 'Weekend' if we else 'Weekday' }} estimates for {{ clock }} near <b>{{ q }}</b>.</p>
{% if results[0].dist > 40 %}<div class="note">The nearest Voltnexa station is {{ results[0].dist }} km from this location. Station coverage is currently limited to the Bengaluru region and nearby highways.</div>{% endif %}
<div class="res">{% for r in results %}<div class="card row"><div class="rk">{{ loop.index }}</div>
<div><h3>{{ r.name }}</h3><div class="m"><span class="pill {{ r.css }}">{{ r.level }}</span>about {{ r.wait }} min wait · {{ r.kwh }} kWh expected demand</div>
<div class="m">{{ r.dist }} km away · {{ r.chargers }} chargers · {{ r.kw }} kW {{ r.conn }} · ₹{{ r.price }}/kWh</div>
<a class="btn o go2" target="_blank" rel="noopener" href="{{ r.nav }}">Navigate</a></div>
<div class="sc"><b>{{ r.score }}</b><div class="m">match score</div></div></div>{% endfor %}</div>
{% if bars %}<section style="padding-top:34px"><h2>Today’s crowd pattern</h2><p class="m">Predicted busy level by hour at {{ results[0].name }}.</p>
<div class="card" style="margin-top:12px"><div class="ch">{% for b in bars %}<div class="{{ 'on' if loop.index0 == hour }}" style="height:{{ b }}%"></div>{% endfor %}</div>
<div class="ax"><span>12 AM</span><span>6 AM</span><span>12 PM</span><span>6 PM</span><span>11 PM</span></div></div></section>{% endif %}
{% elif mode == 'trip' and trip %}
<div class="card best" style="margin-top:16px"><h3>{{ q }} → {{ dq }}</h3><p style="margin:6px 0 0">{{ trip.msg }}</p></div>
<div class="res">{% for r in trip.stops %}<div class="card row {{ 'best' if r.best }}"><div class="rk">km<br>{{ r.prog }}</div>
<div><h3>{{ r.name }}{% if r.best %} ✅ Recommended{% endif %}</h3><div class="m"><span class="pill {{ r.css }}">{{ r.level }}</span>arrive about {{ r.arrive }} · about {{ r.wait }} min wait</div>
<div class="m">{{ r.off }} km from your route · {{ r.chargers }} chargers · {{ r.kw }} kW {{ r.conn }} · ₹{{ r.price }}/kWh</div>
<a class="btn o go2" target="_blank" rel="noopener" href="{{ r.nav }}">Navigate</a></div>
<div class="sc"><b>{{ r.busy }}%</b><div class="m">busy</div></div></div>{% else %}<div class="note">No Voltnexa station was found within 10 km of this route. Station coverage is currently limited to the Bengaluru region and nearby highways.</div>{% endfor %}</div>
<p class="m" style="margin-top:14px">{% if trip.live %}Distance and driving time come from the road route (OpenStreetMap routing). Traffic is not included.{% else %}The road route could not be loaded, so distance and time are estimates (straight line x 1.25, 50 km/h).{% endif %} Navigate opens turn-by-turn directions through the recommended stop.</p>
{% endif %}
</section></div>{% endblock %}
"""

T["about.html"] = """{% extends "base.html" %}{% block body %}<div class="w"><section>
<h2>About Voltnexa</h2>
<p class="lead">Voltnexa helps electric vehicle drivers find a charging station that is convenient and not crowded, whether it is a daily trip or a long outstation drive.</p>
<div class="card"><h3>Why Voltnexa</h3><p class="m" style="margin:6px 0 0;font-size:15.5px">Finding a charger is easy. Finding one that is free when you arrive is not. Drivers often reach a station to find every charger in use, or spend precious battery driving from one station to the next. Voltnexa predicts how busy each station will be at the hour you arrive, so you can pick a station with confidence before you leave.</p></div>

<section><h2>What you can do</h2><div class="g3">
<div class="card"><div class="ico">📍</div><h3>Search any place</h3><p class="m">Type a town, address or landmark or use your current location.</p></div>
<div class="card"><div class="ico">📊</div><h3>See the expected crowd</h3><p class="m">Each station shows a quiet, moderate or busy rating and an estimated waiting time.</p></div>
<div class="card"><div class="ico">🛣️</div><h3>Plan outstation routes</h3><p class="m">Enter your battery level and destination to get a recommended charging stop along the way.</p></div>
<div class="card"><div class="ico">🧭</div><h3>Navigate in one tap</h3><p class="m">Open directions to any station in your maps app.</p></div></div></section>

<section><h2>How it works</h2><div class="g3 steps">
<div class="card step"><h3>Learn from charging history</h3><p class="m">Voltnexa studies how much electricity each station delivers hour by hour, across weekdays and weekends.</p></div>
<div class="card step"><h3>Predict the crowd</h3><p class="m">A machine learning model (XGBoost) estimates the demand at every station for every hour of the day.</p></div>
<div class="card step"><h3>Rank the best options</h3><p class="m">Stations are ranked by distance, expected crowd and charging speed for your time of travel.</p></div></div></section>

<section><h2>Typical demand through the day</h2><p class="m">Average charging demand across stations, by hour. Evenings and mid-mornings are usually the busiest.</p>
<img class="fig" src="/figures/hourly_demand.png" alt="Average charging demand by hour of the day"></section>

<section><h2>Good to know</h2><div class="card"><ul class="o">
<li>Crowd levels and waiting times are estimates based on historical charging patterns, not live availability.</li>
<li>Route distances and driving times come from the road route (OpenStreetMap routing) when the internet is available. Traffic is not included.</li>
<li>Station coverage is currently limited to the Bengaluru region and nearby highways. You can search anywhere in India, and Voltnexa shows the nearest stations it knows.</li>
<li>Always confirm availability and pricing with the station operator before you travel.</li></ul></div></section>

<section><h2>Frequently asked questions</h2>
<details><summary>Does Voltnexa show live availability?</summary><p>No. It shows predicted crowd levels based on charging history. Live availability depends on each station operator.</p></details>
<details><summary>Do I need an account?</summary><p>Yes. Create a free account with your phone number (verified by OTP) and email, then log in to search for chargers.</p></details>
<details><summary>How is my location used?</summary><p>Your location is used only when you choose “Use my current location”, to find stations near you.</p></details>
<details><summary>Can I search outside my city?</summary><p>Yes. You can search any place in India, but Voltnexa currently lists stations only in the Bengaluru region and nearby highways.</p></details></section>
</section></div>{% endblock %}"""


PWCHK = """
var PW=[["At least 8 characters",function(p){return p.length>=8}],["An uppercase letter (A-Z)",function(p){return /[A-Z]/.test(p)}],["A lowercase letter (a-z)",function(p){return /[a-z]/.test(p)}],
["A number (0-9)",function(p){return /[0-9]/.test(p)}],["A symbol (! @ # $ % ...)",function(p){return /[^A-Za-z0-9\\s]/.test(p)}],["No spaces",function(p){return p.length>0&&!/\\s/.test(p)}]];
document.querySelectorAll(".pw button").forEach(function(b){b.onclick=function(){var i=b.parentNode.querySelector("input");i.type=i.type==="password"?"text":"password";b.textContent=i.type==="password"?"Show":"Hide"}});
"""

T["login.html"] = """{% extends "base.html" %}{% block body %}<div class="w"><section>
<div class="card auth-card"><h2>Welcome back</h2><p class="m">Log in with your phone number or email and your password.</p>
{% if msg %}<div class="hint ok" style="margin-top:10px">{{ msg }}</div>{% endif %}
{% if error %}<div class="err">{{ error }}</div>{% endif %}
<form method="post" action="/login" class="f1" style="margin-top:14px" autocomplete="on">
<input type="hidden" name="csrf" value="{{ csrf() }}"><input type="hidden" name="next" value="{{ nxt }}">
<div><label for="ident">Phone number or email</label><input id="ident" name="ident" value="{{ ident }}" required autocomplete="username" inputmode="email" placeholder="9876543210 or you@example.com"></div>
<div><label for="pw">Password</label><div class="pw"><input id="pw" name="password" type="password" required autocomplete="current-password" placeholder="Your password"><button type="button">Show</button></div></div>
<div style="text-align:right;margin-top:-4px"><a href="/forgot" style="color:var(--ac);font-weight:700;font-size:14px">Forgot password?</a></div>
<button class="go" type="submit" style="margin-top:4px">Log in</button></form>
<p class="m" style="margin:16px 0 0">New to Voltnexa? <a href="/register" style="color:var(--ac);font-weight:700">Create an account</a></p></div></section></div>{% endblock %}
{% block scripts %}<script>""" + PWCHK + """</script>{% endblock %}"""

T["forgot.html"] = """{% extends "base.html" %}{% block body %}<div class="w"><section>
<div class="card auth-card"><h2>Forgot password?</h2><p class="m">Enter the phone number you registered with. We send an OTP, then you can set a new password.</p>
{% if error %}<div class="err">{{ error }}</div>{% endif %}
<form id="rf" method="post" action="/forgot" class="f1" style="margin-top:14px" autocomplete="off" novalidate>
<input type="hidden" name="csrf" value="{{ csrf() }}">
<div><label for="ph">Registered phone number</label><div class="ph"><span class="pre">+91</span><input id="ph" name="phone" value="{{ phone }}" required inputmode="numeric" maxlength="10" autocomplete="tel-national" placeholder="10-digit mobile number"><button type="button" id="sendotp">Send OTP</button></div>
<div class="hint" id="phh"></div></div>
<div id="otpbox" hidden><label for="otp">Enter the 6-digit OTP sent to your phone</label><div class="ph"><input id="otp" name="otp" inputmode="numeric" maxlength="6" autocomplete="one-time-code" placeholder="••••••"><button type="button" id="verify">Verify</button></div>
<div class="hint" id="oth"></div></div>
<div id="pwbox" hidden>
<div><label for="pw">New password</label><div class="pw"><input id="pw" name="password" type="password" required autocomplete="new-password" placeholder="Create a strong password"><button type="button">Show</button></div>
<div class="meter"><i id="mt"></i></div><ul class="rules" id="rules"></ul></div>
<div style="margin-top:12px"><label for="pw2">Confirm new password</label><div class="pw"><input id="pw2" name="password2" type="password" required autocomplete="new-password" placeholder="Re-enter the new password"><button type="button">Show</button></div><div class="hint" id="mh"></div></div>
<button class="go" id="sub" type="submit" style="margin-top:12px" disabled>Change password</button></div></form>
<p class="m" style="margin:16px 0 0">Remembered it? <a href="/login" style="color:var(--ac);font-weight:700">Back to log in</a></p></div></section></div>{% endblock %}
{% block scripts %}<script>""" + PWCHK + r"""
var $=function(i){return document.getElementById(i)},verified=false,verPhone="",CS=document.querySelector("[name=csrf]").value,cool=0,cooltimer=null;
function hint(id,t,c){var e=$(id);e.textContent=t;e.className="hint "+(c||"")}
function phone(){return $("ph").value.replace(/\D/g,"")}
function phoneOk(){return /^[6-9][0-9]{9}$/.test(phone())}
function api(u,b){return fetch(u,{method:"POST",headers:{"Content-Type":"application/json","X-CSRF-Token":CS},body:JSON.stringify(b)}).then(function(r){return r.json()})}
$("rules").innerHTML=PW.map(function(r){return "<li>"+r[0]+"</li>"}).join("");
function check(){
 var p=$("pw").value,sc=PW.filter(function(r){return r[1](p)}).length,lis=$("rules").children;
 PW.forEach(function(r,i){lis[i].className=r[1](p)?"ok":""});
 $("mt").style.width=(sc/PW.length*100)+"%";$("mt").style.background=sc<4?"var(--bad)":sc<PW.length?"var(--warn)":"var(--ac)";
 var same=$("pw2").value.length>0&&$("pw2").value===p;
 if($("pw2").value)hint("mh",same?"Passwords match":"Passwords do not match",same?"ok":"bad");else hint("mh","");
 $("sub").disabled=!(verified&&phoneOk()&&sc===PW.length&&same)}
["pw","pw2"].forEach(function(i){$(i).addEventListener("input",check)});
$("ph").addEventListener("input",function(){this.value=this.value.replace(/\D/g,"").slice(0,10);hint("phh","")});
$("otp").addEventListener("input",function(){this.value=this.value.replace(/\D/g,"").slice(0,6)});
function startCool(n){cool=n;clearInterval(cooltimer);var b=$("sendotp");b.disabled=true;cooltimer=setInterval(function(){cool--;if(cool<=0){clearInterval(cooltimer);b.disabled=false;b.textContent="Resend OTP"}else b.textContent="Resend in "+cool+"s"},1000);b.textContent="Resend in "+cool+"s"}
$("sendotp").onclick=function(){
 if(!phoneOk()){hint("phh","Enter a valid 10-digit Indian mobile number (starts with 6-9).","bad");$("ph").focus();return}
 var b=this;b.disabled=true;b.textContent="Sending…";hint("phh","");
 api("/api/forgot-otp",{phone:phone()}).then(function(d){
  if(!d.ok){b.disabled=false;b.textContent="Send OTP";hint("phh",d.error||"Could not send the OTP.","bad");if(d.wait)startCool(d.wait);return}
  $("otpbox").hidden=false;hint("phh","OTP sent to +91 "+phone(),"ok");startCool(30);$("otp").value="";$("otp").focus();
  hint("oth",d.demo_otp?"Demo mode (no SMS service connected). Your OTP is "+d.demo_otp:"Check your messages. The OTP is valid for 5 minutes.",d.demo_otp?"ok":"")
 }).catch(function(){b.disabled=false;b.textContent="Send OTP";hint("phh","Network error. Please try again.","bad")})};
$("verify").onclick=function(){
 var o=$("otp").value;if(o.length!==6){hint("oth","Enter the 6-digit OTP.","bad");return}
 api("/api/forgot-verify",{phone:phone(),otp:o}).then(function(d){
  if(d.ok){verified=true;verPhone=phone();hint("oth","✓ Phone number verified. Now set your new password.","ok");$("ph").readOnly=true;$("verify").disabled=true;$("sendotp").disabled=true;$("pwbox").hidden=false;$("pw").focus();check()}
  else hint("oth",d.error||"Wrong OTP.","bad")}).catch(function(){hint("oth","Network error. Please try again.","bad")})};
$("rf").addEventListener("submit",function(e){if($("sub").disabled)e.preventDefault()});
</script>{% endblock %}"""

T["register.html"] = """{% extends "base.html" %}{% block body %}<div class="w"><section>
<div class="card auth-card"><h2>Create your account</h2><p class="m">Register with your phone number. We send a one-time password (OTP) to verify it.</p>
{% if error %}<div class="err">{{ error }}</div>{% endif %}
<form id="rf" method="post" action="/register" class="f1" style="margin-top:14px" autocomplete="on" novalidate>
<input type="hidden" name="csrf" value="{{ csrf() }}"><input type="hidden" name="next" value="{{ nxt }}">
<div class="f2"><div><label for="fn">First name</label><input id="fn" name="first_name" value="{{ v.first_name }}" required maxlength="50" autocomplete="given-name"></div>
<div><label for="ln">Last name</label><input id="ln" name="last_name" value="{{ v.last_name }}" required maxlength="50" autocomplete="family-name"></div></div>
<div><label for="em">Email</label><input id="em" name="email" type="email" value="{{ v.email }}" required maxlength="120" autocomplete="email" placeholder="you@example.com"></div>
<div><label for="ph">Phone number</label><div class="ph"><span class="pre">+91</span><input id="ph" name="phone" value="{{ v.phone }}" required inputmode="numeric" maxlength="10" autocomplete="tel-national" placeholder="10-digit mobile number"><button type="button" id="sendotp">Send OTP</button></div>
<div class="hint" id="phh"></div></div>
<div id="otpbox" hidden><label for="otp">Enter the 6-digit OTP sent to your phone</label><div class="ph"><input id="otp" name="otp" inputmode="numeric" maxlength="6" autocomplete="one-time-code" placeholder="••••••"><button type="button" id="verify">Verify</button></div>
<div class="hint" id="oth"></div></div>
<div><label for="pw">Set password</label><div class="pw"><input id="pw" name="password" type="password" required autocomplete="new-password" placeholder="Create a strong password"><button type="button">Show</button></div>
<div class="meter"><i id="mt"></i></div><ul class="rules" id="rules"></ul></div>
<div><label for="pw2">Confirm password</label><div class="pw"><input id="pw2" name="password2" type="password" required autocomplete="new-password" placeholder="Re-enter the password"><button type="button">Show</button></div><div class="hint" id="mh"></div></div>
<button class="go" id="sub" type="submit" style="margin-top:4px" disabled>Create account</button></form>
<p class="m" style="margin:16px 0 0">Already registered? <a href="/login" style="color:var(--ac);font-weight:700">Log in</a></p></div></section></div>{% endblock %}
{% block scripts %}<script>""" + PWCHK + r"""
var $=function(i){return document.getElementById(i)},verified={{ 'true' if pv else 'false' }},verPhone="",CS=document.querySelector("[name=csrf]").value,cool=0,cooltimer=null;
function hint(id,t,c){var e=$(id);e.textContent=t;e.className="hint "+(c||"")}
function phone(){return $("ph").value.replace(/\D/g,"")}
function phoneOk(){return /^[6-9][0-9]{9}$/.test(phone())}
function api(u,b){return fetch(u,{method:"POST",headers:{"Content-Type":"application/json","X-CSRF-Token":CS},body:JSON.stringify(b)}).then(function(r){return r.json()})}
$("rules").innerHTML=PW.map(function(r){return "<li>"+r[0]+"</li>"}).join("");
function pwScore(p){return PW.filter(function(r){return r[1](p)}).length}
function check(){
 var p=$("pw").value,sc=pwScore(p),lis=$("rules").children;
 PW.forEach(function(r,i){lis[i].className=r[1](p)?"ok":""});
 $("mt").style.width=(sc/PW.length*100)+"%";$("mt").style.background=sc<4?"var(--bad)":sc<PW.length?"var(--warn)":"var(--ac)";
 var same=$("pw2").value.length>0&&$("pw2").value===p;
 if($("pw2").value)hint("mh",same?"Passwords match":"Passwords do not match",same?"ok":"bad");else hint("mh","");
 if(verified&&verPhone!==phone()){verified=false;hint("oth","Phone number changed. Please verify again.","bad")}
 var ok=$("fn").value.trim()&&$("ln").value.trim()&&/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test($("em").value.trim())&&phoneOk()&&verified&&sc===PW.length&&same;
 $("sub").disabled=!ok}
["fn","ln","em","ph","pw","pw2","otp"].forEach(function(i){$(i).addEventListener("input",check)});
$("ph").addEventListener("input",function(){this.value=this.value.replace(/\D/g,"").slice(0,10);if(this.value.length===10&&!phoneOk())hint("phh","Enter a valid 10-digit Indian mobile number (starts with 6-9).","bad");else hint("phh","")});
$("otp").addEventListener("input",function(){this.value=this.value.replace(/\D/g,"").slice(0,6)});
function startCool(n){cool=n;clearInterval(cooltimer);var b=$("sendotp");b.disabled=true;cooltimer=setInterval(function(){cool--;if(cool<=0){clearInterval(cooltimer);b.disabled=false;b.textContent="Resend OTP"}else b.textContent="Resend in "+cool+"s"},1000);b.textContent="Resend in "+cool+"s"}
$("sendotp").onclick=function(){
 if(!phoneOk()){hint("phh","Enter a valid 10-digit Indian mobile number (starts with 6-9).","bad");$("ph").focus();return}
 var b=this;b.disabled=true;b.textContent="Sending…";hint("phh","");
 api("/api/send-otp",{phone:phone()}).then(function(d){
  if(!d.ok){b.disabled=false;b.textContent="Send OTP";hint("phh",d.error||"Could not send the OTP.","bad");if(d.wait)startCool(d.wait);return}
  $("otpbox").hidden=false;hint("phh","OTP sent to +91 "+phone(),"ok");startCool(30);$("otp").value="";$("otp").focus();
  hint("oth",d.demo_otp?"Demo mode (no SMS service connected). Your OTP is "+d.demo_otp:"Check your messages. The OTP is valid for 5 minutes.",d.demo_otp?"ok":"")
 }).catch(function(){b.disabled=false;b.textContent="Send OTP";hint("phh","Network error. Please try again.","bad")})};
$("verify").onclick=function(){
 var o=$("otp").value;if(o.length!==6){hint("oth","Enter the 6-digit OTP.","bad");return}
 api("/api/verify-otp",{phone:phone(),otp:o}).then(function(d){
  if(d.ok){verified=true;verPhone=phone();hint("oth","✓ Phone number verified","ok");$("ph").readOnly=true;$("verify").disabled=true;$("sendotp").disabled=true;check()}
  else hint("oth",d.error||"Wrong OTP.","bad")}).catch(function(){hint("oth","Network error. Please try again.","bad")})};
if(verified){verPhone=phone();$("otpbox").hidden=false;hint("oth","✓ Phone number verified","ok");$("ph").readOnly=true;$("verify").disabled=true;$("sendotp").disabled=true}check();
$("rf").addEventListener("submit",function(e){if($("sub").disabled)e.preventDefault()});
</script>{% endblock %}"""

T["offline.html"] = """{% extends "base.html" %}{% block body %}<div class="w"><section style="text-align:center;padding:60px 0">
<div class="ico" style="font-size:54px">📡</div><h2>You are offline</h2><p class="lead" style="margin:8px auto 18px">Voltnexa needs an internet connection to find charging stations. Please reconnect and try again.</p>
<a class="btn" href="/">Try again</a></section></div>{% endblock %}"""

app = Flask(__name__)
app.jinja_env.loader = DictLoader(T)               # template names end in .html so text is auto-escaped (safe)


@app.context_processor
def inject():
    return {"demo": DEMO_NOTICE}



# ============================================================ ACCOUNTS (SQLite), OTP, SECURITY
DATA_DIR = os.environ.get("VOLTNEXA_DATA", HERE)           # on a web host, point this at a persistent disk so accounts are kept
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "voltnexa_users.db")
KEY_PATH = os.path.join(DATA_DIR, "voltnexa_secret.key")
DEMO_OTP = os.environ.get("VOLTNEXA_DEMO_OTP", "1") != "0"      # show the OTP on screen when no SMS service is connected
OTP_TTL, OTP_TRIES, OTP_COOLDOWN, OTP_PER_HOUR = 300, 5, 30, 5
if os.environ.get("VOLTNEXA_SECRET"):
    app.secret_key = os.environ["VOLTNEXA_SECRET"]
else:                                                            # created once and reused so logins survive a restart
    if not os.path.exists(KEY_PATH):
        with open(KEY_PATH, "w") as fh: fh.write(secrets.token_hex(32))
    app.secret_key = open(KEY_PATH).read().strip()
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax", PERMANENT_SESSION_LIFETIME=timedelta(days=30),
                  MAX_CONTENT_LENGTH=64 * 1024)
from werkzeug.middleware.proxy_fix import ProxyFix                # the web host sits in front of the app: trust its https / client-address headers
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
if os.environ.get("VOLTNEXA_HTTPS") == "1":
    app.config["SESSION_COOKIE_SECURE"] = True


def sql(q, args=(), one=False, write=False):
    with contextlib.closing(sqlite3.connect(DB_PATH)) as c:
        c.row_factory = sqlite3.Row
        cur = c.execute(q, args)
        if write:
            c.commit(); return cur.lastrowid
        return cur.fetchone() if one else cur.fetchall()


def init_db():
    with contextlib.closing(sqlite3.connect(DB_PATH)) as c:
        c.executescript("""CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY AUTOINCREMENT, first_name TEXT NOT NULL, last_name TEXT NOT NULL,
            phone TEXT UNIQUE NOT NULL, email TEXT UNIQUE NOT NULL, pw_hash TEXT NOT NULL, created TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS otps(phone TEXT PRIMARY KEY, code_hash TEXT NOT NULL, expires REAL NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
            sent_at REAL NOT NULL, sent_count INTEGER NOT NULL DEFAULT 1, window_start REAL NOT NULL);""")
        c.commit()


def norm_phone(v):
    d = re.sub(r"[\s\-()]", "", v or "")
    d = d[3:] if d.startswith("+91") else d[2:] if (d.startswith("91") and len(d) == 12) else d[1:] if (d.startswith("0") and len(d) == 11) else d
    return d if re.fullmatch(r"[6-9][0-9]{9}", d) else None


def valid_name(v):
    v = (v or "").strip()
    return v if 1 <= len(v) <= 50 and all(ch.isalpha() or ch in " .'-" for ch in v) and any(ch.isalpha() for ch in v) else None


def valid_email(v):
    v = (v or "").strip().lower()
    return v if len(v) <= 120 and re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]{2,}", v) else None


COMMON = {"password", "password1", "password@123", "qwerty123", "12345678", "123456789", "iloveyou", "admin123", "welcome123", "abcd1234"}


def password_problem(pw, first="", phone="", email=""):
    """Returns a message describing what is wrong with the password, or None when it is strong."""
    if len(pw) < 8: return "Password must be at least 8 characters long."
    if len(pw) > 64: return "Password must be at most 64 characters long."
    if re.search(r"\s", pw): return "Password must not contain spaces."
    if not re.search(r"[A-Z]", pw): return "Password needs at least one uppercase letter."
    if not re.search(r"[a-z]", pw): return "Password needs at least one lowercase letter."
    if not re.search(r"[0-9]", pw): return "Password needs at least one number."
    if not re.search(r"[^A-Za-z0-9]", pw): return "Password needs at least one symbol such as ! @ # $ %."
    low = pw.lower()
    if low in COMMON: return "That password is too common. Please choose another."
    for bit in (first.lower(), phone, email.split("@")[0].lower() if email else ""):
        if len(bit) >= 4 and bit in low: return "Password must not contain your name, phone number or email."
    return None


def csrf():
    if "csrf" not in session: session["csrf"] = secrets.token_hex(16)
    return session["csrf"]


def csrf_ok():
    sent = request.headers.get("X-CSRF-Token") or request.form.get("csrf") or ""
    return bool(sent) and hmac.compare_digest(sent, session.get("csrf", ""))


def current_user():
    uid = session.get("uid")
    return sql("SELECT id, first_name, last_name FROM users WHERE id=?", (uid,), one=True) if uid else None


def safe_next(v):
    return v if v and v.startswith("/") and not v.startswith("//") and "\\" not in v else "/"


def hash_otp(phone, code):
    return hashlib.sha256((app.secret_key + phone + code).encode()).hexdigest()


def send_sms(phone, text):
    """Sends a real SMS through Twilio when TWILIO_SID / TWILIO_TOKEN / TWILIO_FROM are set. Returns True when sent."""
    sid, tok, frm = os.environ.get("TWILIO_SID"), os.environ.get("TWILIO_TOKEN"), os.environ.get("TWILIO_FROM")
    if not (sid and tok and frm):
        return False
    body = urllib.parse.urlencode({"To": "+91" + phone, "From": frm, "Body": text}).encode()
    req = urllib.request.Request(f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json", data=body)
    req.add_header("Authorization", "Basic " + base64.b64encode(f"{sid}:{tok}".encode()).decode())
    with urllib.request.urlopen(req, timeout=10):
        return True


FAILS = {}      # login attempts per account / address: key -> [count, locked_until]


def locked(key):
    f = FAILS.get(key)
    return bool(f and f[1] > time.time())


def fail(key):
    f = FAILS.setdefault(key, [0, 0]); f[0] += 1
    if f[0] >= 5: f[1] = time.time() + 300; f[0] = 0


@app.after_request
def headers(r):
    r.headers.setdefault("X-Content-Type-Options", "nosniff")
    r.headers.setdefault("X-Frame-Options", "DENY")
    r.headers.setdefault("Referrer-Policy", "same-origin")
    if request.endpoint in ("login", "register", "send_otp", "verify_otp", "forgot", "forgot_otp", "forgot_verify"):
        r.headers["Cache-Control"] = "no-store"
    return r


app.jinja_env.globals["csrf"] = csrf


@app.context_processor
def inject_user():
    return {"user": current_user()}


OPEN_PAGES = {"login", "register", "send_otp", "verify_otp", "forgot", "forgot_otp", "forgot_verify", "icon", "manifest", "sw", "offline", "healthz"}


@app.route("/healthz")
def healthz():
    return "ok"


def known(resp):
    """Remembers (for a year) that this browser has an account, so the next visit opens the Login page instead of Register."""
    resp.set_cookie("vn_known", "1", max_age=31536000, httponly=True, samesite="Lax", secure=request.is_secure)
    return resp


@app.before_request
def gate():
    """Opening Voltnexa without being logged in: new visitor -> Register page, already registered -> Login page."""
    if request.endpoint in OPEN_PAGES or request.endpoint is None or current_user():
        return None
    target = "/login" if request.cookies.get("vn_known") else "/register"
    here = request.full_path.rstrip("?")
    return redirect(target + ("" if here in ("", "/") else "?next=" + urllib.parse.quote(here, safe="/")))


def reg_page(v, error, nxt):
    pv = bool(norm_phone(v["phone"])) and session.get("otp_phone") == norm_phone(v["phone"]) and session.get("otp_until", 0) > time.time()
    return render_template("register.html", v=v, error=error, nxt=nxt, pv=pv)


@app.route("/register", methods=["GET", "POST"])
def register():
    nxt = safe_next(request.values.get("next"))
    if current_user(): return redirect(nxt)
    v = {k: (request.form.get(k) or "").strip() for k in ("first_name", "last_name", "email", "phone")}
    if request.method == "GET":
        return reg_page(v, None, nxt)
    err = None
    first, last, mail, ph = valid_name(v["first_name"]), valid_name(v["last_name"]), valid_email(v["email"]), norm_phone(v["phone"])
    pw, pw2 = request.form.get("password") or "", request.form.get("password2") or ""
    if not csrf_ok(): err = "Your session expired. Please try again."
    elif not first: err = "Please enter a valid first name (letters only)."
    elif not last: err = "Please enter a valid last name (letters only)."
    elif not mail: err = "Please enter a valid email address."
    elif not ph: err = "Please enter a valid 10-digit Indian mobile number."
    elif session.get("otp_phone") != ph or session.get("otp_until", 0) < time.time(): err = "Please verify your phone number with the OTP first."
    elif password_problem(pw, first, ph, mail): err = password_problem(pw, first, ph, mail)
    elif pw != pw2: err = "Password and confirm password do not match."
    if not err and sql("SELECT 1 FROM users WHERE phone=?", (ph,), one=True): err = "This phone number is already registered. Please log in."
    if not err and sql("SELECT 1 FROM users WHERE email=?", (mail,), one=True): err = "This email is already registered. Please log in."
    if err:
        return reg_page(v, err, nxt), 400
    uid = sql("INSERT INTO users(first_name,last_name,phone,email,pw_hash,created) VALUES (?,?,?,?,?,?)",
              (first, last, ph, mail, generate_password_hash(pw), datetime.now().isoformat(timespec="seconds")), write=True)
    session.clear(); session["uid"] = uid; session.permanent = True; csrf()
    return known(redirect(nxt))


@app.route("/login", methods=["GET", "POST"])
def login():
    nxt = safe_next(request.values.get("next"))
    if current_user(): return redirect(nxt)
    if request.method == "GET":
        return render_template("login.html", error=None, ident="", nxt=nxt, msg="Password changed successfully. Please log in with your new password." if request.args.get("reset") else None)
    ident = (request.form.get("ident") or "").strip()[:120]
    pw = request.form.get("password") or ""
    key = (valid_email(ident) or norm_phone(ident) or ident).lower()
    if not csrf_ok(): err = "Your session expired. Please try again."
    elif locked(key) or locked(request.remote_addr): err = "Too many wrong attempts. Please wait 5 minutes and try again."
    else:
        email, ph = valid_email(ident) if "@" in ident else None, norm_phone(ident) if "@" not in ident else None
        row = sql("SELECT * FROM users WHERE email=?", (email,), one=True) if email else sql("SELECT * FROM users WHERE phone=?", (ph,), one=True) if ph else None
        if row and check_password_hash(row["pw_hash"], pw):
            FAILS.pop(key, None); session.clear(); session["uid"] = row["id"]; session.permanent = True; csrf()
            return known(redirect(nxt))
        fail(key); fail(request.remote_addr)
        err = "Wrong phone number / email or password."
    return render_template("login.html", error=err, ident=ident, nxt=nxt), 400


@app.route("/logout", methods=["POST"])
def logout():
    if csrf_ok(): session.clear()
    return redirect("/")


@app.route("/api/send-otp", methods=["POST"])
def send_otp():
    if not csrf_ok(): return jsonify(ok=False, error="Session expired. Please refresh the page."), 400
    ph = norm_phone((request.get_json(silent=True) or {}).get("phone"))
    if not ph: return jsonify(ok=False, error="Enter a valid 10-digit Indian mobile number."), 400
    if sql("SELECT 1 FROM users WHERE phone=?", (ph,), one=True): return jsonify(ok=False, error="This phone number is already registered. Please log in."), 409
    now = time.time(); row = sql("SELECT * FROM otps WHERE phone=?", (ph,), one=True)
    count, wstart = 1, now
    if row:
        if now - row["sent_at"] < OTP_COOLDOWN:
            return jsonify(ok=False, error="Please wait before requesting another OTP.", wait=int(OTP_COOLDOWN - (now - row["sent_at"])) + 1), 429
        if now - row["window_start"] < 3600:
            if row["sent_count"] >= OTP_PER_HOUR: return jsonify(ok=False, error="Too many OTP requests. Please try again after an hour."), 429
            count, wstart = row["sent_count"] + 1, row["window_start"]
    code = f"{secrets.randbelow(10 ** 6):06d}"
    sql("INSERT OR REPLACE INTO otps(phone,code_hash,expires,attempts,sent_at,sent_count,window_start) VALUES (?,?,?,?,?,?,?)",
        (ph, hash_otp(ph, code), now + OTP_TTL, 0, now, count, wstart), write=True)
    msg = f"Your Voltnexa verification code is {code}. It is valid for 5 minutes. Do not share it with anyone."
    try:
        sent = send_sms(ph, msg)
    except Exception:
        return jsonify(ok=False, error="The SMS could not be sent right now. Please try again."), 502
    if not sent:
        print(f"[DEMO SMS] OTP for +91{ph}: {code}   (set TWILIO_SID / TWILIO_TOKEN / TWILIO_FROM to send real SMS)")
    return jsonify(ok=True, demo_otp=None if sent or not DEMO_OTP else code)


@app.route("/api/verify-otp", methods=["POST"])
def verify_otp():
    if not csrf_ok(): return jsonify(ok=False, error="Session expired. Please refresh the page."), 400
    d = request.get_json(silent=True) or {}
    ph, code = norm_phone(d.get("phone")), str(d.get("otp") or "")
    row = sql("SELECT * FROM otps WHERE phone=?", (ph,), one=True) if ph else None
    if not row or row["expires"] < time.time(): return jsonify(ok=False, error="The OTP has expired. Please request a new one."), 400
    if row["attempts"] >= OTP_TRIES: return jsonify(ok=False, error="Too many wrong attempts. Please request a new OTP."), 429
    if not re.fullmatch(r"[0-9]{6}", code) or not hmac.compare_digest(row["code_hash"], hash_otp(ph, code)):
        sql("UPDATE otps SET attempts=attempts+1 WHERE phone=?", (ph,), write=True)
        return jsonify(ok=False, error=f"Wrong OTP. {max(0, OTP_TRIES - row['attempts'] - 1)} attempts left."), 400
    sql("DELETE FROM otps WHERE phone=?", (ph,), write=True)
    session["otp_phone"], session["otp_until"] = ph, time.time() + 900
    return jsonify(ok=True)


# ============================================================ FORGOT PASSWORD (OTP to the registered phone, then set a new password)
@app.route("/api/forgot-otp", methods=["POST"])
def forgot_otp():
    if not csrf_ok(): return jsonify(ok=False, error="Session expired. Please refresh the page."), 400
    ph = norm_phone((request.get_json(silent=True) or {}).get("phone"))
    if not ph: return jsonify(ok=False, error="Enter a valid 10-digit Indian mobile number."), 400
    if not sql("SELECT 1 FROM users WHERE phone=?", (ph,), one=True): return jsonify(ok=False, error="This phone number is not registered. Please create an account."), 404
    now = time.time(); row = sql("SELECT * FROM otps WHERE phone=?", (ph,), one=True)
    count, wstart = 1, now
    if row:
        if now - row["sent_at"] < OTP_COOLDOWN:
            return jsonify(ok=False, error="Please wait before requesting another OTP.", wait=int(OTP_COOLDOWN - (now - row["sent_at"])) + 1), 429
        if now - row["window_start"] < 3600:
            if row["sent_count"] >= OTP_PER_HOUR: return jsonify(ok=False, error="Too many OTP requests. Please try again after an hour."), 429
            count, wstart = row["sent_count"] + 1, row["window_start"]
    code = f"{secrets.randbelow(10 ** 6):06d}"
    sql("INSERT OR REPLACE INTO otps(phone,code_hash,expires,attempts,sent_at,sent_count,window_start) VALUES (?,?,?,?,?,?,?)",
        (ph, hash_otp(ph, code), now + OTP_TTL, 0, now, count, wstart), write=True)
    msg = f"Your Voltnexa password reset code is {code}. It is valid for 5 minutes. Do not share it with anyone."
    try:
        sent = send_sms(ph, msg)
    except Exception:
        return jsonify(ok=False, error="The SMS could not be sent right now. Please try again."), 502
    if not sent:
        print(f"[DEMO SMS] Password reset OTP for +91{ph}: {code}   (set TWILIO_SID / TWILIO_TOKEN / TWILIO_FROM to send real SMS)")
    return jsonify(ok=True, demo_otp=None if sent or not DEMO_OTP else code)


@app.route("/api/forgot-verify", methods=["POST"])
def forgot_verify():
    if not csrf_ok(): return jsonify(ok=False, error="Session expired. Please refresh the page."), 400
    d = request.get_json(silent=True) or {}
    ph, code = norm_phone(d.get("phone")), str(d.get("otp") or "")
    row = sql("SELECT * FROM otps WHERE phone=?", (ph,), one=True) if ph else None
    if not row or row["expires"] < time.time(): return jsonify(ok=False, error="The OTP has expired. Please request a new one."), 400
    if row["attempts"] >= OTP_TRIES: return jsonify(ok=False, error="Too many wrong attempts. Please request a new OTP."), 429
    if not re.fullmatch(r"[0-9]{6}", code) or not hmac.compare_digest(row["code_hash"], hash_otp(ph, code)):
        sql("UPDATE otps SET attempts=attempts+1 WHERE phone=?", (ph,), write=True)
        return jsonify(ok=False, error=f"Wrong OTP. {max(0, OTP_TRIES - row['attempts'] - 1)} attempts left."), 400
    sql("DELETE FROM otps WHERE phone=?", (ph,), write=True)
    session["reset_phone"], session["reset_until"] = ph, time.time() + 900
    return jsonify(ok=True)


@app.route("/forgot", methods=["GET", "POST"])
def forgot():
    if current_user(): return redirect("/")
    if request.method == "GET":
        return render_template("forgot.html", error=None, phone="")
    ph = norm_phone(request.form.get("phone"))
    pw, pw2 = request.form.get("password") or "", request.form.get("password2") or ""
    row = sql("SELECT * FROM users WHERE phone=?", (ph,), one=True) if ph else None
    if not csrf_ok(): err = "Your session expired. Please try again."
    elif not ph or not row or session.get("reset_phone") != ph or session.get("reset_until", 0) < time.time(): err = "Please verify your phone number with the OTP first."
    elif password_problem(pw, row["first_name"], ph, row["email"]): err = password_problem(pw, row["first_name"], ph, row["email"])
    elif pw != pw2: err = "Password and confirm password do not match."
    else: err = None
    if err:
        return render_template("forgot.html", error=err, phone=request.form.get("phone", "")[:15]), 400
    sql("UPDATE users SET pw_hash=? WHERE phone=?", (generate_password_hash(pw), ph), write=True)
    session.pop("reset_phone", None); session.pop("reset_until", None)
    FAILS.pop(ph, None); FAILS.pop(row["email"], None)          # clear any login lock after a successful reset
    return redirect("/login?reset=1")


# ============================================================ INSTALLABLE APP (PWA): manifest, icons, service worker
def make_icon(size, maskable=False):
    from PIL import Image, ImageDraw
    k = 4; S = size * k
    im = Image.new("RGBA", (S, S), (15, 157, 88, 255) if maskable else (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    if not maskable: d.rounded_rectangle([0, 0, S - 1, S - 1], radius=int(S * .24), fill=(15, 157, 88, 255))
    sc = .5 if maskable else .62; ox = oy = S * (1 - sc) / 2
    bolt = [(27, 7), (13, 27), (22, 27), (19, 41), (35, 19), (26, 19)]          # same lightning bolt as the logo (48 x 48 grid)
    d.polygon([(ox + x / 48 * S * sc, oy + y / 48 * S * sc) for x, y in bolt], fill=(246, 249, 247, 255))
    buf = io.BytesIO(); im.resize((size, size), Image.LANCZOS).save(buf, "PNG"); return buf.getvalue()


ICONS = {}


@app.route("/icon-<name>.png")
def icon(name):
    spec = {"192": (192, False), "512": (512, False), "maskable": (512, True)}.get(name)
    if not spec: abort(404)
    if name not in ICONS: ICONS[name] = make_icon(*spec)
    return Response(ICONS[name], mimetype="image/png", headers={"Cache-Control": "public, max-age=86400"})


@app.route("/manifest.webmanifest")
def manifest():
    m = dict(name="Voltnexa - EV charging stations", short_name="Voltnexa", description="Find EV charging stations that will be quiet when you arrive.",
             start_url="/", scope="/", display="standalone", orientation="portrait", background_color="#f6f9f7", theme_color="#0f9d58",
             icons=[dict(src="/icon-192.png", sizes="192x192", type="image/png", purpose="any"), dict(src="/icon-512.png", sizes="512x512", type="image/png", purpose="any"),
                    dict(src="/icon-maskable.png", sizes="512x512", type="image/png", purpose="maskable")],
             shortcuts=[dict(name="Find a charger", url="/find"), dict(name="Plan a route", url="/find?mode=trip")])
    return Response(json.dumps(m), mimetype="application/manifest+json")


SW = """const V='voltnexa-v3',SHELL=['/offline','/icon-192.png','/icon-512.png'],PUBLIC=[];
self.addEventListener('install',e=>{e.waitUntil(caches.open(V).then(c=>c.addAll(SHELL)).then(()=>self.skipWaiting()))});
self.addEventListener('activate',e=>{e.waitUntil(caches.keys().then(k=>Promise.all(k.filter(x=>x!==V).map(x=>caches.delete(x)))).then(()=>self.clients.claim()))});
self.addEventListener('fetch',e=>{const r=e.request,u=new URL(r.url);
 if(r.method!=='GET'||u.origin!==location.origin||u.pathname.startsWith('/api/'))return;
 if(r.mode==='navigate'){e.respondWith(fetch(r).then(res=>{if(res.ok&&PUBLIC.includes(u.pathname)){const c=res.clone();caches.open(V).then(x=>x.put(r,c))}return res}).catch(()=>caches.match(r).then(m=>m||caches.match('/offline'))));return}
 if(SHELL.includes(u.pathname)||u.pathname.startsWith('/figures/')){e.respondWith(caches.match(r).then(m=>m||fetch(r).then(res=>{const c=res.clone();caches.open(V).then(x=>x.put(r,c));return res})))}});
"""


@app.route("/sw.js")
def sw():
    return Response(SW, mimetype="application/javascript", headers={"Service-Worker-Allowed": "/", "Cache-Control": "no-cache"})


@app.route("/offline")
def offline():
    return render_template("offline.html")


def time_error(day, hh, mm):
    """Error text when the chosen date/time has already passed (uses the visitor's own clock sent by the browser)."""
    try:
        cn = datetime.strptime(request.args.get("cnow", ""), "%Y-%m-%d %H:%M")
    except ValueError:
        cn = datetime.now()
    try:
        d = date.fromisoformat(day)
    except ValueError:
        return "Please choose a valid date."
    if d < cn.date():
        return "That date has already passed. Please choose today or a later date."
    if d == cn.date() and hh * 60 + mm < cn.hour * 60 + cn.minute - 1:
        return f"That time has already passed today. It is now {clock(cn.hour * 60 + cn.minute)}, so please choose {clock(cn.hour * 60 + cn.minute)} or a later time."
    return None


def num(name, lo, hi):
    try:
        v = float(request.args[name])
        return v if lo <= v <= hi else None
    except (KeyError, ValueError):
        return None


def point(prefix):          # a chosen place: (label, lat, lng), or None when nothing was chosen
    la, ln = num(prefix + "lat", -90, 90), num(prefix + "lng", -180, 180)
    if la is None or ln is None:
        return None
    return (request.args.get(prefix + "q", "").strip()[:150] or "Selected location", la, ln)


@app.route("/")
def home():
    return render_template("home.html", n=len(STATIONS))


@app.route("/find")
def find():
    mode = "trip" if request.args.get("mode") == "trip" else "nearby"
    now = datetime.now()
    live = not request.args.get("time")                      # a fresh form: date/time boxes follow the current time
    day = request.args.get("date") or now.date().isoformat()
    try:
        we = 1 if date.fromisoformat(day).weekday() >= 5 else 0
    except ValueError:
        day = now.date().isoformat(); we = 1 if now.weekday() >= 5 else 0
    tm = request.args.get("time") or now.strftime("%H:%M")
    try:
        hh, mm = [int(x) for x in tm.split(":")[:2]]
    except ValueError:
        hh, mm, tm = now.hour, now.minute, now.strftime("%H:%M")
    hh, mm = hh % 24, mm % 60
    conn = request.args.get("conn", "any"); conn = conn if conn in ("any", "AC", "DC") else "any"
    try:
        rng_km = max(50, min(600, int(request.args.get("rng", 250)))); chg = max(5, min(100, int(request.args.get("chg", 80))))
    except ValueError:
        rng_km, chg = 250, 80
    start, dest = point(""), point("d_")
    markers = []
    for s in STATIONS.itertuples():
        c, txt = level(LOOK[(s.Station_ID, we, hh)][1])
        markers.append(dict(n=s.Name, lat=float(s.Lat), lng=float(s.Lng), css=c, level=txt, kw=int(s.Power_kW), conn=s.Connector))
    ctx = dict(mode=mode, day=day, tm=tm, hour=hh, we=we, conn=conn, rng=rng_km, chg=chg, clock=clock(hh * 60),
               q=start[0] if start else "", lat=start[1] if start else "", lng=start[2] if start else "",
               dq=dest[0] if dest else "", dlat=dest[1] if dest else "", dlng=dest[2] if dest else "",
               results=[], bars=[], trip=None, error=None, live=live, today=now.date().isoformat())
    searching = bool(start) if mode == "nearby" else bool(request.args.get("go"))
    terr = time_error(day, hh, mm) if searching else None
    if terr:
        ctx["error"] = terr
    elif mode == "nearby" and start:
        ctx["results"] = nearby(start[1], start[2], hh, we, conn, 6, None if start[0] == "My current location" else (start[1], start[2]))
        if ctx["results"]:
            sid = STATIONS.loc[STATIONS.Name == ctx["results"][0]["name"], "Station_ID"].iloc[0]
            ctx["bars"] = [round(LOOK[(sid, we, k)][1] * 100) for k in range(24)]
    elif mode == "trip" and request.args.get("go"):
        if start and dest:
            ctx["trip"] = plan_trip(start[1:], dest[1:], rng_km, chg, hh * 60 + mm, we, conn, start[0] == "My current location")
        else:
            ctx["error"] = "Please choose both a starting point and a destination from the suggestions."
    ctx["data"] = dict(mode=mode, markers=markers, me=list(start) if start else None, dest=list(dest) if dest and mode == "trip" else None,
                       route=ctx["trip"]["pts"] if ctx["trip"] else None)
    return render_template("find.html", **ctx)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/figures/<name>")
def figures(name):
    return send_from_directory(FIG_DIR, name)


# ---- build the data, train the XGBoost model and create the figures once, when the server starts
init_db()
LOOK, METRICS = build_pipeline()


def lan_ip():
    try:
        t = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); t.connect(("8.8.8.8", 80)); ip = t.getsockname()[0]; t.close(); return ip
    except Exception:
        return "your-computer-ip"

if __name__ == "__main__":
    if "google.colab" in sys.modules:                       # Google Colab: run the site in a background thread and open a link
        import threading
        from google.colab import output
        threading.Thread(target=lambda: app.run(port=5000, debug=False, use_reloader=False), daemon=True).start()
        output.serve_kernel_port_as_window(5000)
    else:
        print("\nVoltnexa is running (press Ctrl+C to stop)")
        print("  Computer (Chrome):  http://127.0.0.1:5000")
        print(f"  Phone (same Wi-Fi): http://{lan_ip()}:5000")
        app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
