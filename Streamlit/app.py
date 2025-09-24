# app.py — Amazon Sales Analysis (Streamlit)
# ------------------------------------------------------------
# - Auto-loads Amazon-style CSV (supports .zip or .csv)
# - Cleans data (drops refunds/returns/cancellations, coerces types)
# - Shows KPIs, trends, revenue by category/products
# - Computes RFM segmentation and weekday × hour grid
# - Supports Dark Mode and formats revenue axes in "M" (millions)
# ------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os, zipfile

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="Amazon Sales Analysis", layout="wide")

# -------------------------------
# Dark mode toggle + CSS
# -------------------------------
dark = st.sidebar.toggle("Dark mode", value=True)

DARK_CSS = """
<style>
:root {--bg:#0b1220;--card:#111827;--fg:#e5e7eb;}
.stApp {background-color:var(--bg); color:var(--fg);}
.block-container {background:var(--bg);}
[data-testid="stSidebar"] {background-color:var(--card);}
h1,h2,h3,h4,h5,h6, p, span, div {color:var(--fg) !important;}
</style>
"""
LIGHT_CSS = "<style></style>"

st.markdown(DARK_CSS if dark else LIGHT_CSS, unsafe_allow_html=True)
plt.style.use("dark_background" if dark else "default")

fmtM = mticker.FuncFormatter(lambda x, _: f'{x * 1e-6:.1f}M')

# -------------------------------
# Load CSV (support .csv or .zip)
# -------------------------------
@st.cache_data
def load_csv(file):
    if str(file).endswith(".zip"):
        with zipfile.ZipFile(file) as z:
            # assume the first CSV file inside
            csv_files = [f for f in z.namelist() if f.endswith(".csv")]
            if not csv_files:
                raise ValueError("No CSV found inside ZIP file")
            with z.open(csv_files[0]) as f:
                df = pd.read_csv(f)
    else:
        df = pd.read_csv(file)

    # Normalize common column names
    rename_map = {}
    for dcol in ["Date", "order_date", "Order Date", "date", "Order Date Time", "purchase_date"]:
        if dcol in df.columns:
            rename_map[dcol] = "order_date"
            df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
            break
    for oc in ["Order ID", "order_id", "OrderId", "OrderID"]:
        if oc in df.columns: rename_map[oc] = "order_id"; break
    for cc in ["buyer-email", "customer_id", "Buyer Email", "buyer_email"]:
        if cc in df.columns: rename_map[cc] = "customer_id"; break
    for pc in ["SKU", "product_id", "ASIN", "Item ID"]:
        if pc in df.columns: rename_map[pc] = "product_id"; break
    for pn in ["item-name", "product_name", "Style", "Item Name"]:
        if pn in df.columns: rename_map[pn] = "product_name"; break
    for cat in ["Category", "category", "Product Category"]:
        if cat in df.columns: rename_map[cat] = "category"; break
    for qc in ["Qty", "quantity", "Quantity"]:
        if qc in df.columns: rename_map[qc] = "quantity"; break
    for amt in ["Amount", "amount", "Item Total", "item_total", "Unit Price", "Price", "price", "Total"]:
        if amt in df.columns: rename_map[amt] = "amount"; break
    for pm in ["Payment Instrument Type", "payment_method", "payment", "Payment Method"]:
        if pm in df.columns: rename_map[pm] = "payment_method"; break

    df = df.rename(columns=rename_map)
    return df

# -------------------------------
# Clean data
# -------------------------------
def clean(df):
    if "Status" in df.columns:
        bad = df["Status"].str.contains("Cancel|Refund|Return", case=False, na=False)
        df = df[~bad].copy()
    df["quantity"] = pd.to_numeric(df.get("quantity", 1), errors="coerce").fillna(1).astype(int)
    df["amount"] = pd.to_numeric(df.get("amount", 0.0), errors="coerce").fillna(0.0)
    df["revenue"] = df["amount"]
    if "order_date" in df.columns:
        df = df[df["order_date"].notna()].copy()
    for c in ["customer_id", "product_id", "order_id", "category", "product_name", "payment_method"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df

# -------------------------------
# KPIs, Charts, Views (same as before)
# -------------------------------
def kpis(f):
    orders = f["order_id"].nunique() if "order_id" in f else len(f)
    custs = f["customer_id"].nunique() if "customer_id" in f else np.nan
    rev = float(f["revenue"].sum())
    aov = f.groupby("order_id")["revenue"].sum().mean() if "order_id" in f else np.nan
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Revenue", f"{rev:,.0f}")
    c2.metric("Orders", f"{orders:,}")
    c3.metric("Customers", f"{custs:,}" if pd.notna(custs) else "—")
    c4.metric("AOV", f"{aov:,.0f}" if pd.notna(aov) else "—")

def trend(f):
    if "order_date" not in f: return
    ts = f.set_index("order_date")["revenue"].resample("M").sum()
    if ts.empty: st.info("No data for this period."); return
    fig, ax = plt.subplots()
    ts.plot(ax=ax); ax.set_ylabel("Revenue"); ax.yaxis.set_major_formatter(fmtM)
    st.pyplot(fig)

def category_view(f):
    if "category" not in f: return
    g = f.groupby("category")["revenue"].sum().sort_values(ascending=False)
    if g.empty: st.info("No data by category."); return
    fig, ax = plt.subplots()
    g.plot(kind="bar", ax=ax); ax.set_ylabel("Revenue"); ax.yaxis.set_major_formatter(fmtM)
    st.pyplot(fig)

def top_products(f, n=15):
    cols = [c for c in ["product_id","product_name"] if c in f.columns]
    if not cols: return
    g = f.groupby(cols)["revenue"].sum().sort_values(ascending=False).head(n).reset_index()
    st.dataframe(g, use_container_width=True)

def repeat_rate(f):
    if "order_id" not in f or "customer_id" not in f:
        st.metric("Repeat purchase rate","—"); return
    o = f.groupby("order_id", as_index=False).agg(customer_id=("customer_id","first"))
    r = o["customer_id"].duplicated().mean()
    st.metric("Repeat purchase rate", f"{r:.1%}")

def rfm(f):
    if "customer_id" not in f or "order_date" not in f:
        st.info("RFM needs customer_id and order_date."); return
    snap = f["order_date"].max() + pd.Timedelta(days=1)
    g = f.groupby("customer_id").agg(
        Recency=("order_date", lambda x: (snap - x.max()).days),
        Frequency=("order_id","nunique") if "order_id" in f else ("order_date","count"),
        Monetary=("revenue","sum")
    )
    g = g[g["Monetary"]>0]
    g["R"] = pd.qcut(g["Recency"],5,labels=[5,4,3,2,1]).astype(int)
    g["F"] = pd.qcut(g["Frequency"].rank(method="first"),5,labels=[1,2,3,4,5]).astype(int)
    g["M"] = pd.qcut(g["Monetary"],5,labels=[1,2,3,4,5]).astype(int)
    g["Score"] = g[["R","F","M"]].sum(axis=1)
    st.dataframe(g.sort_values("Score",ascending=False).head(25), use_container_width=True)

def timegrid(f):
    if "order_date" not in f: return
    f = f.copy()
    f["weekday"] = f["order_date"].dt.day_name()
    f["hour"] = f["order_date"].dt.hour
    pvt = f.pivot_table(index="weekday", columns="hour", values="revenue", aggfunc="sum").fillna(0)
    st.dataframe(pvt, use_container_width=True)

# -------------------------------
# Always auto-load default dataset
# -------------------------------
default_path = r"Streamlit/Amazon_Sale_Report.zip"

if os.path.exists(default_path):
    df = load_csv(default_path)
else:
    st.warning("⚠ Default dataset not found. Using a generated sample dataset.")
    dates = pd.date_range("2023-01-01", periods=200, freq="D")
    df = pd.DataFrame({
        "order_date": dates,
        "order_id": range(1, 201),
        "customer_id": np.random.choice([f"C{i}" for i in range(1, 21)], size=200),
        "product_id": np.random.choice([f"P{i}" for i in range(1, 11)], size=200),
        "product_name": np.random.choice(["Shirt","Shoes","Watch","Phone","Bag"], size=200),
        "category": np.random.choice(["Fashion","Electronics","Accessories"], size=200),
        "quantity": np.random.randint(1, 5, size=200),
        "amount": np.random.randint(100, 1000, size=200)
    })

df = clean(df)

# -------------------------------
# Main layout & tabs
# -------------------------------
st.title("Amazon Sales Analysis")

if "order_date" not in df.columns:
    st.error("❌ No 'order_date' column found in dataset.")
    st.write("Available columns:", df.columns.tolist())
    st.stop()

mind, maxd = df["order_date"].min(), df["order_date"].max()
if pd.isna(mind) or pd.isna(maxd):
    st.warning("Invalid order_date values."); st.stop()

d1,d2 = st.slider("Date range", min_value=mind.to_pydatetime(), max_value=maxd.to_pydatetime(),
                  value=(mind.to_pydatetime(), maxd.to_pydatetime()))
f = df[(df["order_date"]>=pd.to_datetime(d1)) & (df["order_date"]<=pd.to_datetime(d2))].copy()

tabs = st.tabs(["Overview","Trends","Categories & Products","Customers & RFM","Time Grid","Downloads","About"])

with tabs[0]:
    kpis(f); repeat_rate(f)
with tabs[1]:
    st.subheader("Monthly Revenue Trend"); trend(f)
with tabs[2]:
    st.subheader("Revenue by Category"); category_view(f)
    st.subheader("Top Products"); top_products(f,15)
with tabs[3]:
    st.subheader("RFM Segmentation"); rfm(f)
with tabs[4]:
    st.subheader("Weekday × Hour Revenue"); timegrid(f)
with tabs[5]:
    c1,c2 = st.columns(2)
    with c1:
        st.download_button("Download filtered data (CSV)", data=f.to_csv(index=False).encode("utf-8"), file_name="amazon_filtered.csv")
    if "customer_id" in f.columns:
        cust = f.groupby("customer_id").agg(orders=("order_id","nunique"), revenue=("revenue","sum"))
        with c2:
            st.download_button("Download customer aggregates (CSV)", data=cust.reset_index().to_csv(index=False).encode("utf-8"), file_name="customers_agg.csv")
with tabs[6]:
    st.markdown("Single-CSV analysis for Amazon Seller data: cleaning, KPIs, trends, category/product insights, repeat behavior, RFM, and time grid.")
