# app.py — Amazon Sales Analysis (Streamlit)
# ------------------------------------------------------------
# What this app does:
# - Loads a single CSV (Amazon Seller–style) and auto-maps common column names
# - Lets you override mappings from the sidebar
# - Cleans data (drops refunds/returns/cancellations, coerces types)
# - Shows KPIs, monthly trend, revenue by category, top products
# - Computes RFM segmentation and a weekday×hour revenue grid
# - Supports Dark Mode (default) and formats large y-axes as "M" (millions)
# ------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# -------------------------------
# App/page config
# -------------------------------
st.set_page_config(page_title="Amazon Sales Analysis", layout="wide")

# -------------------------------
# Theme: Dark/Light toggle + CSS
# -------------------------------
# Sidebar toggle to switch themes on the fly
dark = st.sidebar.toggle("Dark mode", value=True)

# Base dark theme overrides (backgrounds + text)
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

# Apply chosen theme CSS
st.markdown(DARK_CSS if dark else LIGHT_CSS, unsafe_allow_html=True)

# Make Matplotlib figures match the theme
plt.style.use("dark_background" if dark else "default")

# Common y-axis formatter: show values in millions (e.g., 35.2M)
fmtM = mticker.FuncFormatter(lambda x, _: f'{x * 1e-6:.1f}M')

# Extra: Improve widget (selectbox/input) readability in dark theme
WIDGET_CSS = """
<style>
/* Sidebar labels (e.g., 'order_id', 'order_date') */
.stSelectbox label, .stTextInput label {
    color: #e5e7eb !important;
    font-weight: 600;
}
/* Text inside select/input widgets */
.stSelectbox div[data-baseweb="select"] span,
.stTextInput input {
    color: #f9fafb !important;
    background-color: #111827 !important;
}
/* Dropdown option menu */
.stSelectbox [data-baseweb="menu"] div {
    color: #f9fafb !important;
    background-color: #1f2937 !important;
}
</style>
"""
st.markdown(WIDGET_CSS, unsafe_allow_html=True)

# -------------------------------
# Data loading + column auto-map
# -------------------------------
@st.cache_data
def load_csv(file):
    """
    Read CSV and normalize common Amazon column names to:
    order_date, order_id, customer_id, product_id, product_name,
    category, quantity, amount, payment_method
    """
    # Read from uploaded file-like or path string
    df = pd.read_csv(file) if hasattr(file, "read") else pd.read_csv(file)

    # Order date (try multiple known variants)
    for dcol in ["Date", "order_date", "Order Date", "date"]:
        if dcol in df.columns:
            # Some Amazon 'Date' columns are in %m-%d-%y; fallback to generic parse
            if dcol == "Date":
                df[dcol] = pd.to_datetime(df[dcol], errors="coerce", format="%m-%d-%y")
            else:
                df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
            df = df.rename(columns={dcol: "order_date"})
            break

    # Order ID
    for oc in ["Order ID", "order_id", "OrderId", "OrderID"]:
        if oc in df.columns:
            df = df.rename(columns={oc: "order_id"})
            break

    # Customer ID (often buyer email)
    for cc in ["buyer-email", "customer_id", "Buyer Email", "buyer_email"]:
        if cc in df.columns:
            df = df.rename(columns={cc: "customer_id"})
            break

    # Product ID (ASIN or SKU)
    for pc in ["SKU", "product_id", "ASIN", "Item ID"]:
        if pc in df.columns:
            df = df.rename(columns={pc: "product_id"})
            break

    # Product Name
    for pn in ["item-name", "product_name", "Style", "Item Name"]:
        if pn in df.columns:
            df = df.rename(columns={pn: "product_name"})
            break

    # Category
    for cat in ["Category", "category", "Product Category"]:
        if cat in df.columns:
            df = df.rename(columns={cat: "category"})
            break

    # Quantity
    for qc in ["Qty", "quantity", "Quantity"]:
        if qc in df.columns:
            df = df.rename(columns={qc: "quantity"})
            break

    # Amount (revenue)
    for amt in ["Amount", "amount", "Item Total", "item_total", "Unit Price", "Price", "price", "Total"]:
        if amt in df.columns:
            df = df.rename(columns={amt: "amount"})
            break

    # Payment method
    for pm in ["Payment Instrument Type", "payment_method", "payment", "Payment Method"]:
        if pm in df.columns:
            df = df.rename(columns={pm: "payment_method"})
            break

    return df

# -------------------------------
# Cleaning
# -------------------------------
def clean(df):
    """
    - Remove cancelled/refunded/returned rows (if Status available)
    - Coerce quantity/amount to numeric
    - Create 'revenue' column
    - Ensure key id/text columns are strings
    - Drop rows with null order_date
    """
    if "Status" in df.columns:
        bad = df["Status"].str.contains("Cancel|Refund|Return", case=False, na=False)
        df = df[~bad].copy()

    if "quantity" in df.columns:
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(1).astype(int)
    else:
        df["quantity"] = 1

    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    else:
        df["amount"] = 0.0

    # For this report, 'amount' is already the per-line revenue
    df["revenue"] = df["amount"]

    if "order_date" in df.columns:
        df = df[df["order_date"].notna()].copy()

    for c in ["customer_id", "product_id", "order_id", "category", "product_name", "payment_method"]:
        if c in df.columns:
            df[c] = df[c].astype(str)

    return df

# -------------------------------
# KPIs
# -------------------------------
def kpis(f):
    """Show main KPIs: revenue, orders, customers, AOV."""
    orders = f["order_id"].nunique() if "order_id" in f.columns else len(f)
    custs = f["customer_id"].nunique() if "customer_id" in f.columns else np.nan
    rev = float(f["revenue"].sum())
    aov = f.groupby("order_id")["revenue"].sum().mean() if "order_id" in f.columns else np.nan

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Revenue", f"{rev:,.0f}")
    c2.metric("Orders", f"{orders:,}")
    c3.metric("Customers", f"{custs:,}" if pd.notna(custs) else "—")
    c4.metric("AOV", f"{aov:,.0f}" if pd.notna(aov) else "—")

# -------------------------------
# Charts
# -------------------------------
def trend(f):
    """Monthly revenue time series with 'M' y-axis labels."""
    if "order_date" not in f.columns:
        return
    ts = f.set_index("order_date")["revenue"].resample("M").sum()
    if ts.empty:
        st.info("No data for the selected range.")
        return
    fig, ax = plt.subplots()
    ts.plot(ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Revenue")
    ax.yaxis.set_major_formatter(fmtM)
    st.pyplot(fig)

def category_view(f):
    """Bar chart: revenue by category."""
    if "category" not in f.columns:
        st.info("No category column found.")
        return
    g = f.groupby("category")["revenue"].sum().sort_values(ascending=False)
    if g.empty:
        st.info("No data for selected filters.")
        return
    fig, ax = plt.subplots()
    g.plot(kind="bar", ax=ax)
    ax.set_xlabel("Category")
    ax.set_ylabel("Revenue")
    ax.yaxis.set_major_formatter(fmtM)
    st.pyplot(fig)

def top_products(f, n=15):
    """Table: top-N products by revenue."""
    cols = [c for c in ["product_id", "product_name"] if c in f.columns]
    if not cols:
        st.info("No product columns.")
        return
    g = (
        f.groupby(cols)["revenue"]
         .sum()
         .sort_values(ascending=False)
         .head(n)
         .reset_index()
    )
    st.dataframe(g, use_container_width=True)

def repeat_rate(f):
    """Metric: share of orders from returning customers."""
    if "order_id" not in f.columns or "customer_id" not in f.columns:
        st.metric("Repeat purchase rate", "—")
        return
    # Collapse to one row per order, then mark returning customers
    o = f.groupby("order_id", as_index=False).agg(customer_id=("customer_id", "first"))
    r = o["customer_id"].duplicated().mean()
    st.metric("Repeat purchase rate", f"{r:.1%}")

def rfm(f):
    """
    RFM segmentation:
    - Recency: days since last order (lower is better)
    - Frequency: unique orders count
    - Monetary: total revenue
    Each scored into quintiles (1–5), then summed as R+F+M.
    """
    if "customer_id" not in f.columns or "order_date" not in f.columns:
        st.info("RFM needs customer_id and order_date.")
        return

    snap = f["order_date"].max() + pd.Timedelta(days=1)
    g = f.groupby("customer_id").agg(
        Recency=("order_date", lambda x: (snap - x.max()).days),
        Frequency=("order_id", "nunique") if "order_id" in f.columns else ("order_date", "count"),
        Monetary=("revenue", "sum"),
    )

    g = g[g["Monetary"] > 0]

    # Quintile scoring; for Frequency we rank first to avoid ties breaking qcut
    g["R"] = pd.qcut(g["Recency"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    g["F"] = pd.qcut(g["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    g["M"] = pd.qcut(g["Monetary"], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    g["Score"] = g[["R", "F", "M"]].sum(axis=1)

    st.dataframe(g.sort_values("Score", ascending=False).head(25), use_container_width=True)

def timegrid(f):
    """Weekday × hour pivot of total revenue (table)."""
    if "order_date" not in f.columns:
        return
    f = f.copy()
    f["weekday"] = f["order_date"].dt.day_name()
    f["hour"] = f["order_date"].dt.hour
    pvt = f.pivot_table(index="weekday", columns="hour", values="revenue", aggfunc="sum").fillna(0)
    st.dataframe(pvt, use_container_width=True)

# -------------------------------
# Sidebar: data source selection
# -------------------------------
st.sidebar.header("Data")
mode = st.sidebar.selectbox("Load mode", ["Upload CSV", "Use file path"])
file = None
if mode == "Upload CSV":
    file = st.sidebar.file_uploader("Upload Amazon Sale Report.csv", type=["csv"])
else:
    p = st.sidebar.text_input("Path", value="data/Amazon_Sale_Report.csv")
    file = p

# Load data (cached)
df = load_csv(file) if file else pd.DataFrame()
if df.empty:
    st.warning("Load the CSV to proceed.")
    st.stop()

# -------------------------------
# Sidebar: optional column mapping
# -------------------------------
st.sidebar.subheader("Column mapping (optional)")

# Each selectbox defaults to the existing column if it already matches,
# otherwise picks the first column alphabetically.
col_order_id = st.sidebar.selectbox(
    "order_id", options=sorted(df.columns),
    index=sorted(df.columns).index("order_id") if "order_id" in df.columns else 0
)
col_order_date = st.sidebar.selectbox(
    "order_date", options=sorted(df.columns),
    index=sorted(df.columns).index("order_date") if "order_date" in df.columns else 0
)
col_customer = st.sidebar.selectbox(
    "customer_id", options=sorted(df.columns),
    index=sorted(df.columns).index("customer_id") if "customer_id" in df.columns else 0
)
col_product_id = st.sidebar.selectbox(
    "product_id", options=sorted(df.columns),
    index=sorted(df.columns).index("product_id") if "product_id" in df.columns else 0
)
col_product_name = st.sidebar.selectbox(
    "product_name", options=sorted(df.columns),
    index=sorted(df.columns).index("product_name") if "product_name" in df.columns else 0
)
col_category = st.sidebar.selectbox(
    "category", options=sorted(df.columns),
    index=sorted(df.columns).index("category") if "category" in df.columns else 0
)
col_quantity = st.sidebar.selectbox(
    "quantity", options=sorted(df.columns),
    index=sorted(df.columns).index("quantity") if "quantity" in df.columns else 0
)
col_amount = st.sidebar.selectbox(
    "amount", options=sorted(df.columns),
    index=sorted(df.columns).index("amount") if "amount" in df.columns else 0
)
col_payment = (
    st.sidebar.selectbox(
        "payment_method", options=sorted(df.columns),
        index=sorted(df.columns).index("payment_method") if "payment_method" in df.columns else 0
    )
    if "payment_method" in df.columns else None
)

# Apply the user-selected mappings
df = df.rename(columns={
    col_order_id: "order_id",
    col_order_date: "order_date",
    col_customer: "customer_id",
    col_product_id: "product_id",
    col_product_name: "product_name",
    col_category: "category",
    col_quantity: "quantity",
    col_amount: "amount",
    **({col_payment: "payment_method"} if col_payment else {})
})

# Clean the dataset
df = clean(df)

# -------------------------------
# Main layout & tabs
# -------------------------------
st.title("Amazon Sales Analysis")

# Guard against invalid dates after cleaning
mind, maxd = df["order_date"].min(), df["order_date"].max()
if pd.isna(mind) or pd.isna(maxd):
    st.warning("Invalid order_date.")
    st.stop()

# Date filter slider
d1, d2 = st.slider(
    "Date range",
    min_value=mind.to_pydatetime(),
    max_value=maxd.to_pydatetime(),
    value=(mind.to_pydatetime(), maxd.to_pydatetime()),
)
f = df[(df["order_date"] >= pd.to_datetime(d1)) & (df["order_date"] <= pd.to_datetime(d2))].copy()

# Tabs
tabs = st.tabs(["Overview", "Trends", "Categories & Products", "Customers & RFM", "Time Grid", "Downloads", "About"])

with tabs[0]:
    kpis(f)
    repeat_rate(f)

with tabs[1]:
    st.subheader("Monthly Revenue Trend")
    trend(f)

with tabs[2]:
    st.subheader("Revenue by Category")
    category_view(f)
    st.subheader("Top Products")
    top_products(f, n=15)

with tabs[3]:
    st.subheader("RFM Segmentation")
    rfm(f)

with tabs[4]:
    st.subheader("Weekday × Hour Revenue")
    timegrid(f)

with tabs[5]:
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download filtered data (CSV)",
            data=f.to_csv(index=False).encode("utf-8"),
            file_name="amazon_filtered.csv",
        )
    if "customer_id" in f.columns:
        cust = f.groupby("customer_id").agg(
            orders=("order_id", "nunique"),
            revenue=("revenue", "sum"),
        )
        with c2:
            st.download_button(
                "Download customer aggregates (CSV)",
                data=cust.reset_index().to_csv(index=False).encode("utf-8"),
                file_name="customers_agg.csv",
            )

with tabs[6]:
    st.markdown(
        "Single-CSV analysis for Amazon Seller data: cleaning, KPIs, trends, "
        "category/product insights, repeat behavior, RFM, and time grid."
    )
