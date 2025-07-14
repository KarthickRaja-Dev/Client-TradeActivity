from flask import Flask, request, jsonify
import pandas as pd
from datetime import timedelta

app = Flask(__name__)

def loadCsv(csv="data.csv"):
    df = pd.read_csv(csv, parse_dates=["trade_date"])
    df["buy_qty"] = df["buy_qty"].fillna(0)
    df["buy_price"] = df["buy_price"].fillna(0)
    df["sell_qty"] = df["sell_qty"].fillna(0)
    df["sell_price"] = df["sell_price"].fillna(0)
    df["buy_value"] = df["buy_qty"] * df["buy_price"]
    df["sell_value"] = df["sell_qty"] * df["sell_price"]
    df = df[(df["buy_qty"] > 0) | (df["sell_qty"] > 0)]
    df = df.sort_values("trade_date").reset_index(drop=True)
    return df.drop_duplicates()

@app.route("/", methods=["GET"])
def getTradeSummary():
    df = loadCsv(request.files.get("file"))
    summary = df.groupby("client_id").agg(
        total_buy_qty=("buy_qty", "sum"),
        total_buy_value=("buy_value", "sum"),
        total_sell_qty=("sell_qty", "sum"),
        total_sell_value=("sell_value", "sum"),
        trade_days=("trade_date", pd.Series.nunique)
    )
    top_scrips = (
        df.groupby(["client_id", "scrip_name"])
        .size()
        .reset_index(name="trade_count")
        .sort_values(["trade_count"], ascending=[False])
        .drop_duplicates("client_id")
        .rename(columns={"scrip_name": "top_traded_scrip"})
    )[["client_id", "top_traded_scrip"]]
    summary = summary.merge(top_scrips, on="client_id", how="left")
    return jsonify(summary.to_dict(orient="records"))

@app.route("/get_daily_summary", methods=["GET"])
def getDailySummary():
    df = loadCsv(request.files.get("file"))
    daily_summary = df.groupby("trade_date").agg(
        total_buy_qty=("buy_qty", "sum"),
        total_buy_value=("buy_value", "sum"),
        total_sell_qty=("sell_qty", "sum"),
        total_sell_value=("sell_value", "sum"),
        unique_clients=("client_id", pd.Series.nunique)
    ).reset_index()
    client_daily = df.groupby(["trade_date", "client_id"]).agg(
        total_trade_value=("buy_value", "sum")
    ).reset_index()
    client_daily["total_trade_value"] += df.groupby(["trade_date", "client_id"])["sell_value"].sum().values
    top_clients = (
        client_daily.sort_values(["trade_date", "total_trade_value"], ascending=[True, False])
        .groupby("trade_date")
        .head(5)
        .groupby("trade_date")["client_id"]
        .apply(list)
        .reset_index()
        .rename(columns={"client_id": "top_5_clients"})
    )
    df["total_qty"] = df["buy_qty"] + df["sell_qty"]
    top_scrips = (
        df.groupby(["trade_date", "scrip_name"])["total_qty"].sum().reset_index()
        .sort_values(["trade_date", "total_qty"], ascending=[True, False])
        .groupby("trade_date")
        .head(5)
        .groupby("trade_date")["scrip_name"]
        .apply(list)
        .reset_index()
        .rename(columns={"scrip_name": "top_5_scrips"})
    )
    daily_summary = daily_summary.merge(top_clients, on="trade_date", how="left")
    daily_summary = daily_summary.merge(top_scrips, on="trade_date", how="left")
    return jsonify(daily_summary.to_dict(orient="records"))

@app.route("/get_management_report", methods=["GET"])
def getManagementReport():
    df = loadCsv(request.files.get("file"))
    client_net = df.groupby("client_id").agg(total_trade_value=("buy_value", "sum"))
    client_net["total_trade_value"] += df.groupby("client_id")["sell_value"].sum()
    top_clients = client_net.sort_values("total_trade_value", ascending=False).head(10).reset_index()
    df["total_qty"] = df["buy_qty"] + df["sell_qty"]
    df["total_value"] = df["buy_value"] + df["sell_value"]
    top_scrips_qty = df.groupby("scrip_name")["total_qty"].sum().sort_values(ascending=False).head(10).reset_index()
    top_scrips_value = df.groupby("scrip_name")["total_value"].sum().sort_values(ascending=False).head(10).reset_index()
    buy_stats = df[df["buy_value"] > 0]["buy_value"].describe().to_dict()
    sell_stats = df[df["sell_value"] > 0]["sell_value"].describe().to_dict()
    df["week"] = df["trade_date"] - pd.to_timedelta(df["trade_date"].dt.dayofweek, unit='D')
    weekly_trades = df.groupby(["client_id", "week"]).size().reset_index(name="trades")
    client_avg = weekly_trades.groupby("client_id")["trades"].mean().reset_index(name="avg_weekly_trades")
    def classify(trades):
        if trades >= 5:
            return "Active"
        elif trades >= 1:
            return "Moderate"
        return "Dormant"
    client_avg["category"] = client_avg["avg_weekly_trades"].apply(classify)
    last_date = df["trade_date"].max()
    cutoff = last_date - timedelta(days=30)
    active_clients = df[df["trade_date"] >= cutoff]["client_id"].unique()
    client_avg.loc[~client_avg["client_id"].isin(active_clients), "category"] = "Dormant"
    return jsonify({
        "top_clients": top_clients.to_dict(orient="records"),
        "top_scrips_by_quantity": top_scrips_qty.to_dict(orient="records"),
        "top_scrips_by_value": top_scrips_value.to_dict(orient="records"),
        "buy_value_distribution": buy_stats,
        "sell_value_distribution": sell_stats,
        "client_activity_categorization": client_avg.to_dict(orient="records")
    })

@app.route("/get_anomalies", methods=["GET"])
def getAnomalies():
    df = loadCsv(request.files.get("file"))
    high_value_trades = df[
        (df["buy_value"] > 5000000) | (df["sell_value"] > 5000000)
    ]
    trade_counts = df.groupby(["client_id", "trade_date"]).size().reset_index(name="trade_count")
    high_freq_clients = trade_counts[trade_counts["trade_count"] > 20]
    return jsonify({
        "high_value_trades": high_value_trades.to_dict(orient="records"),
        "high_frequency_clients": high_freq_clients.to_dict(orient="records")
    })

if __name__ == "__main__":
    app.run(debug=True)
