from __future__ import annotations
import json
import asyncio
from langchain_core.tools import tool


@tool
async def get_company_financials(ticker: str) -> str:
    """
    Get comprehensive financial data for a public company by ticker symbol.
    Returns income statement metrics, balance sheet, cash flow, and key valuation ratios.
    Use this first for any public company financial analysis.
    """
    def _fetch() -> dict:
        import yfinance as yf
        t = yf.Ticker(ticker)
        info = t.info

        result: dict = {
            "company_name": info.get("longName", ticker),
            "ticker": ticker,
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "description": (info.get("longBusinessSummary") or "")[:600],
            "website": info.get("website"),
            "country": info.get("country"),
            "employees": info.get("fullTimeEmployees"),
            # Scale
            "market_cap": info.get("marketCap"),
            "enterprise_value": info.get("enterpriseValue"),
            # Income
            "revenue_ttm": info.get("totalRevenue"),
            "gross_profit_ttm": info.get("grossProfits"),
            "ebitda_ttm": info.get("ebitda"),
            "net_income_ttm": info.get("netIncomeToCommon"),
            "eps_ttm": info.get("trailingEps"),
            # Balance sheet
            "total_cash": info.get("totalCash"),
            "total_debt": info.get("totalDebt"),
            "book_value_per_share": info.get("bookValue"),
            # Cash flow
            "free_cash_flow": info.get("freeCashflow"),
            "operating_cash_flow": info.get("operatingCashflow"),
            # Valuation ratios
            "ratios": {
                "pe_trailing": info.get("trailingPE"),
                "pe_forward": info.get("forwardPE"),
                "peg_ratio": info.get("pegRatio"),
                "price_to_sales": info.get("priceToSalesTrailing12Months"),
                "price_to_book": info.get("priceToBook"),
                "ev_to_ebitda": info.get("enterpriseToEbitda"),
                "ev_to_revenue": info.get("enterpriseToRevenue"),
            },
            # Profitability
            "margins": {
                "gross_margin": info.get("grossMargins"),
                "operating_margin": info.get("operatingMargins"),
                "net_margin": info.get("profitMargins"),
                "roe": info.get("returnOnEquity"),
                "roa": info.get("returnOnAssets"),
            },
            # Growth
            "growth": {
                "revenue_growth_yoy": info.get("revenueGrowth"),
                "earnings_growth_yoy": info.get("earningsGrowth"),
                "revenue_growth_quarterly": info.get("revenueQuarterlyGrowth"),
                "earnings_growth_quarterly": info.get("earningsQuarterlyGrowth"),
            },
            # Leverage
            "leverage": {
                "debt_to_equity": info.get("debtToEquity"),
                "current_ratio": info.get("currentRatio"),
                "quick_ratio": info.get("quickRatio"),
            },
        }

        # Annual revenue for trend analysis
        try:
            financials = t.financials
            if not financials.empty and "Total Revenue" in financials.index:
                result["annual_revenue_trend"] = {
                    str(col.year): int(financials.loc["Total Revenue", col])
                    for col in financials.columns
                    if not financials.loc["Total Revenue", col] != financials.loc["Total Revenue", col]  # not NaN
                }
        except Exception:
            pass

        return result

    try:
        data = await asyncio.to_thread(_fetch)
    except Exception as e:
        data = {"error": str(e), "ticker": ticker}

    return json.dumps(data, indent=2, default=str)


@tool
async def get_stock_price_history(ticker: str, period: str = "2y") -> str:
    """
    Get stock price history and technical indicators for a company.
    Period options: 1mo, 3mo, 6mo, 1y, 2y, 5y.
    Returns current price, 52-week range, moving averages, and return metrics.
    """
    def _fetch() -> dict:
        import yfinance as yf
        t = yf.Ticker(ticker)
        hist = t.history(period=period)

        if hist.empty:
            return {"error": f"No price data found for {ticker}"}

        close = hist["Close"]
        current_price = float(close.iloc[-1])
        start_price = float(close.iloc[0])
        ma_50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
        ma_200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None

        return {
            "ticker": ticker,
            "period": period,
            "current_price": current_price,
            "price_at_period_start": start_price,
            "period_return_pct": round((current_price / start_price - 1) * 100, 2),
            "52w_high": float(hist["High"].max()),
            "52w_low": float(hist["Low"].min()),
            "ma_50": ma_50,
            "ma_200": ma_200,
            "avg_daily_volume": float(hist["Volume"].mean()),
            "trend_vs_ma50": "above" if ma_50 and current_price > ma_50 else "below",
            "trend_vs_ma200": "above" if ma_200 and current_price > ma_200 else "below",
            "volatility_annualized_pct": round(
                float(close.pct_change().std() * (252 ** 0.5) * 100), 2
            ),
        }

    try:
        data = await asyncio.to_thread(_fetch)
    except Exception as e:
        data = {"error": str(e), "ticker": ticker}

    return json.dumps(data, indent=2, default=str)


@tool
async def get_peer_companies(ticker: str) -> str:
    """
    Get sector and industry information to identify comparable peer companies.
    Returns company metadata useful for building a comparable companies set.
    """
    def _fetch() -> dict:
        import yfinance as yf
        t = yf.Ticker(ticker)
        info = t.info
        return {
            "company": info.get("longName", ticker),
            "ticker": ticker,
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "market_cap": info.get("marketCap"),
            "country": info.get("country"),
            "note": (
                "Use search_web to find comparable public companies in the same "
                "sector/industry for trading multiples analysis. Search for "
                f"'{info.get('industry', '')} public companies list market cap'"
            ),
        }

    try:
        data = await asyncio.to_thread(_fetch)
    except Exception as e:
        data = {"error": str(e), "ticker": ticker}

    return json.dumps(data, indent=2, default=str)
