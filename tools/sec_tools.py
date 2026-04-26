from __future__ import annotations
import json
import httpx
from langchain_core.tools import tool
from config import settings

_EDGAR = "https://data.sec.gov"
_EFTS = "https://efts.sec.gov"
_HEADERS = {"User-Agent": settings.edgar_user_agent, "Accept-Encoding": "gzip, deflate"}


@tool
async def search_sec_filings(company_name: str, form_type: str = "10-K") -> str:
    """
    Search SEC EDGAR for company filings by company name.
    form_type options: '10-K' (annual), '10-Q' (quarterly), '8-K' (material events),
    'DEF 14A' (proxy/compensation), 'S-1' (IPO), 'SC 13G' (ownership disclosures).
    Returns the most recent matching filings with dates and descriptions.
    """
    async with httpx.AsyncClient(timeout=30) as client:
        params = {
            "q": f'"{company_name}"',
            "forms": form_type,
            "dateRange": "custom",
            "startdt": "2021-01-01",
        }
        resp = await client.get(
            f"{_EFTS}/LATEST/search-index", params=params, headers=_HEADERS
        )

        if resp.status_code != 200:
            return json.dumps({"error": f"EDGAR search returned {resp.status_code}"})

        data = resp.json()
        hits = data.get("hits", {}).get("hits", [])

        if not hits:
            return json.dumps({
                "message": f"No {form_type} filings found for '{company_name}' in EDGAR",
                "filings": [],
            })

        filings = []
        for hit in hits[:6]:
            src = hit.get("_source", {})
            filings.append({
                "entity_name": src.get("entity_name"),
                "form_type": src.get("file_type"),
                "filed_date": src.get("file_date"),
                "period_of_report": src.get("period_of_report"),
                "accession_number": src.get("accession_no", "").replace("-", ""),
                "description": (src.get("file_description") or "")[:200],
            })

        return json.dumps({"filings": filings, "total_found": len(hits)}, indent=2)


@tool
async def get_company_facts(ticker: str) -> str:
    """
    Get structured financial facts directly from SEC EDGAR XBRL data.
    Returns key financial metrics as reported in SEC filings — revenue, net income,
    total assets, debt, cash flow, etc. More reliable than estimated data.
    """
    async with httpx.AsyncClient(timeout=60) as client:
        # Step 1: Resolve ticker → CIK
        resp = await client.get(
            f"{_EDGAR}/files/company_tickers.json", headers=_HEADERS
        )
        if resp.status_code != 200:
            return json.dumps({"error": "Could not reach EDGAR company tickers endpoint"})

        ticker_upper = ticker.upper()
        cik = None
        company_name = ticker

        for entry in resp.json().values():
            if entry.get("ticker", "").upper() == ticker_upper:
                cik = str(entry["cik_str"]).zfill(10)
                company_name = entry.get("title", ticker)
                break

        if not cik:
            return json.dumps({"error": f"Ticker '{ticker}' not found in EDGAR. Company may be private."})

        # Step 2: Fetch XBRL company facts
        resp = await client.get(
            f"{_EDGAR}/api/xbrl/companyfacts/CIK{cik}.json", headers=_HEADERS
        )
        if resp.status_code != 200:
            return json.dumps({"error": f"EDGAR facts endpoint returned {resp.status_code} for CIK {cik}"})

        us_gaap = resp.json().get("facts", {}).get("us-gaap", {})

        def latest_annual(concept: str) -> dict | None:
            values = us_gaap.get(concept, {}).get("units", {}).get("USD", [])
            annual = [v for v in values if v.get("form") == "10-K"]
            if not annual:
                return None
            annual.sort(key=lambda x: x.get("end", ""), reverse=True)
            return {"value": annual[0].get("val"), "period_end": annual[0].get("end")}

        def last_n_annual(concept: str, n: int = 4) -> list[dict]:
            values = us_gaap.get(concept, {}).get("units", {}).get("USD", [])
            annual = [v for v in values if v.get("form") == "10-K"]
            annual.sort(key=lambda x: x.get("end", ""), reverse=True)
            return [{"value": v.get("val"), "period_end": v.get("end")} for v in annual[:n]]

        concepts = {
            "Revenue": "Revenues",
            "NetIncomeLoss": "NetIncomeLoss",
            "OperatingIncomeLoss": "OperatingIncomeLoss",
            "Assets": "Assets",
            "Liabilities": "Liabilities",
            "StockholdersEquity": "StockholdersEquity",
            "Cash": "CashAndCashEquivalentsAtCarryingValue",
            "LongTermDebt": "LongTermDebt",
            "OperatingCashFlow": "NetCashProvidedByUsedInOperatingActivities",
            "CapEx": "PaymentsToAcquirePropertyPlantAndEquipment",
            "RAndD": "ResearchAndDevelopmentExpense",
        }

        extracted = {}
        for label, concept in concepts.items():
            val = latest_annual(concept)
            if val:
                extracted[label] = val

        revenue_trend = last_n_annual("Revenues", 4)

        return json.dumps({
            "company": company_name,
            "cik": cik,
            "ticker": ticker,
            "latest_annual_financials": extracted,
            "revenue_trend_4yr": revenue_trend,
        }, indent=2)
