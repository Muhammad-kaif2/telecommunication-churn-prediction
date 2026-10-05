"""
Customer Churn Prediction API (FastAPI)

Runtime dependencies sirf fastapi/pydantic hain: LightGBM model ke trees `artifacts/trees.json` mein
export hain aur yahan pure Python mein evaluate hote hain. Is wajah se Vercel par deploy halka rehta hai
(lightgbm / scikit-learn / pandas ki zaroorat nahi). Artifacts dobara banane ke liye: tools/build_artifacts.py
"""
import json
import math
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

BASE_DIR = Path(__file__).resolve().parent
ART = json.loads((BASE_DIR / "artifacts" / "preprocess.json").read_text())
TREES = json.loads((BASE_DIR / "artifacts" / "trees.json").read_text())

FEATURES = ART["features"]
SCALER = ART["scaler"]
STATES = set(ART["states"])
INS = ART["insights"]
DECISION_THRESHOLD = 0.5  # LightGBM .predict() ka default cut-off

app = FastAPI(title="Customer Churn Prediction")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Customer(BaseModel):
    state: str = Field(..., min_length=2, max_length=2, description="US state code, e.g. KS")
    account_length: int = Field(..., ge=1, le=500, description="Days since account opened")
    area_code: Literal[408, 415, 510]
    international_plan: Literal["yes", "no"]
    voice_mail_plan: Literal["yes", "no"]
    total_day_calls: int = Field(..., ge=0, le=500)
    total_day_charge: float = Field(..., ge=0, le=200)
    total_eve_calls: int = Field(..., ge=0, le=500)
    total_eve_charge: float = Field(..., ge=0, le=100)
    total_night_calls: int = Field(..., ge=0, le=500)
    total_night_charge: float = Field(..., ge=0, le=100)
    total_intl_charge: float = Field(..., ge=0, le=50)
    total_intl_calls: int = Field(..., ge=0, le=100)
    customer_service_calls: int = Field(..., ge=0, le=50)

    @field_validator("state")
    @classmethod
    def known_state(cls, v: str) -> str:
        v = v.upper()
        if v not in STATES:
            raise ValueError("Unknown state code")
        return v


class Insight(BaseModel):
    tone: Literal["risk", "good"]
    text: str


class Prediction(BaseModel):
    churn_probability: float
    will_churn: bool
    risk_level: Literal["Low", "Elevated", "High"]
    insights: list[Insight]


def _scale(col: str, value: float) -> float:
    s = SCALER[col]
    return (value - s["center"]) / s["scale"]


def build_row(c: Customer) -> list[float]:
    """Notebook ke training preprocessing ko dohrata hai; order = model.feature_name_."""
    csc = c.customer_service_calls
    csc_bin = "Low" if csc < 3 else ("Medium" if csc < 5 else "High")  # bins [0,3) [3,5) [5,inf)
    f = {
        "state": ART["state_enc"][c.state],
        "account_length": _scale("account length", c.account_length),
        "area_code": ART["area_enc"][str(c.area_code)],
        "international_plan": 1.0 if c.international_plan == "yes" else 0.0,
        "voice_mail_plan": 1.0 if c.voice_mail_plan == "yes" else 0.0,
        "total_day_calls": _scale("total day calls", c.total_day_calls),
        "total_day_charge": _scale("total day charge", c.total_day_charge),
        "total_eve_calls": _scale("total eve calls", c.total_eve_calls),
        "total_eve_charge": _scale("total eve charge", c.total_eve_charge),
        "total_night_calls": _scale("total night calls", c.total_night_calls),
        "total_night_charge": _scale("total night charge", c.total_night_charge),
        "total_intl_charge": _scale("total intl charge", c.total_intl_charge),
        "total_intl_calls_log": _scale("total_intl_calls_log", math.log1p(c.total_intl_calls)),
        "binned_customer_service_calls": ART["csc_enc"][csc_bin],
    }
    return [f[name] for name in FEATURES]


def _walk(node, row):
    while isinstance(node, list):  # [feature_index, threshold, left, right]
        node = node[2] if row[node[0]] <= node[1] else node[3]
    return node


def churn_probability(row: list[float]) -> float:
    raw = sum(_walk(t, row) for t in TREES)
    return 1.0 / (1.0 + math.exp(-raw))


def explain(c: Customer) -> list[Insight]:
    pct = lambda x: f"{x:.0%}"
    out: list[Insight] = []
    if c.customer_service_calls >= 4:
        out.append(Insight(tone="risk", text=f"{c.customer_service_calls} customer service calls: in the training data, "
                   f"{pct(INS['csc_4plus'])} of customers with 4 or more calls churned (overall {pct(INS['base_rate'])})."))
    if c.international_plan == "yes":
        out.append(Insight(tone="risk", text=f"International plan holders churned at {pct(INS['intl_plan_yes'])} "
                   f"versus {pct(INS['intl_plan_no'])} without the plan."))
    if c.total_day_charge >= INS["day_charge_q75"]:
        out.append(Insight(tone="risk", text=f"Day charge of ${c.total_day_charge:.2f} is in the top quarter "
                   f"(from ${INS['day_charge_q75']:.2f}); that group churned at {pct(INS['high_day_charge'])}."))
    if c.voice_mail_plan == "yes":
        out.append(Insight(tone="good", text=f"Voice mail plan subscribers churned at only {pct(INS['vmail_yes'])}."))
    if c.customer_service_calls <= 1 and c.international_plan == "no":
        out.append(Insight(tone="good", text="Few service calls and no international plan: both are linked to lower churn in the data."))
    return out


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/meta")
def meta():
    return {
        "states": ART["states"],
        "area_codes": ART["area_codes"],
        "ranges": ART["ranges"],
        "examples": ART["examples"],
        "base_rate": INS["base_rate"],
        "test_metrics": ART["test_metrics"],
    }


@app.post("/api/predict", response_model=Prediction)
def predict(customer: Customer):
    p = churn_probability(build_row(customer))
    level = "High" if p >= DECISION_THRESHOLD else ("Elevated" if p >= 0.25 else "Low")
    return Prediction(churn_probability=round(p, 4), will_churn=p >= DECISION_THRESHOLD,
                      risk_level=level, insights=explain(customer))


# Frontend: Vercel `public/` ko khud CDN se serve karta hai. Local run (uvicorn) ke liye yehi folder
# yahan mount hota hai. Must be the LAST route.
PUBLIC_DIR = BASE_DIR / "public"
if PUBLIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="public")

# python -m uvicorn main:app --reload
