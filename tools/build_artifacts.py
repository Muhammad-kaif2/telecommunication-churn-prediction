"""
Training-time artifacts -> deploy-time artifacts.

Notebook mein model sirf LightGBM hai; preprocessing (encodings + scaler) alag objects the
jo save nahi huye. Ye script wohi preprocessing training split se dobara banati hai
(same random_state=42, stratify) aur LightGBM trees ko ek halki JSON file mein export karti hai,
taake Vercel par lightgbm/scikit-learn/pandas install karne ki zaroorat na paray.

Run:  python tools/build_artifacts.py      (needs: lightgbm scikit-learn pandas numpy joblib)
"""
import json
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix, roc_auc_score

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "artifacts"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(HERE / "Data_Science_Challenge.csv")
df["area code"] = df["area code"].astype("object")
df = df.drop(columns=["phone number"])
df["total_intl_calls_log"] = np.log1p(df["total intl calls"])
df["binned_customer_service_calls"] = pd.cut(
    df["customer service calls"], bins=[0, 3, 5, np.inf], labels=["Low", "Medium", "High"], right=False)

X = df.drop(columns=["churn"]); y = df["churn"].astype(int)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# encodings (train only)
state_enc = pd.concat([X_tr["state"], y_tr], axis=1).groupby("state")["churn"].mean().to_dict()
area_enc = X_tr["area code"].value_counts(normalize=True).to_dict()
csc_enc = pd.concat([X_tr["binned_customer_service_calls"], y_tr], axis=1) \
            .groupby("binned_customer_service_calls", observed=False)["churn"].mean().to_dict()

SCALE_COLS = ["account length", "total day calls", "total day charge", "total eve calls", "total eve charge",
              "total night calls", "total night charge", "total intl charge", "total_intl_calls_log"]
sc = RobustScaler().fit(X_tr[SCALE_COLS])
scaler = {c: {"center": float(m), "scale": float(s)} for c, m, s in zip(SCALE_COLS, sc.center_, sc.scale_)}

def features(d):
    f = pd.DataFrame(index=d.index)
    f["state"] = d["state"].map(state_enc)
    f["account_length"] = d["account length"]
    f["area_code"] = d["area code"].map(area_enc)
    f["international_plan"] = d["international plan"].map({"no": 0, "yes": 1})
    f["voice_mail_plan"] = d["voice mail plan"].map({"no": 0, "yes": 1})
    for a, b in [("total_day_calls", "total day calls"), ("total_day_charge", "total day charge"),
                 ("total_eve_calls", "total eve calls"), ("total_eve_charge", "total eve charge"),
                 ("total_night_calls", "total night calls"), ("total_night_charge", "total night charge"),
                 ("total_intl_charge", "total intl charge"), ("total_intl_calls_log", "total_intl_calls_log")]:
        f[a] = d[b]
    f["binned_customer_service_calls"] = d["binned_customer_service_calls"].map(csc_enc).astype(float)
    for c in SCALE_COLS:
        k = c.replace(" ", "_")
        f[k] = (f[k] - scaler[c]["center"]) / scaler[c]["scale"]
    return f

model = joblib.load(HERE / "best_lgb_model.pkl")
FEATS = list(model.feature_name_)
Xt = features(X_te)[FEATS]

# ---- verify against the notebook's reported test result
pred = model.predict(Xt); proba = model.predict_proba(Xt)[:, 1]
tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
print("confusion (tn fp fn tp):", tn, fp, fn, tp, "| notebook: 561 9 32 65")
print("roc auc:", round(roc_auc_score(y_te, proba), 4))

# ---- export trees
dump = model.booster_.dump_model()
kinds = set()
def conv(n):
    if "leaf_value" in n:
        return round(n["leaf_value"], 12)
    kinds.add((n["decision_type"], n["missing_type"]))
    return [n["split_feature"], n["threshold"], conv(n["left_child"]), conv(n["right_child"])]
trees = [conv(t["tree_structure"]) for t in dump["tree_info"]]
print("split kinds:", kinds, "| trees:", len(trees))
assert dump["feature_names"] == FEATS

# ---- pure-python evaluator check
import math, sys
sys.path.insert(0, str(HERE.parent))
def ev(node, row):
    while isinstance(node, list):
        node = node[2] if row[node[0]] <= node[1] else node[3]
    return node
def p(row): return 1 / (1 + math.exp(-sum(ev(t, row) for t in trees)))
mine = np.array([p(r) for r in Xt.to_numpy(float)])
print("max |pure-python - lightgbm| proba diff:", float(np.abs(mine - proba).max()))

# ---- dataset insights (training-data base rates, shown next to a prediction)
full = pd.read_csv(HERE / "Data_Science_Challenge.csv"); full["churn"] = full["churn"].astype(int)
base = float(full["churn"].mean())
q75 = float(full["total day charge"].quantile(0.75))
insights = {
  "base_rate": base, "n": int(len(full)), "day_charge_q75": q75,
  "csc_4plus": float(full.loc[full["customer service calls"] >= 4, "churn"].mean()),
  "intl_plan_yes": float(full.loc[full["international plan"] == "yes", "churn"].mean()),
  "intl_plan_no": float(full.loc[full["international plan"] == "no", "churn"].mean()),
  "high_day_charge": float(full.loc[full["total day charge"] >= q75, "churn"].mean()),
  "vmail_yes": float(full.loc[full["voice mail plan"] == "yes", "churn"].mean()),
}
ranges = {c: [float(full[c].min()), float(full[c].max())] for c in
          ["account length", "total day calls", "total day charge", "total eve calls", "total eve charge",
           "total night calls", "total night charge", "total intl charge", "total intl calls", "customer service calls"]}

# real sample profiles for the "load example" buttons (phone number never used)
te_idx = X_te.index
ex = full.loc[te_idx].copy(); ex["p"] = proba
loyal = ex[(ex.churn == 0)].sort_values("p").iloc[3]
risky = ex[(ex.churn == 1)].sort_values("p", ascending=False).iloc[3]
cols = ["state", "account length", "area code", "international plan", "voice mail plan", "total day calls",
        "total day charge", "total eve calls", "total eve charge", "total night calls", "total night charge",
        "total intl charge", "total intl calls", "customer service calls"]
examples = {"stable": {c: (loyal[c].item() if hasattr(loyal[c], "item") else loyal[c]) for c in cols},
            "at_risk": {c: (risky[c].item() if hasattr(risky[c], "item") else risky[c]) for c in cols}}

art = {
  "features": FEATS, "state_enc": state_enc, "area_enc": {str(k): v for k, v in area_enc.items()},
  "csc_enc": {k: float(v) for k, v in csc_enc.items()}, "scaler": scaler,
  "states": sorted(state_enc), "area_codes": sorted(int(k) for k in area_enc),
  "insights": insights, "ranges": ranges, "examples": examples,
  "test_metrics": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "roc_auc": float(roc_auc_score(y_te, proba))},
}
(OUT / "preprocess.json").write_text(json.dumps(art, indent=1))
(OUT / "trees.json").write_text(json.dumps(trees, separators=(",", ":")))
print("written:", [(f.name, f.stat().st_size) for f in OUT.iterdir()])
print(json.dumps(examples, indent=1)); print(insights)
