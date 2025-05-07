# main.py
import json
import os
import csv
import requests
import uuid
import glob
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv(".env")
app = FastAPI()

# ---------- intent 對照 ----------
CODE_TO_INTENT = {
    "1": "查看UE的吞吐量",
    "2": "查詢 Cell 裡面的 SINR 熱圖",
    "3": "干擾演算法預測",
    "4": "干擾演算法執行",
    "5": "None",
}
INTENT_INDEX = {v: i for i, v in CODE_TO_INTENT.items()}

# ---------- 資料模型 ----------
class GeneratePayload(BaseModel):
    target_intent: str
    content: str


class OutputItem(BaseModel):
    intent: str
    text: str


class TriggerPayload(BaseModel):
    intent_code: str  # "1"~"5"
    generate_count: str
    stage: str
    query: str


class VerifyPayload(BaseModel):
    model: str            # ex: "gpt-4o", "gpt-4o-mini"
    stage: str = "verify_dataset"


# ---------- util：計算分類指標 ----------
def compute_metrics(y_true: List[str], y_pred: List[str]):
    from collections import Counter

    labels = set(y_true) | set(y_pred)
    tp, fp, fn = Counter(), Counter(), Counter()

    for t, p in zip(y_true, y_pred):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1

    precision, recall = {}, {}
    for lbl in labels:
        precision[lbl] = tp[lbl] / (tp[lbl] + fp[lbl]) if (tp[lbl] + fp[lbl]) else 0
        recall[lbl] = tp[lbl] / (tp[lbl] + fn[lbl]) if (tp[lbl] + fn[lbl]) else 0

    acc = sum(tp.values()) / len(y_true) if y_true else 0
    macro_p = sum(precision.values()) / len(labels)
    macro_r = sum(recall.values()) / len(labels)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
    }


# ---------- endpoint：驗證資料集 ----------
@app.post("/api/verify_dataset")
def verify_dataset(p: VerifyPayload):
    # 1. 讀 dataset -------------------------------------------------
    gt_intents, texts = [], []
    for csv_path in glob.glob("dataset/*/*.csv"):
        with open(csv_path, encoding="utf-8") as f:
            next(f)
            for line in f:
                intent, txt = line.rstrip("\n").split(",", 1)
                gt_intents.append(intent)
                texts.append(txt)

    if not texts:
        raise HTTPException(400, "dataset 資料夾為空，無法驗證")

    # 2. 準備呼叫 Dify ----------------------------------------------
    api_key = os.getenv("DIFY_API_KEY")
    host = os.getenv("HTTP_DIFY_HOST")
    if not api_key or not host:
        raise HTTPException(500, "DIFY_API_KEY / HTTP_DIFY_HOST 未設定")

    url = f"http://{host}/v1/chat-messages"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # ---------- 成本 / 穩定度統計 ----------
    pred_intents, rows = [], []

    total_cost = 0.0
    total_latency = 0.0
    total_prompt_tok = 0
    total_comp_tok = 0
    success_cnt = 0

    # 每個樣本詳細成本，稍後寫 cost.csv
    cost_rows: List[List] = []

    # 3. 逐句送 Dify（streaming + SSE）------------------------------
    for gt_intent, txt in zip(gt_intents, texts):
        body = {
            "query": txt,
            "inputs": {"stage": p.stage, "model": p.model, "text": txt},
            "response_mode": "streaming",
            "user": "verifier_bot",
            "auto_generate_name": False,
        }

        pred = "NONE"
        prompt_tok = comp_tok = 0
        price_usd = latency_sec = 0.0

        try:
            resp = requests.post(url, headers=headers, json=body, timeout=60, stream=True)
            resp.raise_for_status()

            wf_fin = None

            # -- 解析 SSE --
            for ln in resp.iter_lines(decode_unicode=True):
                if ln and ln.startswith("data: "):
                    data = json.loads(ln[6:])

                    # (a) llm node_finished → 成本 & 延遲
                    if data.get("event") == "node_finished" and data.get("data", {}).get("node_type") == "llm":
                        usage = data["data"]["outputs"].get("usage", {})
                        prompt_tok += int(usage.get("prompt_tokens", 0))
                        comp_tok += int(usage.get("completion_tokens", 0))
                        price_usd += float(usage.get("total_price", 0))
                        latency_sec = max(latency_sec, float(usage.get("latency", 0)))

                    # (b) workflow_finished → 最終答案
                    if data.get("event") == "workflow_finished":
                        wf_fin = data
                        break

            # ---------- 解析答案 ----------
            if wf_fin:
                success_cnt += 1
                ans_raw = (
                    wf_fin.get("data", {})
                    .get("outputs", {})
                    .get("answer", "")
                ).strip()

                # 去掉 ```code fence```，並嘗試雙層 loads
                if ans_raw.startswith("```"):
                    ans_raw = re.sub(r"^```[a-zA-Z]*\n?", "", ans_raw)
                    ans_raw = re.sub(r"\n?```$", "", ans_raw).strip()

                try:
                    ans_obj = json.loads(ans_raw)
                except json.JSONDecodeError:
                    ans_obj = json.loads(json.loads(ans_raw))
                pred = ans_obj.get("intent", "NONE")

        except Exception as e:
            print("verify error:", e)

        # ---------- 累積統計 ----------
        total_cost += price_usd
        total_latency += latency_sec
        total_prompt_tok += prompt_tok
        total_comp_tok += comp_tok

        cost_rows.append(
            [p.model, prompt_tok, comp_tok, price_usd, latency_sec]
        )

        pred_intents.append(pred)
        rows.append([p.model, txt, pred, gt_intent])

    # 4. 計算分類指標 -----------------------------------------------
    metrics = compute_metrics(gt_intents, pred_intents)

    # 5. 計算成本 / 穩定度指標 ---------------------------------------
    total_samples = len(texts)
    cost_stats = {
        "total_cost_usd": round(total_cost, 6),
        "avg_cost_usd": round(total_cost / total_samples, 6) if total_samples else 0,
        "avg_latency_sec": round(total_latency / total_samples, 3) if total_samples else 0,
        "success_rate": round(success_cnt / total_samples, 4),
        "fail_count": total_samples - success_cnt,
        "total_prompt_tokens": total_prompt_tok,
        "total_completion_tokens": total_comp_tok,
        "avg_prompt_tokens": round(total_prompt_tok / total_samples, 2) if total_samples else 0,
        "avg_completion_tokens": round(total_comp_tok / total_samples, 2) if total_samples else 0,
    }

    # 6. 寫 result.csv / cost.csv -----------------------------------
    out_dir = Path("verify_dataset") / p.model
    out_dir.mkdir(parents=True, exist_ok=True)

    # (a) result.csv：分類結果
    out_csv = out_dir / "result.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "text", "predict_intent", "groundtruth_intent"])
        w.writerows(rows)

    # (b) cost.csv：每句成本 + aggregate
    cost_csv = out_dir / "cost.csv"
    with cost_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "prompt_tokens", "completion_tokens", "total_price_usd", "latency_sec"])
        w.writerows(cost_rows)
        # 空行再寫 aggregate
        w.writerow([])
        w.writerow(["===== aggregate ====="])
        for k, v in cost_stats.items():
            w.writerow([k, v])

    # 7. 回傳 --------------------------------------------------------
    return {
        "total_samples": total_samples,
        "model": p.model,
        "metrics": metrics,
        "cost_stats": cost_stats,
        "result_file": str(out_csv),
        "cost_file": str(cost_csv),
    }


# ---------- CSV 寫檔 ----------
def write_group_to_csv(intent: str, rows: List[Dict[str, str]]) -> str:
    idx = INTENT_INDEX.get(intent, 5)
    ts = datetime.now().strftime("%Y%m%d%H%M%S%f")
    path = Path("dataset") / str(idx)
    path.mkdir(parents=True, exist_ok=True)

    file = path / f"{ts}.csv"
    with file.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["intent_type", "text"])
        w.writerows([[r["intent"], r["text"]] for r in rows])
    return str(file)


# ---------- Dify 回呼 ----------
@app.post("/api/generate_api", response_model=List[OutputItem])
def generate_api(body: GeneratePayload):
    lines = [ln.strip() for ln in body.content.splitlines() if ln.strip()]
    return [{"intent": body.target_intent, "text": ln} for ln in lines]


# ---------- 觸發 Workflow ----------
@app.post("/api/trigger_workflow")
def trigger_workflow(p: TriggerPayload):
    intent_type = CODE_TO_INTENT.get(p.intent_code, "None")
    conv_uid = str(uuid.uuid4())  # ★ 每次新的 UUID

    api_key = os.getenv("DIFY_API_KEY")
    host = os.getenv("HTTP_DIFY_HOST")
    if not api_key or not host:
        raise HTTPException(500, "環境變數 DIFY_API_KEY / HTTP_DIFY_HOST 未設定")

    url = f"http://{host}/v1/chat-messages"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    body = {
        "query": p.query,
        "inputs": {
            "conversation_uid": conv_uid,
            "stage": p.stage,
            "generate_count": p.generate_count,
            "intent_type": intent_type,
        },
        "response_mode": "streaming",
        "user": "trigger_bot",
        "files": [],
        "auto_generate_name": True,
    }

    # ---- 呼叫 Dify ----
    try:
        resp = requests.post(url, headers=headers, json=body, timeout=90, stream=True)
    except requests.RequestException as e:
        raise HTTPException(502, f"Dify 連線失敗：{e}")

    if resp.status_code >= 400:
        raise HTTPException(resp.status_code, f"Dify 回傳 {resp.status_code}: {resp.text}")

    # ---- 解析 SSE → workflow_finished ----
    wf_fin = next(
        (
            json.loads(ln[6:])
            for ln in resp.iter_lines(decode_unicode=True)
            if ln and ln.startswith("data: ")
            and json.loads(ln[6:]).get("event") == "workflow_finished"
        ),
        None,
    )
    if wf_fin is None:
        raise HTTPException(502, "未收到 workflow_finished")

    raw_answer = wf_fin.get("data", {}).get("outputs", {}).get("answer", "")
    try:
        parsed = json.loads(raw_answer)
        if isinstance(parsed, dict):
            parsed = [parsed]
    except json.JSONDecodeError:
        parsed = [
            {"intent": "None", "text": s.strip()}
            for s in raw_answer.splitlines()
            if s.strip()
        ]

    # ---- 分群寫檔 ----
    groups: Dict[str, List[Dict[str, str]]] = {}
    for r in parsed:
        groups.setdefault(r["intent"], []).append(r)

    files = [write_group_to_csv(k, v) for k, v in groups.items()]

    return {
        "status": True,
        "conversation_uid": conv_uid,
        "intent_type": intent_type,
        "stage": p.stage,
        "message": f"寫入 {sum(len(v) for v in groups.values())} 筆 / {len(files)} 檔",
        "files": files,
    }
