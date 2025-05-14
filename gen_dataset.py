
import csv
import json
import os
import uuid
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import requests
from dotenv import load_dotenv


# ---------- 讀取組態 ----------
with Path("config.json").open(encoding="utf-8") as f:
    conf = json.load(f)
generate_count = conf["generate_count"]
INTENT_MAP = conf.get("INTENT_MAP")
INTENT_INDEX = {v: k for k, v in INTENT_MAP.items()}

# ---------- CSV 寫檔 ----------
def write_group_to_csv(intent: str, rows: List[Dict[str, str]]) -> str:
    """
    將同一 intent 的句子寫入 dataset/<idx>/YYYYMMDDhhmmssfff.csv，並回傳檔案路徑
    """
    idx = INTENT_INDEX.get(intent)
    ts  = datetime.now().strftime("%Y%m%d%H%M%S%f")
    out_dir = Path("dataset") / str(idx)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{ts}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["intent_type", "text"])
        for r in rows:
            w.writerow([r["intent"], r["text"]])
    return str(csv_path)

# ---------- 核心函式 ----------
def trigger_workflow(
    intent_code: str,        # "1"~"5"
    generate_count: str,     # 文字即可，Dify 端吃 str
    stage: str,              # 例如 "gen_dataset"
    query: str,              # 使用者要給 LLM 的 prompt
) -> Dict:
    """
    觸發一次 Dify Workflow，寫入 CSV，並回傳執行結果摘要 dict
    """
    # 1. 轉換 intent ------------------------------------------------
    intent_type = INTENT_MAP.get(intent_code, "None")
    conv_uid = str(uuid.uuid4())

    # 2. 讀取環境變數 ----------------------------------------------
    load_dotenv(".env")
    api_key = os.getenv("DIFY_API_KEY")
    host    = os.getenv("HTTP_DIFY_HOST")
    if not api_key or not host:
        raise RuntimeError("請先在 .env 設定 DIFY_API_KEY / HTTP_DIFY_HOST")

    url = f"http://{host}/v1/chat-messages"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # 3. 準備請求 ---------------------------------------------------
    body = {
        "query": query,
        "inputs": {
            "conversation_uid": conv_uid,
            "stage": stage,
            "generate_count": generate_count,
            "intent_type": intent_type,
        },
        "response_mode": "streaming",
        "user": "trigger_bot",
        "files": [],
        "auto_generate_name": True,
    }

    # 4. 呼叫 Dify --------------------------------------------------
    try:
        resp = requests.post(url, headers=headers, json=body, timeout=90, stream=True)
    except requests.RequestException as e:
        raise RuntimeError(f"Dify 連線失敗：{e}")

    if resp.status_code >= 400:
        raise RuntimeError(f"Dify 回傳 {resp.status_code}: {resp.text}")


    # 5. 解析 SSE，直到 workflow_finished ---------------------------
    wf_fin = None
    for ln in resp.iter_lines(decode_unicode=True):
        if ln and ln.startswith("data: "):
            payload = json.loads(ln[6:])
            if payload.get("event") == "workflow_finished":
                wf_fin = payload
                break

    if wf_fin is None:
        raise RuntimeError("未收到 workflow_finished")

    # 6. 解析 answer ------------------------------------------------
    answer_str  = wf_fin.get("data", {}).get("outputs", {}).get("answer", "")
    # --- 去掉 ```json ... ``` 圍欄 ---
    if answer_str.startswith("```"):
        answer_str = re.sub(r"^```[a-zA-Z]*\n?", "", answer_str)
        answer_str = re.sub(r"\n?```$", "", answer_str).strip()

    # --- 轉成 Python 物件並擷取 text ---
    try:
        payload   = json.loads(answer_str)          # 可能是 list 或 dict
        if isinstance(payload, dict):
            payload = [payload]

        # 把所有 text 串起來（用 \n 分隔，也可改成其他格式）
        raw_answer = "\n".join(item.get("text", "") for item in payload)

    except json.JSONDecodeError:
        # 若不是合法 JSON，就直接帶原字串
        raw_answer = answer_str

    # ➜ Dify 若回傳 JSON，就照原邏輯解析；
    #   若回傳純文字行，就把 intent 設成 *當前 intent_type*
    try:
        parsed = json.loads(raw_answer)
        if isinstance(parsed, dict):
            parsed = [parsed]
    except json.JSONDecodeError:
        parsed = [
            {"intent": intent_type, "text": s.strip()}
            for s in raw_answer.splitlines() if s.strip()
        ]

    # 7. 直接寫一個 CSV，不再分 intent 子資料夾 ---------------------
    files = [write_group_to_csv(intent_type, parsed)]

    # 8. 回傳摘要 ---------------------------------------------------
    return {
        "status": True,
        "conversation_uid": conv_uid,
        "intent_type": intent_type,
        "stage": stage,
        "message": f"寫入 {len(parsed)} 筆 / {len(files)} 檔",
        "files": files,
    }

# ---------- CLI 範例 ----------
if __name__ == "__main__":
    for code, intent in INTENT_MAP.items():
        print(f"\n=== 觸發 {intent} ===")
        try:
            result = trigger_workflow(
                intent_code=code,
                generate_count=generate_count,
                stage="gen_dataset",
                query=intent,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"× 失敗：{e}")
