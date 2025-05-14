#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_intent_pipeline.py
--------------------------------------------------------------
步驟：
1. 讀取 dataset/*/*.csv → 逐句打 Dify /v1/chat-messages
   - 依模型名稱分別寫 verify_dataset/<TEMP>/<model>/cost.csv 與 result.csv
2. 統計 cost.csv → 產生 result_cost/ 圖表與彙總
3. 統計 result.csv → 產生 result/ 圖表與彙總
--------------------------------------------------------------
必要套件：
    pip install requests python-dotenv pandas matplotlib seaborn scikit-learn
"""
import os, re, csv, json, glob, uuid, textwrap, sys
from collections import OrderedDict, Counter
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import requests
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv, find_dotenv
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

# ---------- 0) 讀 .env ----------
load_dotenv(find_dotenv())
API_KEY = os.getenv("DIFY_API_KEY")
HOST    = os.getenv("HTTP_DIFY_HOST")
if not API_KEY or not HOST:
    sys.exit("請在 .env 設定 DIFY_API_KEY / HTTP_DIFY_HOST")

URL     = f"http://{HOST}/v1/chat-messages"     # 若雲端請改 https
HEADERS = {
    "Authorization": f"Bearer " + API_KEY,
    "Content-Type": "application/json",
}

# ---------- 1) 讀取組態 ----------
with Path("config.json").open(encoding="utf-8") as f:
    conf = json.load(f)
MODELS       = conf["MODELS"]
TEMP         = conf["TEMP"]
INTENT_MAP   = conf.get("INTENT_MAP")
INTENT_INDEX = {v: k for k, v in INTENT_MAP.items()}

VERIFY_ROOT = Path(f"verify_dataset/{TEMP}")
RESULT_DIR  = Path(f"result/temperature/{TEMP}")
COST_DIR    = Path("result_cost")
for p in (VERIFY_ROOT, RESULT_DIR, COST_DIR):
    p.mkdir(parents=True, exist_ok=True)

# ---------- 2) 公用函式 ----------
def compute_metrics(y_true, y_pred):
    lbls = set(y_true) | set(y_pred)
    tp = Counter([t for t, p in zip(y_true, y_pred) if t == p])
    fp = Counter([p for t, p in zip(y_true, y_pred) if t != p])
    fn = Counter([t for t, p in zip(y_true, y_pred) if t != p])
    prec = {l: tp[l]/(tp[l]+fp[l]) if tp[l]+fp[l] else 0 for l in lbls}
    reca = {l: tp[l]/(tp[l]+fn[l]) if tp[l]+fn[l] else 0 for l in lbls}
    acc  = sum(tp.values())/len(y_true) if y_true else 0
    return acc, prec, reca

def write_dataset_csv(intent:str, rows:List[Dict[str,str]]):
    idx = INTENT_INDEX.get(intent,"5")
    ts  = datetime.now().strftime("%Y%m%d%H%M%S%f")
    path = Path("dataset")/str(idx); path.mkdir(parents=True, exist_ok=True)
    fp   = path/f"{ts}.csv"
    with fp.open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["intent_type","text"])
        w.writerows([[r['intent'],r['text']] for r in rows])
    return str(fp)

# ---------- 3) 核心：對單一模型驗證 -------------------------------
def verify_one_model(model:str, stage:str="verify_dataset"):
    out_dir = VERIFY_ROOT/model
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3-1 讀 dataset
    gt_ints,texts=[],[]
    for csv_path in glob.glob("dataset/*/*.csv"):
        with open(csv_path,encoding="utf-8") as f:
            rdr=csv.reader(f);next(rdr,None)
            for row in rdr:
                if len(row)<2:continue
                intent,txt=row[0].strip(),row[1].strip()
                if not intent or not txt:continue
                gt_ints.append(intent); texts.append(txt)
    if not texts:
        print("[!] dataset 資料夾為空，跳過")
        return

    # 3-2 逐句呼叫 Dify
    rows,cost_rows,pred_ints=[],[],[]
    tot_prompt=tot_comp=tot_pcost=tot_ccost=tot_lat=succ=0
    for gt,txt in zip(gt_ints,texts):
        body={
            "query":txt,
            "inputs":{"stage":stage,"model":model,"text":txt},
            "response_mode":"streaming",
            "user":"verifier_bot",
            "auto_generate_name":False,
        }
        try:
            resp=requests.post(URL,headers=HEADERS,json=body,timeout=60,stream=True)
            resp.raise_for_status()
        except Exception as e:
            print("   ↳ 連線失敗：",e); continue
        wf_fin=None
        ptok=ctok=0; pcost=ccost=lat=0
        for ln in resp.iter_lines(decode_unicode=True):
            if ln and ln.startswith("data: "):
                data=json.loads(ln[6:])
                if data.get("event")=="node_finished" and data.get("data",{}).get("node_type")=="llm":
                    usage=data["data"]["outputs"].get("usage",{})
                    ptok+=int(usage.get("prompt_tokens",0))
                    ctok+=int(usage.get("completion_tokens",0))
                    pp_unit=float(usage.get("prompt_price_unit",0))
                    cp_unit=float(usage.get("completion_price_unit",0))
                    pp=float(usage.get("prompt_unit_price",0))
                    cp=float(usage.get("completion_unit_price",0))
                    pcost+=ptok*pp_unit*pp
                    ccost+=ctok*cp_unit*cp
                    lat=float(usage.get("latency",0))
                if data.get("event")=="workflow_finished":
                    wf_fin=data;break
        pred="None"
        if wf_fin:
            succ+=1
            ans_raw=wf_fin["data"]["outputs"].get("answer","").strip()
            if ans_raw.startswith("```"):
                ans_raw=re.sub(r"^```[a-zA-Z]*\n?","",ans_raw)
                ans_raw=re.sub(r"\n?```$","",ans_raw).strip()
            try:
                ans=json.loads(ans_raw)
                if isinstance(ans,dict):ans=[ans]
                pred=ans[0].get("intent","None")
            except json.JSONDecodeError:
                pass
        tot_prompt+=ptok; tot_comp+=ctok
        tot_pcost+=pcost; tot_ccost+=ccost; tot_lat+=lat
        cost_rows.append([
            model,ptok,pp_unit,pp,f"{pcost:.9f}",
            ctok,cp_unit,cp,f"{ccost:.9f}",f"{lat:.9f}"
        ])
        pred_ints.append(pred)
        rows.append([model,txt,pred,gt])

    # 3-3 統計 & 寫檔
    total=len(texts); tot=tot_pcost+tot_ccost
    cost_stats={
        "total_prompt_cost_usd":round(tot_pcost,6),
        "total_completion_cost_usd":round(tot_ccost,6),
        "total_cost_usd":round(tot,6),
        "avg_prompt_cost_usd":round(tot_pcost/total,6),
        "avg_completion_cost_usd":round(tot_ccost/total,6),
        "avg_cost_usd":round(tot/total,6),
        "avg_latency_sec":round(tot_lat/total,3),
        "success_rate":round(succ/total,4),
        "fail_count":total-succ,
        "total_prompt_tokens":tot_prompt,
        "total_completion_tokens":tot_comp,
        "avg_prompt_tokens":round(tot_prompt/total,2),
        "avg_completion_tokens":round(tot_comp/total,2),
    }
    (out_dir/"result.csv").write_text(
        "model,text,predict_intent,groundtruth_intent\n"+
        "\n".join(",".join(map(str,r)) for r in rows),
        encoding="utf-8"
    )
    with (out_dir/"cost.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["model","prompt_tokens","prompt_price_unit","prompt_unit_price",
                    "prompt_cost_usd","completion_tokens","completion_price_unit",
                    "completion_unit_price","completion_cost_usd","latency_sec"])
        w.writerows(cost_rows)
        w.writerow([]); w.writerow(["===== aggregate ====="])
        for k,v in cost_stats.items(): w.writerow([k,v])
    print(f"[✓] {model} 驗證完成 → {out_dir}")

# ---------- 4) 評估 cost.csv (result_cost) ------------------------
def evaluate_cost():
    part_metrics=["success_rate"]
    full_metrics=[
        "avg_latency_sec","total_completion_cost_usd","total_cost_usd",
        "total_prompt_cost_usd","avg_prompt_tokens","avg_completion_tokens"
    ]
    def parse_agg(path:Path):
        agg=OrderedDict(); flag=False
        if not path.exists():return agg
        for ln in path.read_text().splitlines():
            ln=ln.strip()
            if not ln:continue
            if re.match(r"^=+.*aggregate",ln,flags=re.I):flag=True;continue
            if flag:
                try:k,v=ln.split(",",1); agg[k.strip()]=float(v)
                except: pass
        return agg

    records=[]
    for m in MODELS:
        agg=parse_agg(VERIFY_ROOT/m/"cost.csv")
        if agg: agg["model"]=m; records.append(agg)
    df=pd.DataFrame(records).set_index("model")
    COST_DIR.mkdir(exist_ok=True)
    df.to_csv(COST_DIR/"summary.csv",float_format="%.6g")

    colors=["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd"]
    def bar(metrics,tag):
        for i,met in enumerate(metrics):
            ax=df[met].plot(kind="bar",color=colors[i%len(colors)],figsize=(9,5))
            ax.set_ylabel(met); ax.set_title(f"{met} by Model")
            plt.tight_layout()
            plt.savefig(COST_DIR/f"{tag}_{met}.png",dpi=300)
            plt.close()
    bar(part_metrics,"cost") ; bar(full_metrics,"cost")

# ---------- 5) 評估 intent result.csv (result) --------------------
def evaluate_intent():
    label_map=OrderedDict({
        "查看UE的吞吐量":"1","查詢 Cell 裡面的 SINR 熱圖":"2",
        "干擾演算法預測":"3","干擾演算法執行":"4","None":"5"
    })
    def to_num(s):
        cur=max(map(int,label_map.values()))
        for lbl in s.unique():
            if lbl not in label_map:
                cur+=1; label_map[lbl]=str(cur)
        return s.replace(label_map)

    metrics,records=[],[]
    for m in MODELS:
        fp=VERIFY_ROOT/m/"result.csv"
        if not fp.exists():continue
        df=pd.read_csv(fp).fillna("None")
        y_true=to_num(df["groundtruth_intent"].astype(str).str.strip())
        y_pred=to_num(df["predict_intent"].astype(str).str.strip())
        acc=accuracy_score(y_true,y_pred)
        pre,rec,f1,_=precision_recall_fscore_support(y_true,y_pred,average="macro",zero_division=0)
        records.append({"model":m,"accuracy":acc,"precision":pre,"recall":rec,"f1":f1})

        labels=sorted(set(y_true)|set(y_pred),key=lambda x:int(x))
        cm=confusion_matrix(y_true,y_pred,labels=labels)
        plt.figure(figsize=(6,5))
        plt.imshow(cm);plt.colorbar()
        plt.xticks(range(len(labels)),labels);plt.yticks(range(len(labels)),labels)
        vmax=cm.max()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val=cm[i,j]; color="black" if val>0.5*vmax else "white"
                plt.text(j,i,val,ha="center",va="center",color=color)
        plt.title(f"Confusion Matrix – {m}")
        plt.xlabel("Predicted Intent (Numeric)")
        plt.ylabel("Ground-Truth Intent (Numeric)")
        plt.tight_layout()
        out=RESULT_DIR/f"{m}_cm.png"; plt.savefig(out,dpi=300); plt.close()

    md=pd.DataFrame(records).set_index("model").sort_values("accuracy",ascending=False)
    # md=pd.DataFrame(records).set_ylabel("Score").set_title("Intent Classification Performance per Model").sort_values("accuracy",ascending=False)
    RESULT_DIR.mkdir(exist_ok=True)
    md.to_csv(RESULT_DIR/"model_metrics_summary.csv")
    ax=md.plot(kind="bar",ylim=(0,1),figsize=(10,6))
    ax.legend(loc="center left",bbox_to_anchor=(1,0.5))
    ax.set_ylabel("Score")
    ax.set_title("Intent Classification Performance per Model")
    plt.tight_layout(rect=[0,0,0.85,1])
    plt.savefig(RESULT_DIR/"model_metrics_bar.png",dpi=300)
    plt.close()

# ---------- 6) 主程式 ----------
if __name__=="__main__":
    # for mdl in MODELS:
    #     verify_one_model(mdl)
    evaluate_cost()
    evaluate_intent()
    print("\n🎉 Pipeline 完成！所有輸出請見：")
    print("   - 驗證結果        →", VERIFY_ROOT.resolve())
    print("   - Cost 圖表彙整   →", COST_DIR.resolve())
    print("   - Intent 圖表彙整 →", RESULT_DIR.resolve())
