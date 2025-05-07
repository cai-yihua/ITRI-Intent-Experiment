#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_intent_models.py
------------------------------------------------
在 verify_dataset 目錄下的每一個 <model>/result.csv 上計算分類指標並繪圖
加入：
1) LABEL_MAP：將中文意圖替換成數字／英文
2) Bar Chart 圖例移到圖片右側
"""

import os
import glob
import pandas as pd
from pathlib import Path
from collections import OrderedDict
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
import matplotlib.pyplot as plt

# 如果想要漂亮的熱圖可啟用 seaborn；若沒裝 seaborn 請將 import 與 sns 相關程式碼註解掉
try:
    import seaborn as sns

    USE_SEABORN = True
except ImportError:
    USE_SEABORN = False

# ========= 1) 中文 → 數字 / 英文 標籤對照 =========
LABEL_MAP = OrderedDict(
    {
        "查看UE的吞吐量": "1",
        "查詢 Cell 裡面的 SINR 熱圖": "2",
        "干擾演算法預測": "3",
        "干擾演算法執行": "4",
        "None": "5",
        # 若有更多中文類別，可在此續增；未列出的程式將自動遞增 6,7,8...
    }
)

def to_numeric_label(series: pd.Series, counter_start: int = 6) -> pd.Series:
    """
    將中文標籤轉為數字字串；若 LABEL_MAP 中沒有則自動遞增。
    """
    current_max = counter_start - 1
    for lbl in series.unique():
        if lbl not in LABEL_MAP:
            current_max += 1
            LABEL_MAP[lbl] = str(current_max)
    return series.replace(LABEL_MAP)

# ========= 2) 路徑設定 =========
ROOT = "/home/mitlab/intent_experiment/verify_dataset"
PATTERN = os.path.join(ROOT, "**", "result.csv")  # 遞迴找所有子資料夾

# ========= 3) 搜尋所有 result.csv =========
csv_files = glob.glob(PATTERN, recursive=True)
if not csv_files:
    raise FileNotFoundError(f"找不到任何符合 {PATTERN!r} 的檔案，請確認路徑是否正確。")

metrics_records = []  # 儲存所有模型指標

for csv_path in csv_files:
    df = pd.read_csv(csv_path)

    # ----------  (A) 標籤數字化  ----------
    y_true = to_numeric_label(df["groundtruth_intent"].astype(str).str.strip())
    y_pred = to_numeric_label(df["predict_intent"].astype(str).str.strip())

    model_name = Path(csv_path).parent.name  # <model> 資料夾名稱
    labels = sorted(set(y_true) | set(y_pred), key=lambda x: int(x))

    # ----------  (B) 計算指標  ----------
    acc = accuracy_score(y_true, y_pred)
    pre, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    metrics_records.append(
        {
            "model": model_name,
            "accuracy": acc,
            "precision": pre,
            "recall": rec,
            "f1": f1,
        }
    )

    # ----------  (C) 產生混淆矩陣圖  ----------
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(8, 6))
    if USE_SEABORN:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
        )
    else:
        plt.imshow(cm, interpolation="nearest")
        plt.colorbar()
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.yticks(range(len(labels)), labels)
        # 在格子上標數字
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="white")

    plt.title(f"Confusion Matrix – {model_name}")
    plt.xlabel("Predicted Intent (Numeric)")
    plt.ylabel("Ground-Truth Intent (Numeric)")
    plt.tight_layout()

    out_name = f"{model_name}_confusion_matrix.png"
    plt.savefig(out_name, dpi=300)
    plt.close()
    print(f"[✓] 已輸出 {out_name}")

# ========= 4) 彙整各模型指標 =========
metrics_df = (
    pd.DataFrame(metrics_records)
    .set_index("model")
    .sort_values("accuracy", ascending=False)
)
metrics_df.to_csv("model_metrics_summary.csv", encoding="utf-8")
print("\n=== 各模型整體指標 ===")
print(metrics_df)

# ----------  (D) 指標長條圖  ----------
ax = metrics_df[["accuracy", "precision", "recall", "f1"]].plot(
    kind="bar",
    figsize=(10, 6),
    ylim=(0, 1),
    rot=45,
)

ax.set_ylabel("Score")
ax.set_title("Intent Classification Performance per Model")

# --- 把圖例移到右側 ---
ax.legend(
    title="Metric",
    loc="center left",
    bbox_to_anchor=(1, 0.5),
)

# tight_layout 右邊留白 15% 給圖例
plt.tight_layout(rect=[0, 0, 0.85, 1])

plt.savefig("model_metrics_bar.png", dpi=300)
plt.close()
print("[✓] 已輸出 model_metrics_bar.png 與 model_metrics_summary.csv")
