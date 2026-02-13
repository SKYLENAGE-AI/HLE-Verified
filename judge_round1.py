# 目标：
# 对“初步筛选出的不合格题目”进行二次修复：
# 1) 利用上一轮 24 次模型回复（历史数据）做“高置信度优先”：
#    - 若某一候选答案/过程共识率 >= 83%（20/24），自动采纳
# 2) 若未达标：GPT-5.1 / Qwen3-Max / Gemini-3-Pro 三模型独立推理给出修复
#    - 允许输出“留空”（空字符串）以控制不确定风险
# 3) 引入 Gemini-3-Pro 作为终审节点，对三方修复结果做最终确认（可选其一或留空）
#
# 输入/输出：
# - 输入：第一阶段输出 Excel（需包含：题干、标准答案、标准过程、历史24次回复文本 or 已提取答案列）
# - 输出：新的 Excel（新增 sheet：Stage2_修复汇总 / Stage2_修复明细 / Stage2_配置）
# ------------------------------------------------------------


import os
import re
import json
import time
import hashlib
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

import base64
from io import BytesIO
from urllib.parse import urlparse
from pathlib import Path
from PIL import Image


# =========================
# 0. 配置区
# =========================

# 输入/输出文件
INPUT_XLSX  = ""   # 第一阶段输出
OUTPUT_XLSX = ""   # 第二阶段输出

# 哪些题进入二次修复（不合格题目筛选）
FILTER_MODE = "BY_FLAG_COLUMN"   # "BY_FLAG_COLUMN" | "ALL" | "ONLY_EMPTY_FIX"
FLAG_COLUMN = "总体合格状态"
BAD_VALUES  = {"不合格", "需复核", "处理失败"}

# 核心字段
ID_COL       = "id"
QUESTION_COL = "question"
STD_ANS_COL  = "answer"
STD_RAT_COL  = "rational"

# ====== 图片字段 ======
# Excel 里可能存的是 URL（包含 /image/000001.jpg?xxx），也可能直接存本地路径。
# 若是 URL，则会按 image/filename 的方式映射到本地文件夹
IMAGE_COL_CANDIDATES = ["图片路径", "image", "rationale_image", "image_url", "img", "题图", "图", "pic"]
IMAGE_FOLDER = "image"   # 本地图默认文件夹：image/xxx.jpg

# 第一阶段“历史24次回复”字段：
HIST_JSON_COL = ""         # 一列存 JSON（含24条文本）
HIST_COL_PREFIX = "回复_"  # 例如 回复_1 ... 回复_24
HIST_COL_RANGE = (1, 24)   # 1..24

# 共识阈值：83% -> 20/24
CONSENSUS_N = 24
CONSENSUS_MIN = 20

# 修复模型（独立推理）
MODEL_GPT51  = "gpt-5.1-vision-preview"
MODEL_QWEN3  = "qwen3-max-2025-09-23"
MODEL_GEMINI = "gemini-3-pro"

# 终审模型（Gemini 终审）
FINAL_GATE_MODEL = MODEL_GEMINI

SLEEP_BETWEEN_CALLS = 0.05


# =========================
# 1. 环境与客户端
# =========================

load_dotenv()
API_KEY  = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
BASE_URL = (os.getenv("BASE_URL") or os.getenv("OPENAI_BASE_URL") or "").rstrip("/")

if not API_KEY or not BASE_URL:
    raise RuntimeError("请在 .env 中设置 API_KEY / BASE_URL")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


# =========================
# 2. 工具函数
# =========================

def s(v) -> str:
    if v is None:
        return ""
    try:
        if isinstance(v, float) and pd.isna(v):
            return ""
    except Exception:
        pass
    return str(v).strip()

def jdump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)

def try_parse_json_block(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    m = re.search(r"(\{[\s\S]*\})", text)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None

def stable_norm_answer(ans: str) -> str:
    t = s(ans)
    if not t:
        return ""
    t = t.replace(" ", "").replace("×", "*").replace("−", "-").replace("，", ",").strip()
    return t

def stable_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# ====== 的图片路径提取 ======
def _extract_local_image_path(image_url: str, folder: str = IMAGE_FOLDER) -> str:
    """
    把 .../image/000001.jpg?xxx → image/000001.jpg（若文件存在才返回）
    与 3_math_solver_text.py 对齐
    """
    if not image_url:
        return ""
    try:
        parsed = urlparse(str(image_url).strip())
        filename = Path(parsed.path).name
        if not filename:
            return ""
        candidate = os.path.join(folder, filename)
        return candidate if os.path.exists(candidate) else ""
    except Exception:
        return ""

def _build_data_url_with_pillow(img_path: str, max_size=(2048, 2048)) -> str:
    """
    Pillow 打开本地图，必要时等比缩小 → dataURL(Base64)
    与 3_math_solver_text.py 对齐
    """
    with Image.open(img_path) as im:
        if im.mode not in ("RGB", "RGBA", "L"):
            im = im.convert("RGB")
        w, h = im.size
        if w > max_size[0] or h > max_size[1]:
            r = min(max_size[0] / w, max_size[1] / h)
            im = im.resize((int(w*r), int(h*r)), Image.Resampling.LANCZOS)

        fmt = (im.format or "JPEG").upper()
        if fmt not in ("PNG", "JPEG", "WEBP"):
            fmt = "JPEG"

        buf = BytesIO()
        if fmt == "PNG":
            im.save(buf, format="PNG", optimize=True)
            mime = "image/png"
        elif fmt == "WEBP":
            im.save(buf, format="WEBP", method=6)
            mime = "image/webp"
        else:
            im.save(buf, format="JPEG", quality=95, optimize=True)
            mime = "image/jpeg"

        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:{mime};base64,{b64}"


def detect_image_path_from_row(row: Dict[str, Any]) -> str:
    """
    从 row 中找图片字段：
    - 若字段值本身就是存在的本地文件路径 -> 直接用
    - 否则当作 URL/文件名 -> 用 _extract_local_image_path 映射到 image/xxx
    """
    for k in IMAGE_COL_CANDIDATES:
        v = s(row.get(k))
        if not v:
            continue
        # 本地路径直接命中
        if os.path.exists(v):
            return v
        # URL/文件名 -> 映射到 image/filename
        p = _extract_local_image_path(v, folder=IMAGE_FOLDER)
        if p:
            return p
    return ""


# =========================
# 3. 提示词
# =========================

EXTRACT_PROMPT = """你是一个严格的解答信息抽取器。
请从下面的“解题文本”中抽取：
1) 最终答案（尽量抽取 \\boxed{} 或最终结论对应的表达式）
2) 核心解题过程（保留关键推导步骤，去掉废话）

如果无法可靠抽取最终答案，请把 empty_answer 设为 true，并将 answer 留空。
输出必须是 JSON，不要输出多余文本。

JSON 格式：
{
  "empty_answer": true/false,
  "answer": "...",
  "rationale": "..."
}

解题文本：
{full_text}
"""

REPAIR_PROMPT = """你是一个严谨的题目修复专家。请审阅题干、标准答案、标准解题过程，以及历史模型回复（用于参考，但不必盲从）。
你的任务是输出你认为“正确”的答案与解题过程。

高风险控制：
- 如果你无法在合理置信度下确定最终答案，请将 empty_answer 设为 true 并留空 answer。
- rationale 仍可给出你能确认的推理骨架，或说明为何无法判断。

输出必须是 JSON：
{
  "empty_answer": true/false,
  "answer": "...",
  "rationale": "...",
  "confidence": 0.0-1.0,
  "notes": "一句话说明你依据的关键点/不确定点"
}

题干：
{question}

标准答案（可能为空）：
{std_answer}

标准解题过程（可能为空）：
{std_rationale}

历史24次回复（每条可能不完整，仅供参考）：
{history}
"""

FINAL_GATE_PROMPT = """你是终审节点。你需要在三方修复结果中做最终确认。
你可以：
- 选择其中一个结果（A/B/C）
- 或者判定三者都不可靠，选择 "EMPTY"（留空）

判决准则：
- 优先选择“自洽、可验证、与题干一致”的答案与过程
- 若答案不确定宁可 EMPTY，不要编造

输出必须是 JSON：
{
  "final_choice": "A"|"B"|"C"|"EMPTY",
  "empty_answer": true/false,
  "answer": "...",
  "rationale": "...",
  "reason": "一句话理由"
}

题干：
{question}

标准答案：
{std_answer}

标准过程：
{std_rationale}

候选A（GPT-5.1）：
{cand_a}

候选B（Qwen3-Max）：
{cand_b}

候选C（Gemini-3-Pro）：
{cand_c}
"""


# =========================
# 4. LLM 调用封装（多模态）
# =========================

def llm_call(model: str, prompt: str, temperature: float = 0.2, max_tokens: int = 1200,
             image_path: str = "") -> str:
    """：
    [{"role":"user","content":[{"type":"text","text":...},{"type":"image_url","image_url":{"url":data_url,"detail":"high"}}]}]
    """
    if image_path and os.path.exists(image_path):
        data_url = _build_data_url_with_pillow(image_path)
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}}
            ]
        }]
    else:
        messages = [{"role": "user", "content": prompt}]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return s(resp.choices[0].message.content)

def llm_extract_answer_rationale(full_text: str, model: str = MODEL_QWEN3, image_path: str = "") -> Dict[str, Any]:
    prompt = EXTRACT_PROMPT.format(full_text=full_text)
    out = llm_call(model=model, prompt=prompt, temperature=0.0, max_tokens=900, image_path=image_path)
    js = try_parse_json_block(out)
    if not js:
        return {"empty_answer": True, "answer": "", "rationale": "", "parse_error": True, "raw": out}
    js.setdefault("empty_answer", False)
    js.setdefault("answer", "")
    js.setdefault("rationale", "")
    js["answer"] = s(js["answer"])
    js["rationale"] = s(js["rationale"])
    return js

def llm_repair(question: str, std_answer: str, std_rationale: str, history: List[str], model: str,
              image_path: str = "") -> Dict[str, Any]:
    prompt = REPAIR_PROMPT.format(
        question=question,
        std_answer=std_answer,
        std_rationale=std_rationale,
        history=jdump(history),
    )
    out = llm_call(model=model, prompt=prompt, temperature=0.6, max_tokens=1200, image_path=image_path)
    js = try_parse_json_block(out)
    if not js:
        return {"empty_answer": True, "answer": "", "rationale": "", "confidence": 0.0, "notes": "parse_error", "raw": out}
    js.setdefault("empty_answer", False)
    js.setdefault("answer", "")
    js.setdefault("rationale", "")
    js.setdefault("confidence", 0.5)
    js.setdefault("notes", "")
    js["answer"] = s(js["answer"])
    js["rationale"] = s(js["rationale"])
    return js

def final_gate(question: str, std_answer: str, std_rationale: str,
               cand_a: Dict[str, Any], cand_b: Dict[str, Any], cand_c: Dict[str, Any],
               image_path: str = "") -> Dict[str, Any]:
    prompt = FINAL_GATE_PROMPT.format(
        question=question,
        std_answer=std_answer,
        std_rationale=std_rationale,
        cand_a=jdump(cand_a),
        cand_b=jdump(cand_b),
        cand_c=jdump(cand_c),
    )
    out = llm_call(model=FINAL_GATE_MODEL, prompt=prompt, temperature=0.0, max_tokens=900, image_path=image_path)
    js = try_parse_json_block(out)
    if not js:
        return {"final_choice": "EMPTY", "empty_answer": True, "answer": "", "rationale": "", "reason": "parse_error", "raw": out}
    js.setdefault("final_choice", "EMPTY")
    js.setdefault("empty_answer", True)
    js.setdefault("answer", "")
    js.setdefault("rationale", "")
    js.setdefault("reason", "")
    js["answer"] = s(js["answer"])
    js["rationale"] = s(js["rationale"])
    return js


# =========================
# 5. 历史24次回复读取与共识计算
# =========================

def load_history_from_row(row: Dict[str, Any]) -> List[str]:
    if HIST_JSON_COL and s(row.get(HIST_JSON_COL)):
        try:
            obj = json.loads(s(row.get(HIST_JSON_COL)))
            if isinstance(obj, list):
                return [s(x) for x in obj][:CONSENSUS_N]
            if isinstance(obj, dict) and "history" in obj and isinstance(obj["history"], list):
                return [s(x) for x in obj["history"]][:CONSENSUS_N]
        except Exception:
            pass

    hist = []
    for i in range(HIST_COL_RANGE[0], HIST_COL_RANGE[1] + 1):
        hist.append(s(row.get(f"{HIST_COL_PREFIX}{i}")))
    return hist[:CONSENSUS_N]


def consensus_from_history(history: List[str]) -> Dict[str, Any]:
    hist = [h for h in history if s(h)]
    if not hist:
        return {"hit": False, "reason": "no_history", "best_key": "", "count": 0, "top_items": []}

    def quick_extract_answer(text: str) -> str:
        t = s(text)
        if not t:
            return ""
        m = re.search(r"\\boxed\{([\s\S]*?)\}", t)
        if m:
            return stable_norm_answer(m.group(1))
        m = re.search(r"(答案[:：]\s*)([^\n\r]+)", t)
        if m:
            return stable_norm_answer(m.group(2))
        m = re.search(r"(Answer[:：]\s*)([^\n\r]+)", t, flags=re.IGNORECASE)
        if m:
            return stable_norm_answer(m.group(2))
        lines = [x.strip() for x in re.split(r"[\r\n]+", t) if x.strip()]
        if lines:
            return stable_norm_answer(lines[-1])
        return ""

    extracted_keys = []
    for h in hist:
        a = quick_extract_answer(h)
        extracted_keys.append(a if a else stable_hash(h[:2000]))

    cnt: Dict[str, int] = {}
    for k in extracted_keys:
        cnt[k] = cnt.get(k, 0) + 1

    best_key, best_count = max(cnt.items(), key=lambda x: x[1])
    top_items = sorted(cnt.items(), key=lambda x: -x[1])[:5]
    hit = best_count >= CONSENSUS_MIN
    return {"hit": hit, "best_key": best_key, "count": best_count, "top_items": top_items}


# =========================
# 6. 单题二次修复流程
# =========================

def should_process(row: Dict[str, Any]) -> bool:
    if FILTER_MODE == "ALL":
        return True
    if FILTER_MODE == "ONLY_EMPTY_FIX":
        return not s(row.get("stage2_final_answer"))
    if FILTER_MODE == "BY_FLAG_COLUMN":
        return s(row.get(FLAG_COLUMN)) in BAD_VALUES
    return True


def process_one(idx: int, row: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    qid = s(row.get(ID_COL) or row.get("题号") or row.get("row") or (idx + 1))
    question = s(row.get(QUESTION_COL) or row.get("题目") or row.get("问题") or row.get("题干"))
    std_answer = s(row.get(STD_ANS_COL) or row.get("答案"))
    std_rationale = s(row.get(STD_RAT_COL) or row.get("过程") or row.get("rationale"))

    image_path = detect_image_path_from_row(row)

    history = load_history_from_row(row)
    cons = consensus_from_history(history)

    detail_rows: List[Dict[str, Any]] = []

    # 1) 高置信度共识优先
    if cons["hit"]:
        target_text = ""
        for h in history:
            if not s(h):
                continue
            if "\\boxed" in h:
                target_text = h
                break
        if not target_text:
            target_text = next((h for h in history if s(h)), "")

        extracted = llm_extract_answer_rationale(target_text, model=MODEL_QWEN3, image_path=image_path)
        final_answer = "" if extracted.get("empty_answer") else s(extracted.get("answer"))
        final_rationale = s(extracted.get("rationale"))

        summary = {
            "id": qid,
            "row_index": idx,
            "stage2_mode": "CONSENSUS_AUTO_ACCEPT",
            "image_path": image_path,
            "consensus_count": cons["count"],
            "consensus_key": cons["best_key"],
            "stage2_final_answer": final_answer,
            "stage2_final_rationale": final_rationale,
            "stage2_final_empty": (final_answer == ""),
            "final_gate_choice": "CONSENSUS",
            "final_gate_reason": f"历史共识 {cons['count']}/{CONSENSUS_N} >= {CONSENSUS_MIN}"
        }

        detail_rows.append({
            "id": qid, "row_index": idx, "step": "consensus",
            "payload": jdump({"image_path": image_path, "consensus": cons, "extracted": extracted})[:20000]
        })
        return summary, detail_rows

    # 2) 未达成共识 -> 三模型独立推理（多模态一起发）
    cand_a = llm_repair(question, std_answer, std_rationale, history, MODEL_GPT51, image_path=image_path)
    time.sleep(SLEEP_BETWEEN_CALLS)
    cand_b = llm_repair(question, std_answer, std_rationale, history, MODEL_QWEN3, image_path=image_path)
    time.sleep(SLEEP_BETWEEN_CALLS)
    cand_c = llm_repair(question, std_answer, std_rationale, history, MODEL_GEMINI, image_path=image_path)
    time.sleep(SLEEP_BETWEEN_CALLS)

    detail_rows.append({"id": qid, "row_index": idx, "step": "repair_A_gpt51", "payload": jdump(cand_a)[:20000]})
    detail_rows.append({"id": qid, "row_index": idx, "step": "repair_B_qwen3", "payload": jdump(cand_b)[:20000]})
    detail_rows.append({"id": qid, "row_index": idx, "step": "repair_C_gemini", "payload": jdump(cand_c)[:20000]})

    # 3) 终审（同样把图一起给终审）
    gate = final_gate(question, std_answer, std_rationale, cand_a, cand_b, cand_c, image_path=image_path)
    detail_rows.append({"id": qid, "row_index": idx, "step": "final_gate", "payload": jdump(gate)[:20000]})

    final_choice = s(gate.get("final_choice", "EMPTY")).upper()
    if final_choice not in {"A", "B", "C", "EMPTY"}:
        final_choice = "EMPTY"

    if final_choice == "A":
        final_answer = "" if cand_a.get("empty_answer") else s(cand_a.get("answer"))
        final_rationale = s(cand_a.get("rationale"))
    elif final_choice == "B":
        final_answer = "" if cand_b.get("empty_answer") else s(cand_b.get("answer"))
        final_rationale = s(cand_b.get("rationale"))
    elif final_choice == "C":
        final_answer = "" if cand_c.get("empty_answer") else s(cand_c.get("answer"))
        final_rationale = s(cand_c.get("rationale"))
    else:
        final_answer, final_rationale = "", ""

    summary = {
        "id": qid,
        "row_index": idx,
        "stage2_mode": "TRI_REPAIR_PLUS_FINAL_GATE",
        "image_path": image_path,
        "consensus_count": cons["count"],
        "consensus_key": cons["best_key"],
        "candA_empty": bool(cand_a.get("empty_answer", False)),
        "candA_conf": float(cand_a.get("confidence", 0.0) or 0.0),
        "candB_empty": bool(cand_b.get("empty_answer", False)),
        "candB_conf": float(cand_b.get("confidence", 0.0) or 0.0),
        "candC_empty": bool(cand_c.get("empty_answer", False)),
        "candC_conf": float(cand_c.get("confidence", 0.0) or 0.0),
        "final_gate_choice": final_choice,
        "final_gate_reason": s(gate.get("reason", "")),
        "stage2_final_empty": (final_answer == ""),
        "stage2_final_answer": final_answer,
        "stage2_final_rationale": final_rationale,
    }
    return summary, detail_rows


# =========================
# 7. 主程序
# =========================

def main():
    df = pd.read_excel(INPUT_XLSX)
    rows = df.to_dict(orient="records")
    if not rows:
        raise RuntimeError("输入表为空")

    targets = [(i, r) for i, r in enumerate(rows) if should_process(r)]
    print(f"[INFO] 总行数={len(rows)}，进入二次修复={len(targets)}")

    summaries: List[Dict[str, Any]] = []
    details: List[Dict[str, Any]] = []

    for k, (idx, row) in enumerate(targets, start=1):
        qid = s(row.get(ID_COL) or row.get("题号") or row.get("row") or (idx + 1))
        print(f"[{k}/{len(targets)}] Processing id={qid} (row_index={idx})")

        try:
            summary, detail = process_one(idx, row)
            summaries.append(summary)
            details.extend(detail)
        except Exception as e:
            summaries.append({
                "id": qid,
                "row_index": idx,
                "stage2_mode": "FAILED",
                "error": str(e),
                "stage2_final_empty": True,
                "stage2_final_answer": "",
                "stage2_final_rationale": ""
            })
            details.append({"id": qid, "row_index": idx, "step": "exception", "payload": str(e)[:20000]})

        time.sleep(SLEEP_BETWEEN_CALLS)

    df_sum = pd.DataFrame(summaries).sort_values(by=["row_index"])
    df_det = pd.DataFrame(details).sort_values(by=["row_index", "step"])

    df_cfg = pd.DataFrame([{
        "INPUT_XLSX": INPUT_XLSX,
        "OUTPUT_XLSX": OUTPUT_XLSX,
        "FILTER_MODE": FILTER_MODE,
        "FLAG_COLUMN": FLAG_COLUMN,
        "BAD_VALUES": ",".join(sorted(BAD_VALUES)),
        "CONSENSUS_MIN": CONSENSUS_MIN,
        "CONSENSUS_N": CONSENSUS_N,
        "MODEL_GPT51": MODEL_GPT51,
        "MODEL_QWEN3": MODEL_QWEN3,
        "MODEL_GEMINI": MODEL_GEMINI,
        "FINAL_GATE_MODEL": FINAL_GATE_MODEL,
        "HIST_JSON_COL": HIST_JSON_COL,
        "HIST_COL_PREFIX": HIST_COL_PREFIX,
        "HIST_COL_RANGE": f"{HIST_COL_RANGE[0]}..{HIST_COL_RANGE[1]}",
        "ID_COL": ID_COL,
        "QUESTION_COL": QUESTION_COL,
        "STD_ANS_COL": STD_ANS_COL,
        "STD_RAT_COL": STD_RAT_COL,
        "IMAGE_COL_CANDIDATES": ",".join(IMAGE_COL_CANDIDATES),
        "IMAGE_FOLDER": IMAGE_FOLDER,
    }])

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        df_sum.to_excel(writer, sheet_name="Stage2_修复汇总", index=False)
        df_det.to_excel(writer, sheet_name="Stage2_修复明细", index=False)
        df_cfg.to_excel(writer, sheet_name="Stage2_配置", index=False)

    empty_cnt = int((df_sum.get("stage2_final_empty") == True).sum()) if "stage2_final_empty" in df_sum.columns else 0
    print(f"✅ 完成二次修复：输出 {OUTPUT_XLSX}")
    print(f"[STAT] 二次修复后 empty 数={empty_cnt} / {len(df_sum)}")


if __name__ == "__main__":
    main()
