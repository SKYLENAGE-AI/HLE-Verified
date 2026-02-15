import os
import re
import json
import time
import base64
from io import BytesIO
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import urlparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI


# =========================
# 0) 配置区
# =========================

INPUT_XLSX  = ""
OUTPUT_XLSX = ""

# 字段
COL_ID            = "id"
COL_QUESTION      = "question"
COL_IMAGE         = "image"          # URL 或路径字符串
COL_ANS_ORIG       = "answer"
COL_ANS_LLM        = "llm_answer"
COL_RAT_ORIG       = "rational"
COL_RAT_LLM        = "llm_rationale"

# 本地图片文件夹
LOCAL_IMAGE_FOLDER = "image"

# 参与交叉验证的三模型
MODEL_GPT51  = "gpt-5.1-vision-preview"
MODEL_QWEN3  = "qwen3-max-2025-09-23"
MODEL_GEMINI = "gemini-3-pro"
MODELS = [MODEL_GPT51, MODEL_QWEN3, MODEL_GEMINI]

PASS_K = 8

# 并行参数：建议保守，避免网关限流
MAX_WORKERS = 6
SLEEP_BETWEEN_CALLS = 0.02

# 采样温度：Pass@k 需要一定随机性；
TEMPERATURE = 0.7
MAX_TOKENS  = 900

# 复核规则：某维度上，>=2 个模型 0/8 则标记人工复核
ZERO_PASS_TRIGGER_MODELS = 2


# =========================
# 1) 环境与客户端
# =========================

load_dotenv()
API_KEY  = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
BASE_URL = (os.getenv("BASE_URL") or os.getenv("OPENAI_BASE_URL") or "").rstrip("/")

if not API_KEY or not BASE_URL:
    raise RuntimeError("请在 .env 中设置 API_KEY / BASE_URL")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


# =========================
# 2) 工具：字符串/JSON
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

def try_parse_json_block(text: str) -> Dict[str, Any]:
    if not text:
        return {"parse_error": "empty", "raw": ""}
    m = re.search(r"(\{[\s\S]*\})", text)
    if not m:
        return {"parse_error": "no_json", "raw": text}
    block = m.group(1)
    try:
        return json.loads(block)
    except Exception:
        return {"parse_error": "json_error", "raw": text}

def jdump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


# =========================
# 3) 读图：URL -> 本地 image/<filename> -> dataURL
# =========================

def _extract_local_image_path(image_url: str, folder: str = LOCAL_IMAGE_FOLDER) -> str:
    """把 .../image/000001.jpg?xxx → image/000001.jpg（若文件存在才返回）"""
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
    """Pillow 打开本地图，必要时等比缩小 → dataURL(Base64)"""
    with Image.open(img_path) as im:
        if im.mode not in ("RGB", "RGBA", "L"):
            im = im.convert("RGB")

        w, h = im.size
        if w > max_size[0] or h > max_size[1]:
            r = min(max_size[0] / w, max_size[1] / h)
            im = im.resize((int(w * r), int(h * r)), Image.Resampling.LANCZOS)

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


# =========================
# 4) Prompt：三维二元判决 + 理由（每次采样一次）
# =========================

JUDGE_PROMPT = """你是一个严格的质量追溯/交叉验证评审员。
我会给你：题干（可能含配图）、原始答案、修复后答案、原始解题过程、修复后解题过程。
你的任务是分别判断三件事是否“正确/可靠”：
(1) 题目本身是否表述清晰自洽（题干+配图一致、条件充分、无明显矛盾） -> question_correct
(2) 修复后答案 llm_answer 是否正确且与题意匹配 -> answer_correct
(3) 修复后过程 llm_rationale 是否逻辑正确且能推出 llm_answer，并与题目一致 -> rationale_correct

注意：
- 若你无法确定，也必须在 true/false 中选择更保守的一边（倾向 false）。
- 只评价“修复后”的答案与过程；原始字段仅作参考。
- 如果图片存在，请结合图片信息判断。

请严格输出 JSON，不要输出 JSON 之外内容：
{
  "question_correct": true/false,
  "answer_correct": true/false,
  "rationale_correct": true/false,
  "reason_question": "一句话理由",
  "reason_answer": "一句话理由",
  "reason_rationale": "一句话理由"
}

题干：
{question}

原始答案：
{ans_orig}

修复后答案（llm_answer）：
{ans_llm}

原始解题过程：
{rat_orig}

修复后解题过程（llm_rationale）：
{rat_llm}
"""


# =========================
# 5) LLM 调用：可选图片
# =========================

def call_model(prompt: str, model: str, image_data_url: Optional[str]) -> str:
    if image_data_url:
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_url, "detail": "high"}}
            ]
        }]
    else:
        messages = [{"role": "user", "content": prompt}]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return (resp.choices[0].message.content or "").strip()


# =========================
# 6) 单题单模型 Pass@8
# =========================

def passk_for_one_model(
    qid: str,
    model: str,
    prompt: str,
    image_data_url: Optional[str],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    返回：
      model_summary: 该模型在该题的 pass@8 统计
      model_details: 8 次采样的逐次判决明细
    """
    details = []
    qc_cnt = ac_cnt = rc_cnt = 0

    for k in range(1, PASS_K + 1):
        t0 = time.time()
        try:
            raw = call_model(prompt, model, image_data_url)
            parsed = try_parse_json_block(raw)

            if "parse_error" in parsed:
                # 解析失败：按保守 false 计
                qc = ac = rc = False
                rq = ra = rr = f"parse_error={parsed.get('parse_error')}"
            else:
                qc = bool(parsed.get("question_correct", False))
                ac = bool(parsed.get("answer_correct", False))
                rc = bool(parsed.get("rationale_correct", False))
                rq = s(parsed.get("reason_question"))
                ra = s(parsed.get("reason_answer"))
                rr = s(parsed.get("reason_rationale"))

            qc_cnt += int(qc)
            ac_cnt += int(ac)
            rc_cnt += int(rc)

            details.append({
                "id": qid,
                "model": model,
                "sample_k": k,
                "question_correct": qc,
                "answer_correct": ac,
                "rationale_correct": rc,
                "reason_question": rq[:800],
                "reason_answer": ra[:800],
                "reason_rationale": rr[:800],
                "has_image": bool(image_data_url),
                "latency_sec": round(time.time() - t0, 3),
                "raw": raw[:4000],
            })
        except Exception as e:
            details.append({
                "id": qid,
                "model": model,
                "sample_k": k,
                "question_correct": False,
                "answer_correct": False,
                "rationale_correct": False,
                "reason_question": f"call_failed: {e}",
                "reason_answer": f"call_failed: {e}",
                "reason_rationale": f"call_failed: {e}",
                "has_image": bool(image_data_url),
                "latency_sec": round(time.time() - t0, 3),
                "raw": "",
            })

        time.sleep(SLEEP_BETWEEN_CALLS)

    model_summary = {
        "id": qid,
        f"{model}_q_pass8": f"{qc_cnt}/{PASS_K}",
        f"{model}_a_pass8": f"{ac_cnt}/{PASS_K}",
        f"{model}_r_pass8": f"{rc_cnt}/{PASS_K}",
        f"{model}_q_rate": qc_cnt / PASS_K,
        f"{model}_a_rate": ac_cnt / PASS_K,
        f"{model}_r_rate": rc_cnt / PASS_K,
        f"{model}_q_zero": (qc_cnt == 0),
        f"{model}_a_zero": (ac_cnt == 0),
        f"{model}_r_zero": (rc_cnt == 0),
    }
    return model_summary, details


# =========================
# 7) 单题处理：三模型并行 + 复核标记
# =========================

def process_one_question(row: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    qid = s(row.get(COL_ID) or row.get("题号") or row.get("row") or "")
    question = s(row.get(COL_QUESTION))
    img_ref = s(row.get(COL_IMAGE))

    ans_orig = s(row.get(COL_ANS_ORIG))
    ans_llm  = s(row.get(COL_ANS_LLM))

    rat_orig = s(row.get(COL_RAT_ORIG))
    rat_llm  = s(row.get(COL_RAT_LLM))

    # 按你的读图方式：URL -> 本地 image/<filename>
    local_img = _extract_local_image_path(img_ref, folder=LOCAL_IMAGE_FOLDER)
    image_data_url = _build_data_url_with_pillow(local_img) if local_img else None

    prompt = JUDGE_PROMPT.format(
        question=question or "(空)",
        ans_orig=ans_orig or "(空)",
        ans_llm=ans_llm or "(空)",
        rat_orig=rat_orig or "(空)",
        rat_llm=rat_llm or "(空)",
    )

    # 三模型并行做 pass@8
    model_summaries = []
    all_details = []

    with ThreadPoolExecutor(max_workers=min(3, MAX_WORKERS)) as ex:
        futs = {
            ex.submit(passk_for_one_model, qid, m, prompt, image_data_url): m
            for m in MODELS
        }
        for fut in as_completed(futs):
            ms, det = fut.result()
            model_summaries.append(ms)
            all_details.extend(det)

    # 合并三模型统计到一行 summary
    summary = {
        "id": qid,
        "question": question,
        "image": img_ref,
        "has_image": bool(image_data_url),
        "answer": ans_orig,
        "llm_answer": ans_llm,
        "rational": rat_orig,
        "llm_rationale": rat_llm,
    }
    for ms in model_summaries:
        summary.update(ms)

    # 复核标记：在某一维度上，>=2 个模型 0/8
    # 维度：question / answer / rationale
    q_zero = sum(1 for m in MODELS if summary.get(f"{m}_q_zero") is True)
    a_zero = sum(1 for m in MODELS if summary.get(f"{m}_a_zero") is True)
    r_zero = sum(1 for m in MODELS if summary.get(f"{m}_r_zero") is True)

    need_review = (q_zero >= ZERO_PASS_TRIGGER_MODELS) or (a_zero >= ZERO_PASS_TRIGGER_MODELS) or (r_zero >= ZERO_PASS_TRIGGER_MODELS)
    summary["need_human_review"] = need_review
    summary["review_reason"] = f"q_zero_models={q_zero}, a_zero_models={a_zero}, r_zero_models={r_zero}"

    # 要求的“正确性比例”字段（示例：1/8）
    # 这里给出：每模型三维 rate；如果你想要“跨模型汇总”，也可以再加一个 overall
    # 额外给个 overall（3模型合并的正确次数 / (3*8)）
    q_correct_total = sum(int(summary.get(f"{m}_q_rate", 0) * PASS_K) for m in MODELS)
    a_correct_total = sum(int(summary.get(f"{m}_a_rate", 0) * PASS_K) for m in MODELS)
    r_correct_total = sum(int(summary.get(f"{m}_r_rate", 0) * PASS_K) for m in MODELS)
    denom = len(MODELS) * PASS_K

    summary["overall_q_rate"] = q_correct_total / denom
    summary["overall_a_rate"] = a_correct_total / denom
    summary["overall_r_rate"] = r_correct_total / denom
    summary["overall_q_pass"] = f"{q_correct_total}/{denom}"
    summary["overall_a_pass"] = f"{a_correct_total}/{denom}"
    summary["overall_r_pass"] = f"{r_correct_total}/{denom}"

    return summary, all_details


# =========================
# 8) 主程序
# =========================

def main():
    df = pd.read_excel(INPUT_XLSX)
    if df.empty:
        raise RuntimeError("输入 Excel 为空")

    rows = df.to_dict(orient="records")
    total = len(rows)
    print(f"[INFO] 输入行数={total}，模型={MODELS}，Pass@{PASS_K}")

    summaries: List[Dict[str, Any]] = []
    details: List[Dict[str, Any]] = []

    # 外层并行：按题目并行（每题内部还有三模型并行 + 8 次调用）
    # 建议不要开太大，否则会爆并发
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(process_one_question, r): i for i, r in enumerate(rows)}
        done = 0
        for fut in as_completed(futs):
            done += 1
            try:
                srow, drows = fut.result()
                summaries.append(srow)
                details.extend(drows)
                print(f"[{done}/{total}] id={srow.get('id')} has_image={srow.get('has_image')} review={srow.get('need_human_review')}")
            except Exception as e:
                idx = futs[fut]
                rid = s(rows[idx].get(COL_ID) or (idx + 1))
                summaries.append({
                    "id": rid,
                    "need_human_review": True,
                    "review_reason": f"exception: {e}",
                })
                details.append({
                    "id": rid,
                    "model": "N/A",
                    "sample_k": 0,
                    "raw": f"exception: {e}",
                })
            time.sleep(SLEEP_BETWEEN_CALLS)

    df_sum = pd.DataFrame(summaries)
    df_det = pd.DataFrame(details)

    df_cfg = pd.DataFrame([{
        "INPUT_XLSX": INPUT_XLSX,
        "OUTPUT_XLSX": OUTPUT_XLSX,
        "MODELS": ",".join(MODELS),
        "PASS_K": PASS_K,
        "MAX_WORKERS": MAX_WORKERS,
        "TEMPERATURE": TEMPERATURE,
        "LOCAL_IMAGE_FOLDER": LOCAL_IMAGE_FOLDER,
        "ZERO_PASS_TRIGGER_MODELS": ZERO_PASS_TRIGGER_MODELS,
        "COL_ID": COL_ID,
        "COL_QUESTION": COL_QUESTION,
        "COL_IMAGE": COL_IMAGE,
        "COL_ANS_ORIG": COL_ANS_ORIG,
        "COL_ANS_LLM": COL_ANS_LLM,
        "COL_RAT_ORIG": COL_RAT_ORIG,
        "COL_RAT_LLM": COL_RAT_LLM,
    }])

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as w:
        df_sum.to_excel(w, sheet_name="Stage3_Summary", index=False)
        df_det.to_excel(w, sheet_name="Stage3_Details", index=False)
        df_cfg.to_excel(w, sheet_name="Stage3_Config", index=False)

    # 简单统计
    review_cnt = int(df_sum.get("need_human_review", False).sum()) if "need_human_review" in df_sum.columns else 0
    print(f"✅ Stage3 完成：{OUTPUT_XLSX}")
    print(f"[STAT] need_human_review = {review_cnt} / {len(df_sum)}")


if __name__ == "__main__":
    main()
