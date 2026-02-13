import os
import json
import time
import pandas as pd
from openai import OpenAI
import re
import concurrent.futures
from threading import Lock
import threading
import base64
from io import BytesIO
from urllib.parse import urlparse
from pathlib import Path
from PIL import Image


# 线程锁，用于保护共享资源
print_lock = Lock()

# 常用模型列表（仅供参考）
COMMON_MODELS = [
    "Qwen3-235B-A22B",
    "gpt-4o-0806-global",
    "gpt-4o-mini-0718-global",
    "claude-3-5-sonnet-20241022",
    "DeepSeek-R1-671B",
    "o1-preview-0912-global",
    "o1-mini-0912-global",
    "gpt-4-turbo",
    "claude-3-opus",
    "gemini-pro",
    "qwq-32b"
]

def _extract_local_image_path(image_url: str, folder: str = "image") -> str:
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
    
def get_openai_client():
    """为每个线程创建独立的OpenAI客户端"""
    return OpenAI(
        api_key="sk-7R6saeXf7d6fGdTOI9hUiB5q5aXKIRSZwpU5SgR9i2x58Ljo",
        base_url="https://api.302.ai/v1",
    )


def safe_print(*args, **kwargs):
    """线程安全的打印函数"""
    with print_lock:
        thread_id = threading.current_thread().ident
        print(f"[Thread-{thread_id}]", *args, **kwargs)


def retry_api_call(func, max_retries=2, retry_delay=3, *args, **kwargs):
    """API调用重试机制"""
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                safe_print(f"第 {attempt + 1} 次尝试调用API...")
                time.sleep(retry_delay)

            result = func(*args, **kwargs)

            if attempt > 0:
                safe_print(f"API调用在第 {attempt + 1} 次尝试后成功")

            return result

        except Exception as e:
            last_exception = e
            error_msg = str(e).lower()

            retryable_errors = [
                'rate limit', 'timeout', 'connection', 'server error',
                'busy', 'overloaded', 'unavailable', '429', '500',
                '502', '503', '504', '模型繁忙', '响应内容过短',
                '模型无法生成答案', '请稍后重试', 'Engine concurrency conflict!'
            ]

            is_retryable = any(error in error_msg for error in retryable_errors)

            if attempt < max_retries and is_retryable:
                safe_print(f"API调用失败 (第 {attempt + 1} 次): {e}")
                safe_print(f"这是可重试的错误，将在 {retry_delay} 秒后重试...")
            elif attempt < max_retries:
                safe_print(f"API调用失败 (第 {attempt + 1} 次): {e}")
                safe_print(f"这可能不是网络问题，但仍将重试...")
            else:
                safe_print(f"API调用最终失败，已重试 {max_retries} 次: {e}")
                break

    raise last_exception


def compare_answers_with_gpt4o(standard_answer, model_answer, question_text, attempt_num, question_num):
    """使用GPT-5比对答案是否正确"""
    if not standard_answer or not model_answer or model_answer in ["[空]", "[解答不完整]"]:
        return "无法比对"

    client = get_openai_client()

    safe_print(f"题号{question_num} 第{attempt_num}次 - 开始使用GPT-4o比对答案...")

    compare_prompt = f"""请比较以下两个答案是否相等或等价：

题目：{question_text}

标准答案：{standard_answer}
模型答案：{model_answer}

比较要求：
1. 判断两个数学答案在是否等价
2. 考虑不同的表示形式（如：2√3 和 2*sqrt(3)，1/2 和 0.5）
3. 考虑答案顺序（如果是多个答案）
4. 忽略格式差异，关注本质

请只回答以下选项之一：
- "正确" - 如果两个答案数学上等价
- "错误" - 如果两个答案数学上不等价
- "无法判断" - 如果无法确定是否等价

回答："""

    try:
        compare_response = client.chat.completions.create(
            model="qwen3-max-2025-09-23",
            messages=[{"role": "user", "content": compare_prompt}],
            temperature=0,
            max_tokens=50
        )

        result = compare_response.choices[0].message.content.strip()
        safe_print(f"题号{question_num} 第{attempt_num}次 - GPT-4o比对结果: {result}")

        # 标准化结果
        if "正确" in result:
            return "正确"
        elif "错误" in result:
            return "错误"
        else:
            return "无法判断"

    except Exception as e:
        safe_print(f"题号{question_num} 第{attempt_num}次 - GPT-4o比对失败: {e}")
        return "比对失败"


def call_model_with_retry(question, model_name, attempt_num, question_num):
    """调用指定模型的函数，修复深度思考问题"""
    client = get_openai_client()
    thread_id = threading.current_thread().ident

    # 构造数学问题提示
    math_prompt = f"""请解决以下数学问题，并将最终答案放在\\boxed{{}}中：

{question}

请提供详细的解题步骤，并在最后用\\boxed{{}}标出最终答案。"""

    start_time = time.time()
    safe_print(f"题号{question_num} 第{attempt_num}次 - 开始调用{model_name}模型...")

    full_response = ""
    thinking_content = ""

    # 根据模型类型选择不同的调用方式
    if "o1" in model_name.lower():
        # o1系列模型的特殊处理
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": math_prompt}],
        )
        full_response = response.choices[0].message.content
        thinking_content = ""

    elif "qwq" in model_name.lower():
        # QwQ模型只支持流式模式
        try:
            safe_print(f"使用QwQ流式模式...")
            stream = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": math_prompt}],
                stream=True,
                # temperature=0.7,
            )

            full_response = ""
            thinking_content = ""

            for chunk in stream:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                # 处理正常回复内容
                if hasattr(delta, "content") and delta.content:
                    full_response += delta.content

            safe_print(f"QwQ流式模式完成 - 回复内容: {len(full_response)}字符")

        except Exception as e:
            safe_print(f"QwQ流式模式失败: {e}")
            raise e

    elif "qwen" in model_name.lower():
        # Qwen模型支持流式输出和思考模式 - 使用正确的参数
        try:
            safe_print(f"使用Qwen深度思考模式...")
            stream = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": math_prompt}],
                stream=True,
                extra_body={"enable_thinking": False}  # 千问模型的思考参数
            )

            full_response = ""
            thinking_content = ""
            is_answering = False

            for chunk in stream:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                # 处理思考内容 - 千问模型使用 reasoning_content 字段
                if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                    thinking_content += delta.reasoning_content
                    if not is_answering:
                        is_answering = False  # 还在思考阶段

                # 处理正常回复内容
                if hasattr(delta, "content") and delta.content:
                    if not is_answering:
                        safe_print(f"开始生成回复，思考过程总长度: {len(thinking_content)}")
                        is_answering = True
                    full_response += delta.content

            safe_print(f"Qwen思考模式完成 - 思考内容: {len(thinking_content)}字符, 回复内容: {len(full_response)}字符")

        except Exception as e:
            # 如果思考模式失败，使用普通流式模式
            safe_print(f"Qwen思考模式失败，使用普通流式模式: {e}")
            stream = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": math_prompt}],
                stream=True,
            )

            full_response = ""
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content

            thinking_content = ""

    elif "deepseek-r1" in model_name.lower():
        # DeepSeek-R1模型支持流式输出和思考模式
        try:
            safe_print(f"使用DeepSeek-R1深度思考模式...")
            stream = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": math_prompt}],
                stream=True,
            )

            full_response = ""
            thinking_content = ""
            is_answering = False

            for chunk in stream:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                # 处理思考内容 - DeepSeek-R1可能使用不同的字段名
                thinking_fields = ['reasoning_content', 'thinking_content', 'thought_content']
                for field in thinking_fields:
                    if hasattr(delta, field) and getattr(delta, field) is not None:
                        thinking_content += getattr(delta, field)
                        if not is_answering:
                            is_answering = False  # 还在思考阶段
                        break

                # 处理正常回复内容
                if hasattr(delta, "content") and delta.content:
                    if not is_answering:
                        safe_print(f"开始生成回复，思考过程总长度: {len(thinking_content)}")
                        is_answering = True
                    full_response += delta.content

            safe_print(
                f"DeepSeek-R1思考模式完成 - 思考内容: {len(thinking_content)}字符, 回复内容: {len(full_response)}字符")

        except Exception as e:
            # 如果思考模式失败，使用普通流式模式
            safe_print(f"DeepSeek-R1思考模式失败，使用普通流式模式: {e}")
            try:
                stream = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": math_prompt}],
                    stream=True,
                )

                full_response = ""
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content

                thinking_content = ""
                safe_print(f"DeepSeek-R1普通流式模式完成 - 回复内容: {len(full_response)}字符")

            except Exception as e2:
                # 如果流式也失败，使用非流式模式
                safe_print(f"DeepSeek-R1流式模式失败，使用非流式模式: {e2}")
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": math_prompt}],
                )
                full_response = response.choices[0].message.content
                thinking_content = ""

    else:
        # 其他模型的标准调用
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": math_prompt}],
        )
        full_response = response.choices[0].message.content
        thinking_content = ""

    end_time = time.time()
    duration = end_time - start_time

    # 检测响应时间过短和错误响应
    if duration < 2:
        safe_print(f"题号{question_num} 第{attempt_num}次 - 警告：调用时间过短({duration:.2f}秒)")

    if len(full_response.strip()) < 50:
        safe_print(f"题号{question_num} 第{attempt_num}次 - 警告：响应内容过短({len(full_response)}字符)")
        raise Exception(f"响应内容过短，可能是API调用异常")

    # 检测错误响应内容
    error_keywords = [
        "concurrency conflict", "engine", "error", "failed",
        "busy", "unavailable", "limit", "quota", "请稍后重试"
    ]

    if any(keyword in full_response.lower() for keyword in error_keywords):
        safe_print(f"题号{question_num} 第{attempt_num}次 - 检测到错误响应: {full_response[:100]}...")
        raise Exception(f"模型返回错误信息: {full_response[:100]}")

    safe_print(
        f"题号{question_num} 第{attempt_num}次 - {model_name}调用完成，耗时: {duration:.2f}秒，响应长度: {len(full_response)}字符，思考长度: {len(thinking_content)}字符")

    return {
        'full_response': full_response.strip(),
        'thinking_content': thinking_content,
        'duration': duration,
        'model_name': model_name
    }


def call_gpt4o_with_retry(response, question_text, attempt_num, question_num):
    """使用大模型提取答案"""
    client = get_openai_client()

    if len(response.strip()) < 50:
        safe_print(f"题号{question_num} 第{attempt_num}次 - 警告：输入的解题过程过短")

    safe_print(f"题号{question_num} 第{attempt_num}次 - 开始调用GPT-4o提取答案...")

    extract_prompt = f"""从以下数学解答中提取最终答案：

{response}

要求：
1. 找到 \\boxed{{}} 中的内容，提取其中的数学表达式
2. 如果没有 \\boxed{{}}，提取最终的数值答案或结论
3. 去掉所有格式符号：\\boxed{{}}、$$、\\(\\)等
4. 如果有多个答案，用逗号分隔在一行内
5. 保持数学符号如 \\sqrt{{}}、\\frac{{}}等

只返回纯净的数学答案，不要任何格式包装。

答案："""

    extract_response = client.chat.completions.create(
        model="qwen3-max-2025-09-23",
        messages=[{"role": "user", "content": extract_prompt}],
        temperature=0,
        max_tokens=200
    )

    result = extract_response.choices[0].message.content.strip()
    safe_print(f"题号{question_num} 第{attempt_num}次 - GPT-5提取答案完成: {result}")

    return result


def extract_answer_with_gpt4o(response, question_text, attempt_num, question_num):
    """使用大模型取答案"""
    if not response:
        return ""

    try:
        safe_print(f"题号{question_num} 第{attempt_num}次 - 使用大模型提取答案...")
        extracted_answer = retry_api_call(
            call_gpt4o_with_retry,
            max_retries=2,
            retry_delay=3,
            response=response,
            question_text=question_text,
            attempt_num=attempt_num,
            question_num=question_num
        )

        if "解答不完整" in extracted_answer:
            safe_print(f"题号{question_num} 第{attempt_num}次 - 大模型检测到解答不完整")
            return "[解答不完整]"
        elif "无法确定" in extracted_answer:
            safe_print(f"题号{question_num} 第{attempt_num}次 - 大模型无法确定答案")
            return ""

        cleaned_answer = extracted_answer.strip()
        return cleaned_answer if cleaned_answer else ""

    except Exception as e:
        safe_print(f"题号{question_num} 第{attempt_num}次 - 大模型提取答案失败: {e}")
        return ""

# === [REPLACE] 单次询问：自动根据是否含图 & 模型是否支持vision 选择调用方式 ===
def ask_model_for_math_single(question, model_name, attempt_num, question_num, standard_answer="", image_path=None):
    thread_id = threading.current_thread().ident
    has_image = bool(image_path) and os.path.exists(image_path)
    use_vision = has_image

    mode_tag = "VISION" if use_vision else "TEXT"
    safe_print(f"题号{question_num} 第{attempt_num}次 - 开始处理({model_name}, {mode_tag})...")

    try:
        if use_vision:
            result = retry_api_call(
                call_model_with_retry_vision,
                max_retries=2,
                retry_delay=5,
                question=question,
                model_name=model_name,
                attempt_num=attempt_num,
                question_num=question_num,
                image_path=image_path
            )
        else:
            result = retry_api_call(
                call_model_with_retry,
                max_retries=2,
                retry_delay=5,
                question=question,
                model_name=model_name,
                attempt_num=attempt_num,
                question_num=question_num
            )

        full_response = result['full_response']
        thinking_content = result.get('thinking_content', "")
        duration = result['duration']

        # 提取最终答案
        answer = extract_answer_with_gpt4o(full_response, question, attempt_num, question_num)

        # 比对
        comparison_result = "无标准答案"
        if standard_answer and standard_answer.strip():
            comparison_result = retry_api_call(
                compare_answers_with_gpt4o,
                max_retries=1,
                retry_delay=2,
                standard_answer=standard_answer,
                model_answer=answer,
                question_text=question,
                attempt_num=attempt_num,
                question_num=question_num
            )

        ans_display = answer if answer else "[空]"
        safe_print(
            f"题号{question_num} 第{attempt_num}次求解完成({model_name},{mode_tag}) - 耗时: {duration:.2f}s - 答案: {ans_display} - 比对: {comparison_result}")

        if thinking_content:
            safe_print(f"题号{question_num} 第{attempt_num}次 - 思考过程长度: {len(thinking_content)} 字符")

        return {
            'attempt': attempt_num,
            'answer': answer,
            'response': full_response,
            'thinking_content': thinking_content,
            'thinking_length': len(thinking_content),
            'duration': duration,
            'thread_id': thread_id,
            'model_name': model_name,
            'comparison_result': comparison_result
        }

    except Exception as e:
        safe_print(f"题号{question_num} 第{attempt_num}次求解最终失败({model_name},{mode_tag}): {e}")
        return {
            'attempt': attempt_num,
            'answer': "",
            'response': f"请求失败: {str(e)}",
            'thinking_content': "",
            'thinking_length': 0,
            'duration': 0,
            'thread_id': thread_id,
            'model_name': model_name,
            'comparison_result': "处理失败"
        }


# === [REPLACE] 解析函数：兼容英文字段并提取本地图片路径 ===
def parse_question_data(data):
    """解析不同格式的题目数据 - 兼容 {question/image/answer} 并生成 图片路径"""
    import pandas as pd

    def safe_str_convert(value):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        return str(value).strip()
    question_num = (
        data.get("row") or
        data.get("question_id") or
        data.get("题号") or
        data.get("id") or
        0
    )


    # 题干
    question_text = safe_str_convert(
        data.get("prompt", "") or data.get("题干", "") or data.get("question", "")
    )

    # 过程（若英文数据无解题过程则为空）
    process = safe_str_convert(
        data.get("rationale", "") or data.get("过程", "")
    )

    # 答案
    answer = safe_str_convert(
        data.get("answer", "") or data.get("答案", "")
    )

    # 图片路径（从 image / rationale_image 提取URL → 本地 image/xxx.jpg）
    image_path = ""
    for k in ("图片路径", "image", "rationale_image"):
        image_path = _extract_local_image_path(safe_str_convert(data.get(k, "")))
        if image_path:
            break

    return {
        "题号": question_num,
        "题干": question_text,
        "过程": process,
        "答案": answer,
        "图片路径": image_path
    }



def solve_question_parallel(question_data, models, num_attempts, pass_threshold, max_workers=2,
                            delay_between_requests=1):
    """并行求解单个问题 - 修正合格判断逻辑为 <= k，并为每个模型单独判断合格"""
    parsed_data = parse_question_data(question_data)
    question_num = parsed_data["题号"]
    question_text = parsed_data["题干"]
    original_process = parsed_data["过程"]
    original_answer = parsed_data["答案"]
    image_path = parsed_data.get("图片路径", "")
    safe_print(f"\n{'=' * 60}")
    safe_print(f"开始并行处理题号 {question_num}")
    safe_print(f"题目: {question_text[:150]}...")
    safe_print(f"使用模型: {', '.join(models)}")
    safe_print(f"每个模型尝试次数: {num_attempts}")
    safe_print(f"合格标准: 正确次数 <= {pass_threshold}")
    safe_print(f"并行线程数: {max_workers}")
    if original_process:
        safe_print(f"原始过程: {original_process[:100]}...")
    if original_answer:
        safe_print(f"标准答案: {original_answer}")
    else:
        safe_print(f"无标准答案，不进行比对")
    safe_print(f"{'=' * 60}")

    # 为每个模型创建多次尝试的任务
    tasks = []
    for model in models:
        for attempt in range(1, num_attempts + 1):
            tasks.append((question_text, model, attempt, question_num, original_answer, image_path))

    # 使用线程池并行执行
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = []
        batch_size = max_workers

        for batch_start in range(0, len(tasks), batch_size):
            batch_end = min(batch_start + batch_size, len(tasks))

            futures = []
            for i in range(batch_start, batch_end):
                question_text, model, attempt, question_num, standard_answer, image_path = tasks[i]
                future = executor.submit(
                    ask_model_for_math_single,
                    question_text, model, attempt, question_num,
                    standard_answer, image_path
                )
                futures.append(future)
                if delay_between_requests > 0 and i < batch_end - 1:
                    time.sleep(delay_between_requests)

            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    safe_print(f"获取结果时出错: {e}")
                    results.append({
                        'attempt': len(results) + 1,
                        'answer': "",
                        'response': f"处理失败: {str(e)}",
                        'thinking_content': "",
                        'thinking_length': 0,
                        'duration': 0,
                        'thread_id': threading.current_thread().ident,
                        'model_name': 'unknown',
                        'comparison_result': "处理失败"
                    })

            if batch_end < len(tasks):
                safe_print(f"批次完成，等待 {delay_between_requests * 2} 秒后继续...")
                time.sleep(delay_between_requests * 2)

    # 按模型和attempt排序
    results.sort(key=lambda x: (x['model_name'], x['attempt']))
    # 1. 构建答案汇总表 - 格式：题号、题目、原始过程、标准答案、各次答案、每个模型的合格状态
    answer_summary_row = {
        "题号": question_num,
        "题目": question_text,
        "原始过程": original_process,
        "标准答案": original_answer
    }
    # 为每个模型的每次尝试添加答案列
    for result in results:
        model = result['model_name']
        attempt = result['attempt']
        col_name = f"答案_{model}_第{attempt}次"
        answer_summary_row[col_name] = result['answer'] if result['answer'] else "[空]"

    # 计算每个模型的合格状态
    model_qualifications = {}
    if original_answer and original_answer.strip():
        # 有标准答案，为每个模型单独计算正确次数
        for model in models:
            model_results = [r for r in results if r['model_name'] == model]
            correct_count = sum(1 for r in model_results if r['comparison_result'] == "正确")
            is_qualified = "合格" if correct_count <= pass_threshold else "不合格"
            qualification_detail = f"{is_qualified}({correct_count}/{len(model_results)})"
            model_qualifications[model] = qualification_detail

            # 添加到答案汇总行
            answer_summary_row[f"{model}_合格状态"] = qualification_detail
    else:
        # 无标准答案
        for model in models:
            model_qualifications[model] = "无标准答案"
            answer_summary_row[f"{model}_合格状态"] = "无标准答案"

    # 2. 构建统计信息表 - 包含性能数据和比对统计
    stats_rows = []
    for result in results:
        stats_rows.append({
            "题号": question_num,
            "模型": result['model_name'],
            "尝试次数": result['attempt'],
            "耗时(秒)": round(result['duration'], 2),
            "思考长度": result['thinking_length'],
            "线程ID": result.get('thread_id', 'unknown'),
            "答案": result['answer'] if result['answer'] else "[空-需人工提取]",
            "比对结果": result['comparison_result']
        })

    # 3. 构建解题过程表 - 完整的解题和思考内容
    process_rows = []
    for result in results:
        process_rows.append({
            "题号": question_num,
            "题目": question_text,
            "模型": result['model_name'],
            "尝试次数": result['attempt'],
            "答案": result['answer'] if result['answer'] else "[空-需人工提取]",
            "比对结果": result['comparison_result'],
            "解题过程": result['response'],
            "思考过程": result['thinking_content'],
            "耗时(秒)": round(result['duration'], 2),
            "思考长度": result['thinking_length'],
            "线程ID": result.get('thread_id', 'unknown')
        })

    # 统计比对结果
    if original_answer and original_answer.strip():
        safe_print(f"\n题号 {question_num} 各模型合格状态:")
        for model, qualification in model_qualifications.items():
            safe_print(f"  {model}: {qualification}")
    else:
        safe_print(f"\n题号 {question_num} 无标准答案，跳过比对统计")

    safe_print(f"\n题号 {question_num} 完成所有求解:")
    for result in results:
        model = result['model_name']
        attempt = result['attempt']
        answer_display = result['answer'] if result['answer'] and result['answer'] != "[空]" else "[空-需人工提取]"
        comparison_display = result['comparison_result']
        safe_print(
            f"  {model}_第{attempt}次: {answer_display} (耗时: {result['duration']:.2f}s, 思考: {result['thinking_length']}字符, 比对: {comparison_display})")
    return answer_summary_row, stats_rows, process_rows


def calculate_overall_pass_rate(answer_summary_results, pass_threshold, models):
    """计算整体合格率统计 - 只包含每个模型的统计"""
    if not answer_summary_results:
        return {}

    total_questions = len(answer_summary_results)

    # 每个模型的统计
    model_stats = {}
    for model in models:
        model_stats[model] = {
            'passed': 0,
            'failed': 0,
            'no_standard': 0
        }

    question_details = []

    for row in answer_summary_results:
        question_num = row['题号']

        # 每个模型的统计
        model_statuses = {}
        for model in models:
            model_qual_key = f"{model}_合格状态"
            if model_qual_key in row:
                model_qualification = row[model_qual_key]
                if model_qualification == "无标准答案":
                    model_stats[model]['no_standard'] += 1
                    model_statuses[model] = "无标准答案"
                elif "合格" in model_qualification:
                    model_stats[model]['passed'] += 1
                    model_statuses[model] = "合格"
                else:
                    model_stats[model]['failed'] += 1
                    model_statuses[model] = "不合格"

        question_detail = {
            '题号': question_num
        }

        # 添加每个模型的状态
        for model in models:
            question_detail[f'{model}_状态'] = model_statuses.get(model, "未知")

        question_details.append(question_detail)

    # 计算每个模型的合格率
    model_pass_rates = {}
    for model in models:
        model_with_standard = model_stats[model]['passed'] + model_stats[model]['failed']
        if model_with_standard > 0:
            model_pass_rates[model] = f"{model_stats[model]['passed'] / model_with_standard * 100:.1f}%"
        else:
            model_pass_rates[model] = "N/A (无标准答案)"

    return {
        'total_questions': total_questions,
        'model_stats': model_stats,
        'model_pass_rates': model_pass_rates,
        'question_details': question_details,
        'pass_threshold': pass_threshold
    }

# === [ADD] 视觉调用版本：把图片与文本一起发给模型 ===
def call_model_with_retry_vision(question, model_name, attempt_num, question_num, image_path):
    client = get_openai_client()

    math_prompt = f"""请解决以下数学/逻辑题，并将最终答案放在\\boxed{{}}中：
若图片中包含关键信息，请结合图片与文字一并作答。

题目：
{question}

请提供清晰步骤，并在最后用\\boxed{{}}标出最终答案。"""

    start_time = time.time()
    safe_print(f"题号{question_num} 第{attempt_num}次 - [VISION] 调用 {model_name}，图片：{image_path}")

    # 构造多模态消息（文本 + image_url dataURI）
    data_url = _build_data_url_with_pillow(image_path)
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": math_prompt},
            {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}}
        ]
    }]

    # 大多模型都能用标准 chat.completions；
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        stream=False
    )
    full_response = (response.choices[0].message.content or "")
    thinking_content = ""
    duration = time.time() - start_time

    if duration < 2:
        safe_print(f"题号{question_num} 第{attempt_num}次 - 警告：调用时间过短({duration:.2f}秒)")
    if len(full_response.strip()) < 50:
        safe_print(f"题号{question_num} 第{attempt_num}次 - 警告：响应内容过短({len(full_response)}字符)")
        raise Exception("响应内容过短，可能是API调用异常")

    safe_print(
        f"题号{question_num} 第{attempt_num}次 - [VISION]{model_name} 调用完成，耗时: {duration:.2f}秒，响应长度: {len(full_response)}字符")
    return {
        'full_response': full_response.strip(),
        'thinking_content': thinking_content,
        'duration': duration,
        'model_name': model_name
    }


def select_models():
    """选择要使用的模型 - 支持自定义输入"""
    print("\n常用模型参考列表:")
    for i, model in enumerate(COMMON_MODELS, 1):
        print(f"{i:2d}. {model}")

    print("\n请输入要使用的模型名称:")
    print("- 可以直接输入模型名称，多个模型用逗号分隔")
    print("- 也可以输入上面列表中的数字，多个数字用逗号分隔")
    print("- 例如: Qwen3-235B-A22B,gpt-4o-0806-global")
    print("- 或者: 1,2,3")

    selection = input("\n输入选择: ").strip()

    if not selection:
        print("未输入任何内容，使用默认模型: Qwen3-235B-A22B")
        return ["Qwen3-235B-A22B"]

    selected_models = []

    # 按逗号分割输入
    choices = [choice.strip() for choice in selection.split(',')]

    for choice in choices:
        if choice.isdigit():
            # 如果是数字，从常用模型列表中选择
            index = int(choice) - 1
            if 0 <= index < len(COMMON_MODELS):
                model_name = COMMON_MODELS[index]
                if model_name not in selected_models:
                    selected_models.append(model_name)
                    print(f"✓ 已选择: {model_name}")
            else:
                print(f"✗ 无效数字: {choice} (超出范围)")
        else:
            # 如果是字符串，直接作为模型名称
            if choice and choice not in selected_models:
                selected_models.append(choice)
                print(f"✓ 已选择: {choice}")

    if not selected_models:
        print("未选择任何有效模型，使用默认模型: Qwen3-235B-A22B")
        selected_models = ["Qwen3-235B-A22B"]

    print(f"\n最终选择的模型: {', '.join(selected_models)}")

    # 确认选择
    confirm = input("确认使用这些模型？(y/n): ").strip().lower()
    if confirm != 'y':
        print("重新选择模型...")
        return select_models()

    return selected_models


def process_math_questions_parallel(input_file, output_file, models, num_attempts, pass_threshold, start_index=0,
                                    max_workers=2, delay_between_requests=1):
    """并行处理数学问题 - 修正合格判断逻辑为 <= k，并为每个模型单独判断合格"""

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    questions_data = []
    for line in lines:
        line = line.strip()
        if line:
            try:
                data = json.loads(line)
                questions_data.append(data)
            except json.JSONDecodeError as e:
                print(f"解析JSON出错: {e}, 行内容: {line}")
                continue
    for idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"解析JSON出错: {e}, 行内容: {line}")
            continue

        # === [ADD] 若无 row/题号，则用行号生成（1-based）
        if not raw.get("row") and not raw.get("题号"):
            raw["row"] = idx + 1
    questions_to_process = questions_data[start_index:]

    print(f"开始并行处理 {len(questions_to_process)} 个问题")
    print(f"使用模型: {', '.join(models)}")
    print(f"每个模型尝试次数: {num_attempts}")
    print(f"合格标准: 正确次数 <= {pass_threshold}")
    print(f"并行线程数: {max_workers}")
    print(f"请求间隔: {delay_between_requests}秒")

    # 分别收集三种类型的数据
    answer_summary_results = []  # 答案汇总表数据
    all_stats_rows = []  # 统计信息表数据
    all_process_rows = []  # 解题过程表数据

    for i, question_data in enumerate(questions_to_process):
        print(f"\n处理进度: {i + 1}/{len(questions_to_process)}")

        # 获取三种类型的数据
        answer_summary_row, stats_rows, process_rows = solve_question_parallel(
            question_data, models, num_attempts, pass_threshold, max_workers, delay_between_requests
        )

        # 分别添加到对应的列表
        answer_summary_results.append(answer_summary_row)
        all_stats_rows.extend(stats_rows)
        all_process_rows.extend(process_rows)

        # 实时保存 - 创建三个工作表
        df_answer_summary = pd.DataFrame(answer_summary_results)
        df_stats = pd.DataFrame(all_stats_rows)
        df_process = pd.DataFrame(all_process_rows)

        # 计算整体合格率统计
        overall_stats = calculate_overall_pass_rate(answer_summary_results, pass_threshold, models)
        df_overall_stats = pd.DataFrame(overall_stats.get('question_details', []))

        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # 工作表1: 答案汇总 - 包含每个模型的合格状态
            df_answer_summary.to_excel(writer, sheet_name='答案汇总', index=False)

            # 工作表2: 统计信息 - 性能分析数据
            df_stats.to_excel(writer, sheet_name='统计信息', index=False)

            # 工作表3: 解题过程 - 完整的解题和思考内容
            df_process.to_excel(writer, sheet_name='解题过程', index=False)

            # 工作表4: 模型统计 (如果有数据)
            if not df_overall_stats.empty:
                df_overall_stats.to_excel(writer, sheet_name='模型统计', index=False)

        print(f"已完成 {i + 1}/{len(questions_to_process)} 个问题，结果已保存")

        # 显示当前统计
        if overall_stats:
            print("各模型合格率:")
            for model in models:
                model_rate = overall_stats['model_pass_rates'].get(model, "N/A")
                model_stat = overall_stats['model_stats'].get(model, {})
                passed = model_stat.get('passed', 0)
                total = passed + model_stat.get('failed', 0)
                print(f"  {model}: {model_rate} ({passed}/{total})")

        if i < len(questions_to_process) - 1:
            print(f"等待 {delay_between_requests * 3} 秒后处理下一题...")
            time.sleep(delay_between_requests * 3)

    print(f"\n所有问题处理完成！结果已保存到 {output_file}")

    # 最终统计报告
    final_stats = calculate_overall_pass_rate(answer_summary_results, pass_threshold, models)
    if final_stats:
        print(f"\n📊 最终统计报告:")
        print(f"总题目数: {final_stats['total_questions']}")
        print(f"合格标准: 正确次数 <= {pass_threshold}")

        print(f"\n各模型详细统计:")
        for model in models:
            model_rate = final_stats['model_pass_rates'].get(model, "N/A")
            model_stat = final_stats['model_stats'].get(model, {})
            passed = model_stat.get('passed', 0)
            failed = model_stat.get('failed', 0)
            no_standard = model_stat.get('no_standard', 0)
            total_with_standard = passed + failed
            print(f"  {model}:")
            print(f"    合格率: {model_rate} ({passed}/{total_with_standard})")
            print(f"    合格: {passed}, 不合格: {failed}, 无标准答案: {no_standard}")

    print("\n表格结构说明:")
    print("📊 答案汇总表: 题号、题目、原始过程、标准答案、各次答案、各模型合格状态")
    print("📈 统计信息表: 题号、模型、尝试次数、耗时、思考长度、答案、比对结果")
    print("📝 解题过程表: 完整的解题过程和思考内容")
    print("📋 模型统计表: 每题各模型的状态汇总")
    print("\n注意：")
    print("- 标记为[空]的答案需要根据解题过程人工提取")
    print("- 只有包含标准答案的题目才参与合格判断")
    print("- 没有标准答案的题目显示'无标准答案'")
    print("- 合格标准：正确次数 <= k (错误次数较少才合格)")
    print("- 每个模型单独判断合格")
    return answer_summary_results


# 使用示例
if __name__ == "__main__":
    print("数学题求解器 - 多模型并行版本 (合格标准: 正确次数 <= k)")
    print("支持自定义模型输入和多种JSONL格式")
    print("答案汇总表格式: 题号、题目、原始过程、标准答案、各次答案、各模型合格状态")
    print("=" * 60)

    choice = input("选择模式:\n1. 测试单题\n2. 处理完整文件\n请输入选择 (1/2): ")

    if choice == "1":
        # 测试单题
        models = select_models()

        num_attempts = input("每个模型尝试几次？(默认3): ")
        try:
            num_attempts = int(num_attempts) if num_attempts.strip() else 3
        except ValueError:
            num_attempts = 3

        pass_threshold = input(f"合格标准：正确次数最多几次？(默认1，必须小于{num_attempts}): ")
        try:
            pass_threshold = int(pass_threshold) if pass_threshold.strip() else 1
            if pass_threshold >= num_attempts:
                print(
                    f"警告：合格标准({pass_threshold})不能大于等于尝试次数({num_attempts})，自动调整为{num_attempts - 1}")
                pass_threshold = num_attempts - 1
        except ValueError:
            pass_threshold = 1

        max_workers = input("输入并行线程数 (推荐4-8): ")
        try:
            max_workers = int(max_workers) if max_workers.strip() else 4
        except ValueError:
            max_workers = 4

        delay = input("输入请求间隔秒数 (推荐2-3): ")
        try:
            delay = float(delay) if delay.strip() else 2
        except ValueError:
            delay = 2

        # 测试题目 - 包含标准答案
        test_question = {
            "题号": 1,
            "题干": "Find the singular values of the matrix $ A = \\begin{bmatrix} \\sqrt{3} & 2 & -\\sqrt{2} & 1 & 0 \\\\ 1 & \\sqrt{5} & 0 & -1 & 3 \\\\ -2 & 1 & 2\\sqrt{2} & \\sqrt{3} & -1 \\\\ 0 & -3 & 1 & \\sqrt{7} & 2 \\\\ 4 & 0 & \\sqrt{6} & -2 & \\sqrt{10} \\end{bmatrix} $.",
            "答案": "2√6, √19, √17, 4, 2√6"  # 示例标准答案
        }

        # 获取三种类型的数据
        answer_summary_row, stats_rows, process_rows = solve_question_parallel(
            test_question, models, num_attempts, pass_threshold, max_workers, delay
        )

        # 保存结果到工作表
        df_answer_summary = pd.DataFrame([answer_summary_row])
        df_stats = pd.DataFrame(stats_rows)
        df_process = pd.DataFrame(process_rows)
        output_file = "test_multi_model.xlsx"

        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df_answer_summary.to_excel(writer, sheet_name='答案汇总', index=False)
            df_stats.to_excel(writer, sheet_name='统计信息', index=False)
            df_process.to_excel(writer, sheet_name='解题过程', index=False)

        print(f"\n测试完成！结果已保存到: {output_file}")
        print("\n表格结构说明:")
        print("📊 答案汇总表: 题号、题目、原始过程、标准答案、各次答案、各模型合格状态")
        print("📈 统计信息表: 题号、模型、尝试次数、耗时、思考长度、答案、比对结果")
        print("📝 解题过程表: 完整的解题过程和思考内容")

    elif choice == "2":
        # 处理完整文件
        input_json = input("输入JSONL文件路径 (默认: 题目.jsonl): ").strip()
        if not input_json:
            input_json = "题目.jsonl"

        output_excel = input("输出Excel文件路径 (默认: 题目答案.xlsx): ").strip()
        if not output_excel:
            output_excel = "题目答案.xlsx"

        if os.path.exists(input_json):
            models = select_models()

            num_attempts = input("每个模型尝试几次？(默认3): ")
            try:
                num_attempts = int(num_attempts) if num_attempts.strip() else 3
            except ValueError:
                num_attempts = 3

            pass_threshold = input(f"合格标准：正确次数最多几次？(默认1，必须小于{num_attempts}): ")
            try:
                pass_threshold = int(pass_threshold) if pass_threshold.strip() else 1
                if pass_threshold >= num_attempts:
                    print(
                        f"警告：合格标准({pass_threshold})不能大于等于尝试次数({num_attempts})，自动调整为{num_attempts - 1}")
                    pass_threshold = num_attempts - 1
            except ValueError:
                pass_threshold = 1

            max_workers = input("输入并行线程数 (推荐4-8): ")
            try:
                max_workers = int(max_workers) if max_workers.strip() else 4
            except ValueError:
                max_workers = 4

            delay = input("输入请求间隔秒数 (推荐2-3): ")
            try:
                delay = float(delay) if delay.strip() else 2
            except ValueError:
                delay = 2

            start_index = input("从第几个问题开始？(默认0): ")
            try:
                start_index = int(start_index) if start_index.strip() else 0
            except ValueError:
                start_index = 0

            print(f"\n配置确认:")
            print(f"模型: {', '.join(models)}")
            print(f"每个模型尝试次数: {num_attempts}")
            print(f"合格标准: 正确次数 <= {pass_threshold}")
            print(f"并行线程数: {max_workers}")
            print(f"请求间隔: {delay}秒")
            print(f"开始位置: {start_index}")

            confirm = input("确认开始处理？(y/n): ")
            if confirm.lower() == 'y':
                process_math_questions_parallel(
                    input_json, output_excel, models, num_attempts, pass_threshold,
                    start_index, max_workers, delay
                )
            else:
                print("已取消处理")
        else:
            print(f"输入文件 {input_json} 不存在")
    else:
        print("无效选择")
