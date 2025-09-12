import re
import json

def json_load_llm_answer(answer):
    """
    解析 LLM 的 JSON 回答，移除 ```json ... ``` 包裹，并去除空行。
    """
    # 移除 ```json 和 ```
    answer = re.sub(r'```json\s*', '', answer, flags=re.IGNORECASE)
    answer = re.sub(r'\s*```', '', answer)

    # 去除空行
    lines = answer.splitlines()
    non_empty_lines = [line for line in lines if line.strip() != '']
    cleaned_answer = '\n'.join(non_empty_lines)

    return json.loads(cleaned_answer)

if __name__ == '__main__':
    answer = """


```json
{
    "kernel-dev": {
        "probability": 0.9,
        "reason": "Discusses kernel tainting, debugging, and developer practices for bug reporting."
    },
    "kernel-config": {
        "probability": 0.4,
        "reason": "Tainting relates to module loading, but not direct kernel parameter configuration."
    },
    "is_version_specific": {
        "value": false,
        "reason": "Taint concept applies generally; no specific kernel version is required."
    }
}
```
"""
    print(json_load_llm_answer(answer))
