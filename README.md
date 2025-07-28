# LKQABench

## Prompts

我们的方法使用了众多大模型，方法具有通用性，下面给出具体的 prompt 细节。

### LLM驱动的深层语义过滤

该方法基于 LLM 进行原始语料的过滤，保证筛选后的语料与 Linux 内核开发高度相关。整体采取宁可遗漏、绝不误纳的严格策略，以高精确、低召回为原则。针对不同来源的原始语料，prompt 略有差别，具体如下。

#### 来源一：技术问答社区

~~~
You will be given a question and its answer from a Q&A community. You need to classify the content specifically focusing on Linux Kernel development relevance.

Please perform the following classification tasks with high precision:

1. Determine whether the **question is related to Linux Kernel development** (i.e., topics that Linux Kernel developers care about). Provide:
   - a probability score (a float between 0 and 1),
   - a short English explanation (reason) within 50 words.
   
   Guidelines for kernel development relevance:
   - High relevance (0.7-1.0): Topics involving kernel source code modification, kernel module development, kernel API usage, deep kernel mechanism understanding, kernel subsystem architecture, or complex kernel debugging issues.
   - Medium relevance (0.3-0.6): Topics requiring understanding of kernel internals but not direct development, kernel performance analysis, or tracing kernel behavior.
   - Low relevance (0.0-0.2): General Linux user questions about applications, desktop environments, or basic system administration that don't require understanding kernel mechanisms.
   
2. Determine whether the **question is related to Linux Kernel configuration** (i.e., configuring kernel parameters by users or developers). Provide:
   - a probability score (a float between 0 and 1),
   - a short English explanation (reason) within 50 words.
   
   Guidelines for kernel configuration relevance:
   - Focus on kernel build configuration (Kconfig, .config), kernel boot parameters, sysctl parameters, /proc or /sys filesystem configurations.
   - Distinguish between user-space configuration (which scores lower) and actual kernel parameter configuration.

Output format:  
Return **only** the following JSON structure:
```json
{
    "kernel-dev": {
        "probability": <float between 0 and 1>,
        "reason": "<short English explanation, max 50 words>"
    },
    "kernel-config": {
        "probability": <float between 0 and 1>,
        "reason": "<short English explanation, max 50 words>"
    }
}
```

Important notes:
- Clearly distinguish between regular Linux user questions and kernel development questions. A question about how to use a Linux application should receive a very low kernel-dev score.
- Topics involving direct kernel code, kernel data structures, kernel debugging, or kernel module development should receive high kernel-dev scores.
- Questions about general Linux usage troubleshooting should score low unless they involve kernel-specific mechanisms.
- Evaluate "kernel-dev" and "kernel-config" independently.
- Ensure the JSON format is strictly followed. No extra text.
~~~

#### 来源二：内核邮件列表

~~~
You will be given an email from a mailing list of a Linux kernel community. You need to classify the content specifically focusing on Linux Kernel development relevance.

Please perform the following classification tasks with high precision:

1. Determine whether the **email is related to Linux Kernel development** (i.e., topics that Linux Kernel developers care about). Provide:
   - a probability score (a float between 0 and 1),
   - a short English explanation (reason) within 50 words.
   
   Guidelines for kernel development relevance:
   - High relevance (0.7-1.0): Topics involving kernel source code modification, kernel module development, kernel API usage, deep kernel mechanism understanding, kernel subsystem architecture, or complex kernel debugging issues.
   - Medium relevance (0.3-0.6): Topics requiring understanding of kernel internals but not direct development, kernel performance analysis, or tracing kernel behavior.
   - Low relevance (0.0-0.2): General Linux user questions about applications, desktop environments, or basic system administration that don't require understanding kernel mechanisms.
   
2. Determine whether the **email is related to Linux Kernel configuration** (i.e., configuring kernel parameters by users or developers). Provide:
   - a probability score (a float between 0 and 1),
   - a short English explanation (reason) within 50 words.
   
   Guidelines for kernel configuration relevance:
   - Focus on kernel build configuration (Kconfig, .config), kernel boot parameters, sysctl parameters, /proc or /sys filesystem configurations.
   - Distinguish between user-space configuration (which scores lower) and actual kernel parameter configuration.

Output format:  
Return **only** the following JSON structure:
```json
{
    "kernel-dev": {
        "probability": <float between 0 and 1>,
        "reason": "<short English explanation, max 50 words>"
    },
    "kernel-config": {
        "probability": <float between 0 and 1>,
        "reason": "<short English explanation, max 50 words>"
    }
}
```

Important notes:
- Clearly distinguish between regular Linux user questions and kernel development questions. A question about how to use a Linux application should receive a very low kernel-dev score.
- Topics involving direct kernel code, kernel data structures, kernel debugging, or kernel module development should receive high kernel-dev scores.
- Email about general Linux usage troubleshooting should score low unless they involve kernel-specific mechanisms.
- Evaluate "kernel-dev" and "kernel-config" independently.
- Ensure the JSON format is strictly followed. No extra text.
~~~

### 基于LLM的标准问答提取与规范化

该方法基于 LLM 从过滤后的原始语料中提取多个问答并规范化成格式统一的问答三元组，整体设计思路遵循“先生成、再自校验”的原则，具体分为子问题提取、参考答案生成、关键知识点提取、输出校验与标准化输出五个环节。针对不同来源的原始语料，prompt 略有差别，具体如下。

#### 来源一：技术问答社区

~~~
Please extract **1 to 3 independent sub-QAs** from an original Q&A related to Linux kernel development or configuration, and meet the following requirements:

### 1. Sub-question Extraction

- Extract 1 to 3 core sub-questions, primarily based on the original question title and description, secondarily on the answer
- Each sub-question must be **self-contained** and **independently answerable**, without relying on other questions
- Include complete key background information:
  - Extract background information relevant to the sub-question from the original question description, such as: kernel version, code snippets, configurations, error logs, etc.
  - Background information must be fully preserved, not just partial excerpts.
  - Prioritize using original code snippets from the question description, rather than paraphrased text.
- Formatting requirements:
  - First provide the complete background information (including detailed code snippets and context), then present a clear question statement.
  - **Write the question from a first-person perspective**, using phrases like "I am trying to...", "I encountered...", or "How can I...".

### 2. Sub-question Answering

- Provide a **complete and self-contained answer**
- **Only use information from the original Q&A**
- The answer must **directly solve the problem**
- Technical elements mentioned in the answer must be reflected in the question's background information

### 3. Key Points

- Include only necessary technical facts (including but not limited to complete commands, configuration steps, technical explanations)
- Do not include: speculation, irrelevant examples, redundant information

### 4. Output Verification

Before finalizing the output, perform the following key checks separately for each component (question, answer, key_points) of every sub-QA:

- Question Source Check:
  - Ensure each sub-question is derived from the original Q&A content

- Question Background Completeness Check:
  - If code is present, the entire original code snippet must be preserved, not a simplified version
  - If contextual elements are present, such as execution results, error logs, configuration details, or user guesses, they must be fully included from the original description

- Question-Answer Consistency Check:
  - Ensure technical elements in the answer (such as code, logs, commands, configurations, error messages, etc.) are mentioned in the question's background
  - If any are missing, either add them to the question background or remove them from the answer

- Key Point Necessity Test:
  - Test each key point:
    - If this point were removed, would the answer fail to solve the problem, or would it significantly increase the difficulty of understanding and solving the problem?
    - If the answer is "no," then this key point should be deleted
  - Ensure each key point represents an independent solution step or core understanding, without duplication

- Answer-Key Points Consistency Check:
  - Strict verification:
    - Each key point must have a corresponding original text or similar statement in the answer
    - If any are missing, either add them to the answer or remove them from the key points

- Format Check:
  - Ensure the output format of question, answer, and key_points complies with markdown syntax
  - Ensure that the question is formatted as: Background context + Question statement
  - Ensure that the question is written in the first person

### 5. Output Format

- Output in JSON format only, example:

```json
{
  "sub_qas": [
    {
      "question": "Self-contained question 1",
      "answer": "Complete answer 1",
      "key_points": [
        "Key point 1",
        "Key point 2"
      ]
    },
    {
      "question": "Self-contained question 2",
      "answer": "Complete answer 2",
      "key_points": [
        "Key point 1",
        "Key point 2"
      ]
    }
  ]
}
```
~~~

#### 来源二：内核邮件列表

鉴于邮件列表的内容通常较为丰富，因此设置最多可提取 5 个标准问题。

~~~
Please extract **1 to 5 independent sub-QAs** from an original mailing list related to Linux kernel development or configuration, and meet the following requirements:

### 1. Sub-question Extraction

- Extract 1 to 5 core sub-questions, primarily based on the original question title and description, secondarily on the answer, you have to discern which part of the email body is description or answer.
- Each sub-question must be **self-contained** and **independently answerable**, without relying on other questions
- Include complete key background information:
  - Extract background information relevant to the sub-question from the original question description, such as: kernel version, code snippets, configurations, error logs, etc.
  - Background information must be fully preserved, not just partial excerpts.
  - Background information must not contains the answer of the question.
  - Background information should be included as part of the complete question, rather than written separately as something like 'Background context:'.
  - Prioritize using original code snippets from the question description, rather than paraphrased text.
- Formatting requirements:
  - First provide the complete background information (including detailed code snippets and context), then present a clear question statement.
  - **Write the question from a first-person perspective**, using phrases like "I am trying to...", "I encountered...", or "How can I...".

### 2. Sub-question Answering

- Provide a **complete and self-contained answer**
- **Only use information from the original mailing list**
- The answer must **directly solve the problem**
- Technical elements mentioned in the answer must be reflected in the question's background information

### 3. Key Points

- Include only necessary technical facts (including but not limited to complete commands, configuration steps, technical explanations)
- Do not include: speculation, irrelevant examples, redundant information

### 4. Output Verification

Before finalizing the output, perform the following key checks separately for each component (question, answer, key_points) of every sub-QA:

- Redundancy Check:
  - Ensure that the output does not contain unnecessary redundant information such as:
    - Personal names of the senders and receivers from the original mailing list
    - Specific dates or times when the email was sent
    - Expressions of gratitude or polite phrases

- Question Source Check:
  - Ensure each sub-question is derived from the original mailing list content

- Question Background Completeness Check:
  - If code is present, the entire original code snippet must be preserved, not a simplified version
  - If contextual elements are present, such as execution results, error logs, configuration details, or user guesses, they must be fully included from the original description

- Question-Answer Consistency Check:
  - Ensure technical elements in the answer (such as code, logs, commands, configurations, error messages, etc.) are mentioned in the question's background
  - If any are missing, either add them to the question background or remove them from the answer

- Key Point Necessity Test:
  - Test each key point:
    - If this point were removed, would the answer fail to solve the problem, or would it significantly increase the difficulty of understanding and solving the problem?
    - If the answer is "no," then this key point should be deleted
  - Ensure each key point represents an independent solution step or core understanding, without duplication

- Answer-Key Points Consistency Check:
  - Strict verification:
    - Each key point must have a corresponding original text or similar statement in the answer
    - If any are missing, either add them to the answer or remove them from the key points

- Format Check:
  - Ensure the output format of question, answer, and key_points complies with markdown syntax
  - Ensure that the question is formatted as: Background context + Question statement
  - Ensure that the question is written in the first person

### 5. Output Format

- Output in JSON format only, example:

```json
{
  "sub_qas": [
    {
      "question": "Self-contained question 1",
      "answer": "Complete answer 1",
      "key_points": [
        "Key point 1",
        "Key point 2"
      ]
    },
    {
      "question": "Self-contained question 2",
      "answer": "Complete answer 2",
      "key_points": [
        "Key point 1",
        "Key point 2"
      ]
    }
  ]
}
```
~~~

### 基于LLM的问题分类

该方法基于 LLM 对每个标准问题进行主题、认知维度和版本相关性的多维标注。具体 prompt 如下。

#### 主题与认知维度分类

~~~
You are a professional Linux kernel issue classification expert. You will receive a Linux kernel-related question and should classify it accordingly:

1. Select 1-3 most relevant topic categories (provide category IDs)  
2. Determine the cognitive dimension of the question (provide cognitive dimension ID)  
3. Provide the reasoning for your classification  

List of Topic Categories:
1. Process Management
2. Memory Management
3. File Systems
4. Networking
5. Architecture
6. Driver Development
7. Security and Permissions
8. Virtualization and Containers
9. Interrupts and Exceptions
10. Power Management
11. Performance Optimization
12. System Boot
13. Debugging and Diagnostics
14. Kernel Configuration

Cognitive Dimension Definitions:
1. Basic Operations Layer (Class 1) – Aimed at end users, focused on basic commands and system configuration
2. Mechanism Understanding Layer (Class 2) – Requires in-depth understanding of kernel mechanisms, but does not involve code changes
3. Development and Debugging Layer (Class 3) – Involves source-level understanding, kernel extensions, or performance tuning

Please strictly return the result in the following JSON format, and do not output anything else:  
{
    "topic_ids": [list of category IDs],
    "cognitive_dimension_id": cognitive dimension ID,
    "reason": "Reason for classification"
}
~~~

#### 版本相关性分类

~~~
You will be given a question and its answer with the key points in the answer. You need to classify the content specifically focusing on Linux Kernel development relevance.

Please perform the following classification tasks with high precision:
Determine whether answering this question requires a specific Linux Kernel version (i.e., it depends on a particular Linux Kernel version). Provide:
   - a boolean value (true or false),
   - **if the value is true**, also provide the **specific kernel version** or version range mentioned or implied in the answer.
   - a short English explanation (reason) within 50 words.

Output format:  
Return **only** the following JSON structure:
```json
{
    "is_version_specific": {
        "value": <true or false>,
        "version": "<specific version or version range if value is true, otherwise null>"
        "reason": "<short English explanation, max 50 words>"
    }
}
```

Important notes:
- Judge "is_version_specific" primarily based on the answer content.
- If `value` is `true`, the `version` field must be a valid Linux kernel version string (e.g., "5.10.2", "4.x–5.x", etc.).
- If `value` is `false`, set `"version"` to `null`.
- Ensure the JSON format is strictly followed. No extra text.
~~~

### 多裁判协同的代码知识问答评测方法

#### (Fetch) 被测大模型的系统提示词

我们更关心模型的内在能力，因此设计了较为简单且统一的系统提示词，用于引导大模型进行输出。具体 prompt 如下。

~~~
You are a Linux Kernel expert. Please provide a concise and professional answer in English.
~~~

#### (Judge) 裁判大模型的系统提示词

采取单示例驱动的提示策略，并通过详细的维度说明与指标定义来指导裁判大模型进行具体评判。具体 prompt 如下。

~~~
You need to evaluate the quality of a candidate answer to a Linux kernel-related question. Provided information includes:
- question: The question being asked
- reference_answer: The correct answer
- key_points: Key knowledge points from the reference answer (numbered)
- candidate_answer: The answer to be evaluated

### Your Task

Analyze the candidate answer and output a strictly formatted JSON evaluation result containing the following fields:

- `"key_points_evaluation"`: An object containing:
  - `"missed"`: Key point numbers absent from the answer
  - `"partial"`: Key point numbers mentioned but lacking details
  - `"matched"`: Key point numbers fully covered

For each of the following fields, you MUST use exact, unmodified text from the candidate answer:
- `"factual_errors"`: List of factually incorrect statements
- `"vague_statements"`: List of ambiguous statements
- `"partially_correct_but_misleading"`: List of misleading statements
- `"irrelevant_but_correct"`: List of irrelevant statements

Each item in these lists must contain:
- `"exact_text"`: The exact text copied from the answer (DO NOT modify or summarize)
- `"explanation"`: Brief explanation of the issue

### Important Rules: Reference Original Text

1. Reference text must be exact:
   - Text must be copied directly from the answer, without modification
   - No summarizing, paraphrasing, or adding/removing words
   - No changing punctuation or formatting
   - If text spans multiple sentences, include complete sentences

2. Reference must be continuous:
   - Text must be a continuous segment from the answer
   - If multiple parts need to be referenced, create separate entries
   - Do not combine or concatenate text from different parts of the answer

### Example Input

```json
{
  "question": "How does the Linux kernel handle process scheduling?",
  "reference_answer": "The Linux kernel uses the Completely Fair Scheduler (CFS) as the default scheduler. CFS manages processes using a red-black tree (RB-Tree) and determines process priority based on vruntime. Additionally, the Linux kernel provides real-time scheduling policies such as SCHED_FIFO and SCHED_RR to meet different scheduling needs.",
  "key_points": {
    "1": "The Linux kernel uses CFS as the default scheduler",
    "2": "CFS manages processes using a red-black tree (RB-Tree)",
    "3": "CFS determines process priority based on vruntime",
    "4": "The Linux kernel provides real-time scheduling policies such as SCHED_FIFO and SCHED_RR"
  },
  "candidate_answer": "Linux uses Completely Fair Scheduler(CFS) for scheduling, and CFS allocates CPU time using time slices. Linux employs a certain data structure to manage processes. Additionally, Linux supports BPF for flexible scheduling."
}
```

### Example Output

Only output the JSON object, no other text or comments.

```json
{
  "key_points_evaluation": {
    "missed": [3, 4],
    "partial": [2],
    "matched": [1]
  },
  "factual_errors": [
    {
      "exact_text": "CFS allocates CPU time using time slices",
      "explanation": "CFS does not use time slices, it uses vruntime for fair scheduling"
    }
  ],
  "vague_statements": [
    {
      "exact_text": "Linux employs a certain data structure to manage processes",
      "explanation": "The statement is vague as it does not specify which data structure (red-black tree) is used"
    }
  ],
  "partially_correct_but_misleading": [],
  "irrelevant_but_correct": [
    {
      "exact_text": "Linux supports BPF for flexible scheduling",
      "explanation": "While BPF can be used for scheduling, it is not directly related to the core scheduling mechanisms being discussed"
    }
  ]
}
```
~~~