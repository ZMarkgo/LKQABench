import warnings
warnings.filterwarnings("ignore", message="torch.utils._pytree._register_pytree_node is deprecated")

import mysql.connector
import json
import math
import os
import re
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from common.timer import timer
from common.logger import setup_logger, get_logger, TimestampType
from common.config import load_config, get_db_config, get_api_key_secret_key
from common.entity import StandardQA, ModelAnswer, ParsedModelEvalResult
from common.llm_api import ModelConfig, load_models_configs
from common.api_key_encryptor import ApiKeyEncryptor
from common.sentence_model import SentenceModelManager
import argparse
from sentence_transformers import util

# 读取配置
CONFIG_NAME = "llm_qa_eval_fusion"
CONFIG = load_config()
DB_CONFIG = get_db_config(CONFIG)
LOG_FILE = CONFIG[CONFIG_NAME]['log_file']

if __name__ == "__main__":
    # 将在main函数中配置
    logger = None
else:
    # 当作为模块导入时，使用默认配置
    logger = get_logger(__name__)

class JudgeResultAggregator:
    """裁判结果融合器。
    
    用于将多个裁判模型的评估结果进行融合，生成最终的评估结果。
    该类负责处理多个裁判模型对同一个问答对的评估结果，通过语义匹配和位置重叠检测
    等方法将相似的评估内容进行合并，最终生成统一的评估结果。
    """
    def __init__(self, judge_results: List[ParsedModelEvalResult], standard_qa: StandardQA, original_text_raw: str, threshold: float = 0.6):
        """初始化融合器。
        
        Args:
            judge_results (List[ParsedModelEvalResult]): 多个裁判模型的评估结果列表
            standard_qa (StandardQA): 标准问答对，用于获取知识点数量
            original_text_raw (str): 原始回答文本，用于精确匹配
            threshold (float): 融合阈值，默认为0.6（60%）
        """
        self.judge_results: List[ParsedModelEvalResult] = judge_results
        self.standard_qa: StandardQA = standard_qa
        self.threshold = threshold
        self.num_judges = len(judge_results)
        self.threshold_count = math.ceil(self.num_judges * self.threshold)
        # 获取知识点总数
        self.total_key_points = len(standard_qa.key_points)
        # 获取句向量模型实例
        try:
            self.sentence_model = SentenceModelManager.get_model()
            if self.sentence_model is None:
                raise ValueError("无法获取句向量模型实例")
        except Exception as e:
            logger.error(f"加载句向量模型失败: {e}", exc_info=True)
            raise
        
        # 保留原始文本，用于精确匹配
        self.original_text_raw = original_text_raw
        
        # 预处理原文
        self.original_text = self.remove_markdown(original_text_raw)
        # 标准化文本，用于模糊匹配
        self.original_text = ' '.join(self.original_text.split())
        # 将原文分割成句子并记录位置
        self.original_sentences = []
        self.original_sentence_positions = []
        current_pos = 0
        for sentence in self.original_text.split('.'):
            sentence = sentence.strip()
            if sentence:
                self.original_sentences.append(sentence)
                start_pos = self.original_text.find(sentence, current_pos)
                end_pos = start_pos + len(sentence)
                self.original_sentence_positions.append((start_pos, end_pos))
                current_pos = end_pos
        # 预计算原文的句向量
        self.original_embeddings = self.sentence_model.encode(self.original_sentences, convert_to_tensor=True)

    def _find_semantic_match(self, text: str, threshold: float = 0.9) -> Tuple[str, float]:
        """使用句向量语义相似度查找最匹配的原文句子。
        
        计算待匹配文本与原文所有句子的语义相似度，返回相似度最高且超过阈值的句子。
        
        Args:
            text (str): 待匹配的文本
            threshold (float): 相似度阈值，默认0.9
            
        Returns:
            Tuple[str, float]: (最匹配的原文句子, 相似度分数)，如果没有匹配则返回(None, 分数)
        """
        # 计算待匹配文本的句向量
        query_embedding = self.sentence_model.encode(text, convert_to_tensor=True)
        
        # 计算与所有原文句子的相似度
        similarities = util.pytorch_cos_sim(query_embedding, self.original_embeddings)[0]
        
        # 找到最相似的句子
        max_sim_idx = similarities.argmax().item()
        max_sim_score = similarities[max_sim_idx].item()
        
        if max_sim_score >= threshold:
            return self.original_sentences[max_sim_idx], max_sim_score
        return None, max_sim_score

    def remove_markdown(self, text: str) -> str:
        """移除文本中的Markdown格式标记。
        
        移除常见的Markdown格式标记，包括加粗、斜体、代码块、标题、列表和引用等。
        
        Args:
            text (str): 包含Markdown格式的文本
            
        Returns:
            str: 移除Markdown格式后的纯文本
        """
        # 移除加粗标记 **
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        # 移除斜体标记 *
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        # 移除代码块标记 ```
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        # 移除行内代码标记 `
        text = re.sub(r'`(.*?)`', r'\1', text)
        # 移除标题标记 #
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
        # 移除列表标记 - 或 *
        text = re.sub(r'^[-*]\s*', '', text, flags=re.MULTILINE)
        # 移除引用标记 >
        text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)
        return text

    def normalize_text(self, text: str) -> str:
        """标准化文本，移除所有格式和多余空白。
        
        对文本进行标准化处理，包括移除Markdown格式、多余空白字符，并转换为小写。
        
        Args:
            text (str): 输入文本
            
        Returns:
            str: 标准化后的文本
        """
        # 移除Markdown格式标记
        text = self.remove_markdown(text)
        # 移除多余的空白字符
        text = ' '.join(text.split())
        # 转换为小写
        text = text.lower()
        return text

    def check_text_in_answer(self, text: str) -> Tuple[bool, int, int]:
        """检查文本是否在回答中，并返回匹配位置。
        
        使用多种匹配策略检查文本是否存在于原始回答中，包括精确匹配、标准化匹配和部分匹配。
        
        Args:
            text (str): 待匹配的文本
            
        Returns:
            Tuple[bool, int, int]: (是否匹配, 开始位置, 结束位置)
        """
        # 1. 尝试精确匹配（保留原始格式）
        start_pos = self.original_text_raw.find(text)
        if start_pos != -1:
            return True, start_pos, start_pos + len(text)
            
        # 2. 尝试标准化后匹配
        normalized_text = ' '.join(text.split())
        start_pos = self.original_text.find(normalized_text)
        if start_pos != -1:
            return True, start_pos, start_pos + len(normalized_text)
            
        # 3. 尝试部分匹配（如果文本较长）
        if len(text) > 20:  # 只对较长的文本尝试部分匹配
            words = normalized_text.split()
            if len(words) > 3:  # 至少需要3个词
                # 尝试匹配连续的3个词
                for i in range(len(words) - 2):
                    phrase = ' '.join(words[i:i+3])
                    start_pos = self.original_text.find(phrase)
                    if start_pos != -1:
                        return True, start_pos, start_pos + len(phrase)
        
        return False, -1, -1

    def find_sentence_by_position(self, start_pos: int, end_pos: int) -> Tuple[int, str]:
        """根据位置找到对应的句子。
        
        根据给定的文本位置范围，查找包含该位置的句子。
        
        Args:
            start_pos (int): 开始位置
            end_pos (int): 结束位置
            
        Returns:
            Tuple[int, str]: (句子索引, 句子内容)，如果未找到则返回(-1, "")
        """
        for i, (s_start, s_end) in enumerate(self.original_sentence_positions):
            if s_start <= start_pos and end_pos <= s_end:
                return i, self.original_sentences[i]
        return -1, ""

    def aggregate_statements(self, statements_list: List[List[Dict[str, str]]], original_text: str) -> List[Dict[str, str]]:
        """融合语句评估结果。
        
        将多个裁判模型对同一类语句的评估结果进行合并，通过多种匹配策略找到对应的原文位置。
        
        Args:
            statements_list (List[List[Dict[str, str]]]): 多个裁判模型对同一类语句的评估结果列表
            original_text (str): 原始回答文本，用于精确匹配
            
        Returns:
            List[Dict[str, str]]: 融合后的语句评估结果列表
        """
        if not statements_list:
            logger.warning("statements_list 为空")
            return []
        
        # 记录无法匹配的语句
        unmatch_statements = []

        # 将所有语句的exact_text提取出来
        all_statements = []
        logger.info(f"开始处理 {len(statements_list)} 个裁判的评估结果")
        
        for judge_idx, statements in enumerate(statements_list):
            logger.info(f"处理第 {judge_idx + 1} 个裁判的 {len(statements)} 条评估结果")
            for stmt in statements:
                # 1. 精确子串匹配（优先使用）
                matched, start_pos, end_pos = self.check_text_in_answer(stmt['exact_text'])
                if matched:
                    # 找到对应的句子
                    sentence_idx, sentence_text = self.find_sentence_by_position(start_pos, end_pos)
                    logger.info(f"精确匹配成功: {stmt['exact_text']} -> {sentence_text}")
                    all_statements.append({
                        'exact_text': stmt['exact_text'],  # 使用原始文本
                        'explanation': stmt['explanation'],
                        'count': 1,
                        'match_type': 'exact',
                        'position': (start_pos, end_pos),
                        'sentence_idx': sentence_idx,
                        'original_text': stmt['exact_text']
                    })
                    continue
                
                # 2. 最小编辑距离匹配（备选方案）
                min_distance = float('inf')
                best_match_idx = -1
                for i, sentence in enumerate(self.original_sentences):
                    distance = self._levenshtein_distance(stmt['exact_text'], sentence)
                    if distance < min_distance:
                        min_distance = distance
                        best_match_idx = i
                
                if min_distance <= 10:  # 编辑距离阈值
                    start_pos, end_pos = self.original_sentence_positions[best_match_idx]
                    logger.info(f"编辑距离匹配成功: {stmt['exact_text']} -> {self.original_sentences[best_match_idx]}")
                    all_statements.append({
                        'exact_text': stmt['exact_text'],  # 使用原始文本
                        'explanation': stmt['explanation'],
                        'count': 1,
                        'match_type': 'edit_distance',
                        'position': (start_pos, end_pos),
                        'sentence_idx': best_match_idx,
                        'original_text': stmt['exact_text']
                    })
                    continue
                
                # 3. 句向量语义相似度（最后选择）
                matched_text, sim_score = self._find_semantic_match(stmt['exact_text'])
                if matched_text:
                    # 找到匹配句子的位置
                    for i, sentence in enumerate(self.original_sentences):
                        if sentence == matched_text:
                            start_pos, end_pos = self.original_sentence_positions[i]
                            logger.info(f"语义相似度匹配成功: {stmt['exact_text']} -> {matched_text}")
                            all_statements.append({
                                'exact_text': stmt['exact_text'],  # 使用原始文本
                                'explanation': stmt['explanation'],
                                'count': 1,
                                'match_type': 'semantic',
                                'similarity_score': sim_score,
                                'position': (start_pos, end_pos),
                                'sentence_idx': i,
                                'original_text': stmt['exact_text']
                            })
                            break
                    continue
                
                # 如果所有匹配方法都失败，记录为无法匹配
                logger.warning(f"语句无法匹配: {stmt['exact_text']}")
                unmatch_statements.append({
                    'exact_text': stmt['exact_text'],
                    'explanation': stmt['explanation'],
                    'match_type': 'unmatched'
                })

        logger.info(f"匹配完成，共找到 {len(all_statements)} 条匹配结果，{len(unmatch_statements)} 条未匹配结果")

        # 使用句子索引和位置信息合并相同语句
        merged_statements = []
        for stmt in all_statements:
            # 检查是否已经存在相同位置或位置重叠的语句
            found = False
            for merged in merged_statements:
                # 检查句子索引和位置关系
                if (merged['sentence_idx'] == stmt['sentence_idx'] and 
                    self.is_position_overlap(merged['position'], stmt['position'])):
                    merged['count'] += 1
                    # 保留所有的解释
                    if 'explanations' not in merged:
                        merged['explanations'] = [merged['explanation']]
                    merged['explanations'].append(stmt['explanation'])
                    # 更新位置信息为并集
                    merged['position'] = self.merge_positions(merged['position'], stmt['position'])
                    # 保存所有原始文本
                    if 'original_texts' not in merged:
                        merged['original_texts'] = [merged['original_text']]
                    merged['original_texts'].append(stmt['original_text'])
                    logger.info(f"合并语句: {stmt['exact_text']} -> {merged['exact_text']}")
                    found = True
                    break
            
            if not found:
                logger.info(f"添加新语句: {stmt['exact_text']}")
                merged_statements.append({
                    'exact_text': stmt['exact_text'],
                    'explanation': stmt['explanation'],
                    'count': 1,
                    'match_type': stmt['match_type'],
                    'position': stmt['position'],
                    'sentence_idx': stmt['sentence_idx'],
                    'original_text': stmt['original_text']
                })

        logger.info(f"合并完成，共 {len(merged_statements)} 条合并结果")
        
        # 根据出现次数筛选语句
        result = {
            'merged_results': [],
            'unmerged_results': [],
            'unmatched_results': unmatch_statements
        }
        
        for stmt in merged_statements:
            if stmt['count'] >= self.threshold_count:  # 至少60%个裁判认为有问题
                logger.info(f"添加到merged_results: {stmt['exact_text']} (count={stmt['count']})")
                result['merged_results'].append({
                    'exact_text': stmt['exact_text'],
                    'explanations': stmt.get('explanations', [stmt['explanation']]),
                    'count': stmt['count'],
                    'match_type': stmt['match_type'],
                    'position': stmt['position'],
                    'sentence_idx': stmt['sentence_idx'],
                    'original_texts': stmt.get('original_texts', [stmt['original_text']])
                })
            else:
                logger.info(f"添加到unmerged_results: {stmt['exact_text']} (count={stmt['count']})")
                result['unmerged_results'].append({
                    'exact_text': stmt['exact_text'],
                    'explanations': stmt.get('explanations', [stmt['explanation']]),
                    'count': stmt['count'],
                    'match_type': stmt['match_type'],
                    'position': stmt['position'],
                    'sentence_idx': stmt['sentence_idx'],
                    'original_texts': stmt.get('original_texts', [stmt['original_text']])
                })
        
        logger.info(f"最终结果: merged_results={len(result['merged_results'])}, unmerged_results={len(result['unmerged_results'])}, unmatched_results={len(result['unmatched_results'])}")
        return result

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """计算两个字符串之间的编辑距离。
        
        使用动态规划算法计算两个字符串之间的最小编辑距离（Levenshtein距离）。
        
        Args:
            s1 (str): 第一个字符串
            s2 (str): 第二个字符串
            
        Returns:
            int: 两个字符串之间的编辑距离
        """
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

    def is_position_overlap(self, pos1: Tuple[int, int], pos2: Tuple[int, int], threshold: float = 0.3) -> bool:
        """检查两个位置是否存在重叠关系。
        
        计算两个位置区间的重叠程度，判断是否超过指定阈值。
        
        Args:
            pos1 (Tuple[int, int]): 第一个位置，格式为(start1, end1)
            pos2 (Tuple[int, int]): 第二个位置，格式为(start2, end2)
            threshold (float): 重叠阈值，默认0.3（30%）
            
        Returns:
            bool: 如果重叠部分超过阈值，返回True
        """
        start1, end1 = pos1
        start2, end2 = pos2
        
        # 计算重叠部分
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        
        if overlap_end <= overlap_start:
            return False
            
        # 计算重叠长度
        overlap_length = overlap_end - overlap_start
        # 计算较短文本的长度
        min_length = min(end1 - start1, end2 - start2)
        
        # 如果重叠部分超过较短文本的阈值，认为存在重叠
        return overlap_length / min_length >= threshold

    def merge_positions(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> Tuple[int, int]:
        """合并两个位置，取并集。
        
        将两个位置区间合并为一个更大的区间，包含两个原始区间的所有位置。
        
        Args:
            pos1 (Tuple[int, int]): 第一个位置，格式为(start1, end1)
            pos2 (Tuple[int, int]): 第二个位置，格式为(start2, end2)
            
        Returns:
            Tuple[int, int]: 合并后的位置，格式为(min(start1, start2), max(end1, end2))
        """
        start1, end1 = pos1
        start2, end2 = pos2
        return min(start1, start2), max(end1, end2)

    def aggregate_key_points(self) -> dict:
        """融合知识点评估结果。
        
        统计每个知识点在多个裁判模型中的评估情况，生成融合后的知识点评估结果。
        
        Returns:
            dict: 融合后的知识点评估结果，包含missed、partial_match、match等字段
        """
        # 统计每个知识点的评估情况
        key_points_stats = {}

        # 初始化 key_points_stats
        for kp in range(1, self.total_key_points + 1):
            key_points_stats[kp] = {'matched': 0, 'partial': 0, 'missed': 0}
        # 用于记录额外的知识点
        extra_key_points = []
        # 记录遗漏的知识点
        missed_key_points = []

        # 统计每个知识点的评估情况
        for judge_result in self.judge_results:
            # 1~self.total_key_points
            standard_qa_pk_number_set = set(range(1, self.total_key_points + 1))
            # 统计matched
            for kp in judge_result.matched_key_points:
                if kp not in key_points_stats:
                    if kp not in extra_key_points:
                        extra_key_points.append({
                            'kp': kp,
                            'judge_model_info_list': [judge_result.format_judge_model_info()]
                        })
                    else:
                        extra_key_points[kp]['judge_model_info_list'].append(judge_result.format_judge_model_info())
                else:
                    key_points_stats[kp]['matched'] += 1
                    standard_qa_pk_number_set.remove(kp)
            
            # 统计partial
            for kp in judge_result.partial_key_points:
                if kp not in key_points_stats:
                    if kp not in extra_key_points:
                        extra_key_points.append({
                            'kp': kp,
                            'judge_model_info_list': [judge_result.format_judge_model_info()]
                        })
                    else:
                        extra_key_points[kp]['judge_model_info_list'].append(judge_result.format_judge_model_info())
                else:
                    key_points_stats[kp]['partial'] += 1
                    standard_qa_pk_number_set.remove(kp)
            
            # 统计missed
            for kp in judge_result.missed_key_points:
                if kp not in key_points_stats:
                    if kp not in extra_key_points:
                        extra_key_points.append({
                            'kp': kp,
                            'judge_model_info_list': [judge_result.format_judge_model_info()]
                        })
                    else:
                        extra_key_points[kp]['judge_model_info_list'].append(judge_result.format_judge_model_info())
                else:
                    key_points_stats[kp]['missed'] += 1
                    standard_qa_pk_number_set.remove(kp)

            # 记录遗漏的知识点
            for kp in standard_qa_pk_number_set:
                if kp not in missed_key_points:
                    missed_key_points.append({
                        'kp': kp,
                        'judge_model_info_list': [judge_result.format_judge_model_info()]
                    })
                else:
                    missed_key_points[kp]['judge_model_info_list'].append(judge_result.format_judge_model_info())

        # 根据阈值进行融合
        result = {
            'matched': [],
            'partial': [],
            'missed': [],
            'extra_key_points': extra_key_points,
            'missed_key_points': missed_key_points,
        }
        
        for kp, stats in key_points_stats.items():
            # Matched（完全匹配）
            if stats['matched'] >= self.threshold_count and stats['missed'] == 0:
                result['matched'].append(kp)
            # Missed（完全缺失）
            elif stats['missed'] >= self.threshold_count:
                result['missed'].append(kp)
            # Partial（部分匹配）
            else:
                result['partial'].append(kp)
    
        return result

    def aggregate(self) -> dict:
        """融合所有评估结果。
        
        将多个裁判模型的评估结果进行融合，包括知识点评估和各类语句评估。
        
        Returns:
            dict: 包含以下字段的融合结果：
                - key_points_evaluation: 知识点评估结果
                    - matched: 完全匹配的知识点列表
                    - partial: 部分匹配的知识点列表
                    - missed: 完全缺失的知识点列表
                    - extra_key_points: 额外评估的知识点列表
                    - missed_key_points: 遗漏的知识点列表
                - factual_errors: 事实性错误评估结果
                    - merged_results: 已融合的结果列表
                    - unmerged_results: 未融合的结果列表
                - vague_statements: 模糊表述评估结果
                    - merged_results: 已融合的结果列表
                    - unmerged_results: 未融合的结果列表
                - partially_correct_but_misleading: 部分正确但具有误导性的表述评估结果
                    - merged_results: 已融合的结果列表
                    - unmerged_results: 未融合的结果列表
                - irrelevant_but_correct: 无关但正确的表述评估结果
                    - merged_results: 已融合的结果列表
                    - unmerged_results: 未融合的结果列表
        """
        # 融合知识点评估结果
        key_points_evaluation = self.aggregate_key_points()
        
        logger.info(f"original_text_raw: \n{self.original_text_raw}")
        # 融合各类语句评估结果
        # logger.info(f"judge_results: \n{ParsedModelEvalResult.list_to_json(self.judge_results, indent=4)}")
        factual_errors_list = [r.factual_errors for r in self.judge_results]
        # logger.info(f"factual_errors_list: \n{json.dumps(factual_errors_list, indent=4, ensure_ascii=False)}")
        vague_statements_list = [r.vague_statements for r in self.judge_results]
        # logger.info(f"vague_statements_list: \n{json.dumps(vague_statements_list, indent=4, ensure_ascii=False)}")
        partially_correct_list = [r.partially_correct_but_misleading for r in self.judge_results]
        # logger.info(f"partially_correct_list: \n{json.dumps(partially_correct_list, indent=4, ensure_ascii=False)}")
        irrelevant_correct_list = [r.irrelevant_but_correct for r in self.judge_results]
        # logger.info(f"irrelevant_correct_list: \n{json.dumps(irrelevant_correct_list, indent=4, ensure_ascii=False)}")

        factual_errors = self.aggregate_statements(factual_errors_list, self.standard_qa.split_answer)
        vague_statements = self.aggregate_statements(vague_statements_list, self.standard_qa.split_answer)
        partially_correct = self.aggregate_statements(partially_correct_list, self.standard_qa.split_answer)
        irrelevant_correct = self.aggregate_statements(irrelevant_correct_list, self.standard_qa.split_answer)
        
        # 记录融合过程的统计信息
        stats = {
            'total_judges': self.num_judges,
            'threshold': self.threshold,
            'threshold_count': self.threshold_count,
            'total_key_points': self.total_key_points,
            'key_points_stats': {
                'matched': len(key_points_evaluation['matched']),
                'partial': len(key_points_evaluation['partial']),
                'missed': len(key_points_evaluation['missed']),
                'extra': len(key_points_evaluation['extra_key_points']),
                'missed': len(key_points_evaluation['missed_key_points'])
            },
            'statements_stats': {
                'factual_errors': {
                    'merged': len(factual_errors['merged_results']),
                    'unmerged': len(factual_errors['unmerged_results']),
                    'unmatched': len(factual_errors['unmatched_results'])
                },
                'vague_statements': {
                    'merged': len(vague_statements['merged_results']),
                    'unmerged': len(vague_statements['unmerged_results']),
                    'unmatched': len(vague_statements['unmatched_results'])
                },
                'partially_correct': {
                    'merged': len(partially_correct['merged_results']),
                    'unmerged': len(partially_correct['unmerged_results']),
                    'unmatched': len(partially_correct['unmatched_results'])
                },
                'irrelevant_correct': {
                    'merged': len(irrelevant_correct['merged_results']),
                    'unmerged': len(irrelevant_correct['unmerged_results']),
                    'unmatched': len(irrelevant_correct['unmatched_results'])
                }
            }
        }
        
        return {
            'key_points_evaluation': key_points_evaluation,
            'factual_errors': factual_errors,
            'vague_statements': vague_statements,
            'partially_correct_but_misleading': partially_correct,
            'irrelevant_but_correct': irrelevant_correct,
            'stats': stats,
            'judge_results': self.judge_results
        }

def save_aggregated_eval_result(model_answer_id: int, threshold: float, aggregated_result: dict):
    """保存融合后的评估结果到数据库。
    
    将融合后的评估结果保存到llm_qa_eval_fusion_result表中。
    
    Args:
        model_answer_id (int): 模型回答ID
        threshold (float): 融合阈值
        aggregated_result (dict): 融合后的评估结果
    """
    conn = mysql.connector.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cursor:
            # 准备知识点评估结果
            key_points_eval = aggregated_result['key_points_evaluation']
            
            # 准备各类语句评估结果
            factual_errors = aggregated_result['factual_errors']
            vague_statements = aggregated_result['vague_statements']
            partially_correct = aggregated_result['partially_correct_but_misleading']
            irrelevant_correct = aggregated_result['irrelevant_but_correct']
            
            # 开始事务
            conn.start_transaction()
            
            # 1. 插入主表数据
            cursor.execute("""
                INSERT INTO aggregated_eval_results
                (model_answer_id, threshold,
                missed_key_points, partial_key_points, matched_key_points,
                extra_key_points, missed_key_points_details,
                factual_errors_merged, factual_errors_unmerged, factual_errors_unmatched,
                vague_statements_merged, vague_statements_unmerged, vague_statements_unmatched,
                partially_correct_merged, partially_correct_unmerged, partially_correct_unmatched,
                irrelevant_correct_merged, irrelevant_correct_unmerged, irrelevant_correct_unmatched,
                stats)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                model_answer_id,
                threshold,
                json.dumps(key_points_eval['missed']),
                json.dumps(key_points_eval['partial']),
                json.dumps(key_points_eval['matched']),
                json.dumps(key_points_eval['extra_key_points']),
                json.dumps(key_points_eval['missed_key_points']),
                json.dumps(factual_errors['merged_results']),
                json.dumps(factual_errors['unmerged_results']),
                json.dumps(factual_errors['unmatched_results']),
                json.dumps(vague_statements['merged_results']),
                json.dumps(vague_statements['unmerged_results']),
                json.dumps(vague_statements['unmatched_results']),
                json.dumps(partially_correct['merged_results']),
                json.dumps(partially_correct['unmerged_results']),
                json.dumps(partially_correct['unmerged_results']),
                json.dumps(irrelevant_correct['merged_results']),
                json.dumps(irrelevant_correct['unmerged_results']),
                json.dumps(irrelevant_correct['unmerged_results']),
                json.dumps(aggregated_result['stats'])
            ))
            
            # 获取新插入的融合结果ID
            aggregated_eval_id = cursor.lastrowid
            
            # 2. 插入关联表数据
            for judge_result in aggregated_result.get('judge_results', []):
                cursor.execute("""
                    INSERT INTO aggregated_eval_judge_models
                    (aggregated_eval_id, judge_model_id, model_eval_result_id)
                    VALUES (%s, %s, %s)
                """, (
                    aggregated_eval_id,
                    judge_result.judgeModelConfig.id, 
                    judge_result.id
                ))
            
            # 提交事务
            conn.commit()
            logger.info(f"成功保存融合结果到数据库，model_answer_id: {model_answer_id}, aggregated_eval_id: {aggregated_eval_id}")
    except Exception as e:
        # 回滚事务
        conn.rollback()
        logger.error(f"保存融合结果到数据库时出错: {e}", exc_info=True)
        raise
    finally:
        conn.close()

def check_if_already_fused(model_answer_id: int, judge_model_ids: List[int]) -> bool:
    """检查是否已经融合过。
    
    检查指定的模型回答是否已经使用相同的裁判模型组合进行过融合。
    
    Args:
        model_answer_id (int): 模型回答ID
        judge_model_ids (List[int]): 裁判模型ID列表
        
    Returns:
        bool: 如果已经融合过则返回True，否则返回False
    """
    conn = mysql.connector.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cursor:
            # 构建查询条件
            judge_model_placeholders = ', '.join(['%s'] * len(judge_model_ids))
            params = [model_answer_id] + judge_model_ids
            
            # 查询是否存在完全匹配的融合结果
            cursor.execute(f"""
                SELECT COUNT(DISTINCT aejm.aggregated_eval_id) as count
                FROM aggregated_eval_results aer
                JOIN aggregated_eval_judge_models aejm ON aer.id = aejm.aggregated_eval_id
                WHERE aer.model_answer_id = %s
                AND aejm.judge_model_id IN ({judge_model_placeholders})
                GROUP BY aejm.aggregated_eval_id
                HAVING COUNT(DISTINCT aejm.judge_model_id) = %s
            """, params + [len(judge_model_ids)])
            
            result = cursor.fetchone()
            return result[0] > 0 if result else False
    finally:
        conn.close()

def get_model_answers_for_fusion(llm_model_id: int, judge_model_ids: List[int], qa_ids: List[int] = None) -> List[Tuple[StandardQA, ModelAnswer]]:
    """获取需要融合的模型回答。
    
    根据指定的被测模型、裁判模型和问答对ID，获取需要进行融合的模型回答数据。
    
    Args:
        llm_model_id (int): 被测模型的数据库ID
        judge_model_ids (List[int]): 裁判模型的数据库ID列表
        qa_ids (List[int], optional): 问答对ID列表，如果为None则获取所有问答对
        
    Returns:
        List[Tuple[StandardQA, ModelAnswer]]: 问答对和对应的模型回答列表
    """
    conn = mysql.connector.connect(**DB_CONFIG)
    try:
        with conn.cursor(dictionary=True) as cursor:
            # 先找到所有的 StandardQA
            # 目标问答对
            qa_placeholders = ', '.join(['%s'] * len(qa_ids))
            # 目标裁判大模型
            judge_model_placeholders = ', '.join(['%s'] * len(judge_model_ids))
            cursor.execute(f"""
                SELECT DISTINCT r.id                 AS id,
                                q.source_question_id AS source_question_id,
                                a.source_answer_id   AS source_answer_id,
                                q.question           AS question,
                                a.answer             AS answer,
                                a.key_points         AS key_points,
                                q.status_id          AS q_status_id,
                                q.status_reason      AS q_status_reason,
                                a.status_id          AS a_status_id,
                                a.status_reason      AS a_status_reason,
                                mea.model_id         AS model_answer_id
                FROM model_eval_results mer
                        JOIN model_eval_answers mea ON mer.model_answer_id = mea.id
                        JOIN standard_qa_relations r ON r.id = mea.qa_id
                        JOIN standard_questions q ON r.question_id = q.id
                        JOIN standard_answers a ON r.answer_id = a.id
                WHERE mer.judge_model_id in ({judge_model_placeholders})
                AND mea.model_id = {llm_model_id}
                AND mea.qa_id IN ({qa_placeholders})
            """, tuple(judge_model_ids) + tuple(qa_ids))
            qa_rows = cursor.fetchall()

            # 再找到所有的 StandardQA对应的 ModelAnswer
            results = []
            for row in qa_rows:
                qa = StandardQA.from_db_row(row)
                cursor.execute("""
                    SELECT mea.id           AS id,
                        mea.qa_id        AS qa_id,
                        llm.id           AS llm_model_id,
                        llm.model_name   AS model_name,
                        llm.model_remark AS model_id,
                        mea.answer_content,
                        mea.reasoning_content,
                        mea.updated_at,
                        mea.created_at
                    FROM model_eval_answers mea
                            JOIN llm_models llm on mea.model_id = llm.id
                    WHERE mea.qa_id = %s AND mea.model_id = %s
                """, (qa.id, llm_model_id))
                result = cursor.fetchone()
                model_answer = ModelAnswer.from_db_row(result) if result else None
                if model_answer:
                    results.append((qa, model_answer))
            return results
    finally:
        conn.close()

def get_model_answers_for_fusion_batch(offset, limit, llm_model_id: int, judge_model_ids: List[int]) -> List[Tuple[StandardQA, ModelAnswer]]:
    """
    获取一批还没有融合过的模型回答：
    :param offset: 偏移量
    :param limit: 限制数量
    :param llm_model_id: 被测模型的数据库ID
    :param judge_model_ids: 裁判模型的数据库ID列表
    :return: 问答对和对应的模型回答列表[(StandardQA, ModelAnswer)]
    """
    conn = mysql.connector.connect(**DB_CONFIG)
    try:
        with conn.cursor(dictionary=True) as cursor:
            # 构建查询条件
            judge_model_placeholders = ', '.join(['%s'] * len(judge_model_ids))
            
            # 1. 首先获取所有未被融合的裁判结果
            logger.info(f"获取未被融合的裁判结果... offset: {offset}, limit: {limit}, llm_model_id: {llm_model_id}, judge_model_ids: {judge_model_ids}")
            cursor.execute(f"""
                WITH unfused_results AS (
                    SELECT mer.model_answer_id, mer.judge_model_id
                    FROM model_eval_results mer
                    LEFT JOIN aggregated_eval_judge_models aejm ON mer.id = aejm.model_eval_result_id
                    WHERE mer.judge_model_id IN ({judge_model_placeholders})
                    AND mer.model_answer_id IN (
                        SELECT id FROM model_eval_answers 
                        WHERE model_id = %s
                    )
                    AND aejm.id IS NULL
                )
                SELECT model_answer_id
                FROM unfused_results
                GROUP BY model_answer_id
                HAVING COUNT(DISTINCT judge_model_id) = %s  -- 需要所有裁判模型的结果
                ORDER BY model_answer_id
                LIMIT %s OFFSET %s
            """, tuple(judge_model_ids) + (llm_model_id, len(judge_model_ids), limit, offset))
            
            model_answer_ids = [row['model_answer_id'] for row in cursor.fetchall()]
            
            if not model_answer_ids:
                logger.info("没有找到符合条件的未融合裁判结果")
                return []
                
            logger.info(f"找到 {len(model_answer_ids)} 个需要融合的model_answer_ids: {model_answer_ids}")
            
            # 2. 获取对应的模型回答和问答对信息
            model_answer_placeholders = ', '.join(['%s'] * len(model_answer_ids))
            
            cursor.execute(f"""
                SELECT DISTINCT 
                    r.id AS id,
                    q.source_question_id AS source_question_id,
                    a.source_answer_id AS source_answer_id,
                    q.question AS question,
                    a.answer AS answer,
                    a.key_points AS key_points,
                    q.status_id AS q_status_id,
                    q.status_reason AS q_status_reason,
                    a.status_id AS a_status_id,
                    a.status_reason AS a_status_reason,
                    mea.id AS model_answer_id,
                    mea.qa_id AS qa_id,
                    llm.id AS llm_model_id,
                    llm.model_name AS model_name,
                    llm.model_remark AS model_remark,
                    llm.model_id AS model_id,
                    mea.answer_content,
                    mea.reasoning_content,
                    mea.updated_at AS model_answer_updated_at,
                    mea.created_at AS model_answer_created_at
                FROM model_eval_answers mea
                JOIN standard_qa_relations r ON r.id = mea.qa_id
                JOIN standard_questions q ON r.question_id = q.id
                JOIN standard_answers a ON r.answer_id = a.id
                JOIN llm_models llm ON mea.model_id = llm.id
                WHERE mea.id IN ({model_answer_placeholders})
            """, tuple(model_answer_ids))
            
            results = []
            for row in cursor.fetchall():
                qa = StandardQA.from_db_row(row)
                model_answer = ModelAnswer(
                    id=row['model_answer_id'],
                    qa_id=row['qa_id'],
                    llm_model_id=row['llm_model_id'],
                    model_name=row['model_name'],
                    model_id=row['model_id'],
                    model_remark=row['model_remark'],
                    reasoning_content=row['reasoning_content'],
                    answer_content=row['answer_content'],
                    updated_at=row['model_answer_updated_at'],
                    created_at=row['model_answer_created_at'],
                    is_new=False
                )
                results.append((qa, model_answer))
                    
            logger.info(f"成功获取 {len(results)} 个问答对和模型回答")
            
            return results
    finally:
        conn.close()

def get_judge_results_for_fusion(model_answer_id: int, judge_model_ids: List[int]) -> List[ParsedModelEvalResult]:
    """获取需要融合的裁判结果。
    
    根据模型回答ID和裁判模型ID列表，获取对应的裁判评估结果。
    
    Args:
        model_answer_id (int): 模型回答ID
        judge_model_ids (List[int]): 裁判模型的数据库ID列表
        
    Returns:
        List[ParsedModelEvalResult]: 解析后的裁判结果列表
    """
    conn = mysql.connector.connect(**DB_CONFIG)
    try:
        with conn.cursor(dictionary=True) as cursor:
            placeholders = ', '.join(['%s'] * len(judge_model_ids))
            cursor.execute(f"""
                SELECT *
                FROM model_eval_results
                WHERE model_answer_id = %s
                AND judge_model_id IN ({placeholders})
            """, (model_answer_id,) + tuple(judge_model_ids))
            results = []
            # 每一行是 一个裁判结果
            for row in cursor.fetchall():
                # 从 llm_models 表中获取裁判模型配置
                cursor.execute("""
                    SELECT *
                    FROM llm_models
                    WHERE id = %s
                """, (row['judge_model_id'],))
                judge_model_config_db_row = cursor.fetchone()
                results.append(ParsedModelEvalResult.from_db_row(
                    judge_result_db_row=row,
                    judge_model_config_db_row=judge_model_config_db_row
                ))
            return results
    finally:
        conn.close()

@timer
def process_model_fusion(model_config: ModelConfig, judge_model_ids: List[int], qa_src: str, qa_id_list: List[int], batch_size: int, limit_qa_count: int):
    """处理单个被测模型的裁判结果融合。
    
    对指定的被测模型，获取其所有裁判结果并进行融合处理，生成最终的评估结果。
    
    Args:
        model_config (ModelConfig): 被测模型配置
        judge_model_ids (List[int]): 裁判模型ID列表
        qa_src (str): 问答对来源
        qa_id_list (List[int]): 问答对ID列表
        batch_size (int): 批处理大小
        limit_qa_count (int): 限制问答对数量
    """
    # 用于日志打印
    ## 被测模型
    eval_model_logger_str = model_config.format_single_info()
    ## 裁判模型
    judge_model_ids_logger_str = ', '.join([f"{judge_model_id}" for judge_model_id in judge_model_ids])
    ## 日志打印
    model_logger_str = f"被测模型: {eval_model_logger_str}, 裁判模型: {judge_model_ids_logger_str}"
    logger.info(f"开始融合... {model_logger_str}")

    if qa_src == 'list':
        logger.info(f"limit_qa_count 原始值: {limit_qa_count}")
        logger.info(f"qa_id_list 长度: {len(qa_id_list)}")
        limit_qa_count = min(limit_qa_count, len(qa_id_list))
        logger.info(f"limit_qa_count 更新为: {limit_qa_count}")
    
    processed_count = 0  # 已处理的问答对数量
    while processed_count < limit_qa_count:
        try:
            # 偏移量不需要进行更新，一直设定为0，因为处理完，会保存融合结果，使得需要融合的结果减少
            offset = 0
            # 获取需要融合的问答对
            if qa_src == 'list':
                qa_answers = get_model_answers_for_fusion(
                    model_config.id,
                    judge_model_ids,
                    qa_id_list[offset:min(offset + batch_size, len(qa_id_list))] if qa_id_list else None
                )
            else:
                qa_answers = get_model_answers_for_fusion_batch(
                    offset=offset,
                    limit=batch_size,
                    llm_model_id=model_config.id,
                    judge_model_ids=judge_model_ids
                )
                
            if not qa_answers:
                logger.info(f"没有需要融合的问答对, 停止处理. {model_logger_str}")
                break

            # 处理查询到的问答对数量，以保证不超过限制数量
            remaining_count = limit_qa_count - processed_count
            if len(qa_answers) > remaining_count:
                logger.info(f"获取到的问答对数量({len(qa_answers)})超过剩余限制({remaining_count}), 仅使用前{remaining_count}个问答对")
                qa_answers = qa_answers[:remaining_count]

            processed_count += len(qa_answers)
            
            qa_answer_logger_str = f"{model_logger_str}, qa_answers(共{len(qa_answers)}个) : qa_ids=[{', '.join([str(qa.id) for qa, _ in qa_answers])}] \
                                    model_answer_ids=[{', '.join([str(model_answer.id) for _, model_answer in qa_answers])}]"
            logger.info(f"开始融合一批评估结果... {qa_answer_logger_str}")
            
            # 对每个问答对进行融合
            for qa, model_answer in qa_answers:
                inner_qa_answer_logger_str = f"{model_logger_str}, 问答对ID: {qa.id}, 模型回答ID: {model_answer.id}"
                try:
                    logger.info(f"开始融合评估结果... {inner_qa_answer_logger_str}")
                    # 获取该问答对的所有裁判结果
                    logger.info(f"获取裁判结果... {inner_qa_answer_logger_str}")
                    judge_results = get_judge_results_for_fusion(
                        model_answer.id,
                        judge_model_ids
                    )

                    if len(judge_results) != len(judge_model_ids):
                        logger.warning(f"裁判结果数量不匹配, 跳过融合. {inner_qa_answer_logger_str}, 裁判结果数量(获得的judge_results): {len(judge_results)}, 预期裁判数量(judge_model_ids): {len(judge_model_ids)}")
                        continue
                    else:
                        logger.info(f"裁判结果数量匹配. {inner_qa_answer_logger_str}, 裁判结果数量: {len(judge_results)}")
                    
                    # 至少需要2个裁判结果才进行融合
                    if len(judge_results) < 2: 
                        logger.warning(f"裁判结果数量不足(<2)，跳过融合. {inner_qa_answer_logger_str}, 裁判数量: {len(judge_results)}")
                        continue
                    
                    logger.info(f"开始融合裁判结果... {inner_qa_answer_logger_str}, 裁判结果数量: {len(judge_results)}")
                    # 创建融合器
                    logger.info(f"创建融合器... {inner_qa_answer_logger_str}")
                    aggregator = JudgeResultAggregator(
                        judge_results=judge_results,
                        standard_qa=qa,
                        original_text_raw=model_answer.answer_content,
                        threshold=0.6
                    )
                    
                    # 融合结果
                    logger.info(f"进行结果融合... {inner_qa_answer_logger_str}")
                    aggregated_result = aggregator.aggregate()
                    
                    # 保存融合结果到数据库
                    logger.info(f"保存融合结果到数据库... {inner_qa_answer_logger_str}")
                    save_aggregated_eval_result(
                        model_answer_id=model_answer.id,
                        threshold=0.6,
                        aggregated_result=aggregated_result
                    )
                    # 打印融合结果
                    result_dict = {
                        'key_points_evaluation': aggregated_result['key_points_evaluation'],
                        'factual_errors': aggregated_result['factual_errors'],
                        'vague_statements': aggregated_result['vague_statements'],
                        'partially_correct_but_misleading': aggregated_result['partially_correct_but_misleading'],
                        'irrelevant_but_correct': aggregated_result['irrelevant_but_correct'],
                        'stats': aggregated_result['stats'],
                        'judge_results': [result.to_dict() for result in aggregated_result['judge_results']]
                    }
                    result_json = json.dumps(result_dict, ensure_ascii=False, indent=2)
                    logger.info(f"融合并保存完成. {inner_qa_answer_logger_str}, 裁判数量: {len(judge_results)}. 融合结果: \n{result_json}")
                        
                except Exception as e:
                    logger.error(f"融合问答对 {qa.id} 时出错. {inner_qa_answer_logger_str}: {e}", exc_info=True)
                    continue
            
            logger.info(f"当前批次处理完成. {qa_answer_logger_str}")
            
        except Exception as e:
            logger.error(f"处理批次时出错. {model_logger_str}: {e} ", exc_info=True)
            break

def parse_args():
    """解析命令行参数。
    
    解析并返回命令行传入的各种参数配置，包括并发数、批处理大小、问答对来源等。
    
    Returns:
        argparse.Namespace: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(
        description='LLM问答评估结果融合工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""使用示例:
  1. 融合指定问答对的评估结果:
     python llm_qa_eval_fusion.py --qa-src list --qa-id-list 1,2,3,4,5
     python llm_qa_eval_fusion.py --qa-src list --qa-id-list 5
  
  2. 融合所有问答对的评估结果:
     python llm_qa_eval_fusion.py --qa-src all
  
  3. 调整批处理参数:
     python llm_qa_eval_fusion.py --limit-qa-count 100 --batch-size 10
  
  4. 启用调试模式:
     python llm_qa_eval_fusion.py --qa-src all --debug
     python llm_qa_eval_fusion.py --qa-src list --qa-id-list 1,2,3 --debug
  
  5. 调整并发数量:
     python llm_qa_eval_fusion.py --max-workers 3 --qa-src all
     python llm_qa_eval_fusion.py --max-workers 3 --qa-src list --qa-id-list 1,2,3
  
  6. 组合使用:
     python llm_qa_eval_fusion.py --max-workers 5 --limit-qa-count 202 --batch-size 100 --qa-src all --debug 
     python llm_qa_eval_fusion.py --max-workers 3 --limit-qa-count 50 --batch-size 4 --qa-src list --qa-id-list 1,2,3 --debug
"""
    )
    
    # 默认参数
    DEFAULT_MAX_WORKERS = 3
    DEFAULT_LIMIT_QA_COUNT = 20
    DEFAULT_BATCH_SIZE = 20
    DEFAULT_QA_SRC = 'all'

    # 最大并发数量
    parser.add_argument(
        '--max-workers',
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f'并发处理模型的最大线程数量 (默认: {DEFAULT_MAX_WORKERS})'
    )
    
    # 限制评估的问答对数量
    parser.add_argument(
        '--limit-qa-count',
        type=int,
        default=DEFAULT_LIMIT_QA_COUNT,
        help=f'限制评估的问答对数量 (默认: {DEFAULT_LIMIT_QA_COUNT})'
    )
    
    # 批处理大小
    parser.add_argument(
        '--batch-size',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f'每批从数据库中获取的问答对数量 (默认: {DEFAULT_BATCH_SIZE})'
    )

    # 问答对来源
    parser.add_argument(
        '--qa-src',
        type=str,
        choices=['all', 'list'],
        default=DEFAULT_QA_SRC,
        help=f'问答对来源: all 表示选择所有 (active or modified-active or pending-active) ；list 表示从qa_id_list作为问答来源 (默认: {DEFAULT_QA_SRC})'
    )

    # 问答对ID列表
    parser.add_argument(
        '--qa-id-list',
        type=str,
        default='5,6,7',
        help='问答对ID列表, 用逗号分隔, 仅在qa-src=list时有效'
    )
    
    # 调试模式
    parser.add_argument(
        '--debug',
        action='store_true',
        help='启用调试模式，包括使用秒级时间戳记录日志等; 默认关闭'
    )
    
    return parser.parse_args()

@timer
def main():
    try:
        # 解析命令行参数
        args = parse_args()

        # 配置日志器
        global logger
        logger = setup_logger(
            log_file=LOG_FILE,
            name=__name__,
            add_timestamp=True,
            timestamp_type=TimestampType.SECOND if args.debug else TimestampType.DAY,
            print_log_location=True
        )
        
        # 处理问答对来源参数
        qa_src = args.qa_src
        qa_id_list = []
        if qa_src == 'list':
            if not args.qa_id_list:
                raise ValueError("使用 --qa-src list 时必须指定 --qa-id-list")
            qa_id_list = [int(id) for id in args.qa_id_list.split(',')]
        
        # 输出运行参数
        logger.info("运行参数配置: ")
        if args.debug:
            logger.warning("运行模式: Debug")
        logger.info(f"  最大并发线程数量: {args.max_workers}")
        logger.info(f"  限制问答对数量: {args.limit_qa_count}")
        logger.info(f"  每批从数据库中获取的问答对数量: {args.batch_size}")
        logger.info(f"  问答对来源: {qa_src}")
        if qa_src == 'list':
            logger.info(f"  问答对ID列表: {qa_id_list}")

        # 被测模型配置文件路径
        EVAL_MODELS_CONFIG_FILE_PATH = CONFIG[CONFIG_NAME]['eval_models_config_file_path']
        ABS_EVAL_MODELS_CONFIG_FILE_PATH = os.path.abspath(EVAL_MODELS_CONFIG_FILE_PATH)
        # 裁判模型配置文件路径
        JUDGE_MODELS_CONFIG_FILE_PATH = CONFIG[CONFIG_NAME]['judge_models_config_file_path']
        ABS_JUDGE_MODELS_CONFIG_FILE_PATH = os.path.abspath(JUDGE_MODELS_CONFIG_FILE_PATH)

        # 初始化API密钥加密器
        logger.info(f"初始化API加解密器")
        apiKeyEncryptor = ApiKeyEncryptor(get_api_key_secret_key(CONFIG))
        
        # 加载初始化所有裁判模型
        logger.info(f"正在从 {ABS_JUDGE_MODELS_CONFIG_FILE_PATH} 加载裁判模型配置")
        jModelConfigs = load_models_configs(
            apiKeyEncryptor=apiKeyEncryptor,
            db_config=DB_CONFIG,
            file_path=ABS_JUDGE_MODELS_CONFIG_FILE_PATH
        )
        logger.info(f"已加载 {len(jModelConfigs)} 个裁判模型配置: {[jModelConfig.model_name for jModelConfig in jModelConfigs]}")
        judge_model_ids = [config.id for config in jModelConfigs]
        
        # 加载被测模型配置
        logger.info(f"正在从 {ABS_EVAL_MODELS_CONFIG_FILE_PATH} 加载被测模型配置")
        evalModelConfigList = load_models_configs(
            apiKeyEncryptor=apiKeyEncryptor,
            db_config=DB_CONFIG,
            file_path=ABS_EVAL_MODELS_CONFIG_FILE_PATH
        )
        logger.info(f"已加载 {len(evalModelConfigList)} 个被测模型配置: {[sModelConfig.model_name for sModelConfig in evalModelConfigList]}")
        
        # 使用线程池并发处理多个模型
        # 优化max_workers设置：取用户指定的worker数量和实际模型数量的最小值
        max_workers = min(args.max_workers, len(evalModelConfigList))
        logger.info(f"线程池配置: 用户指定worker数={args.max_workers}, 实际模型数={len(evalModelConfigList)}, 最终max_workers={max_workers}")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for model_config in evalModelConfigList:
                futures.append(executor.submit(
                    process_model_fusion,
                    model_config=model_config,
                    judge_model_ids=judge_model_ids,
                    qa_src=qa_src,
                    qa_id_list=qa_id_list,
                    batch_size=args.batch_size,
                    limit_qa_count=args.limit_qa_count
                ))
            
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"模型融合过程中出错: {e}", exc_info=True)
                    
    except Exception as e:
        logger.error(f"主程序执行出错: {e}", exc_info=True)

if __name__ == "__main__":
    main()