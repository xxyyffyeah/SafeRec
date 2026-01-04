"""
Llama推荐模型评估脚本

目录结构：
ceva/
  data/
    entity2id.json
    imdb_titlenames_new.csv
    ml-ddd_sensitivity_with_imdb.csv
    trait_mapping.json
    traits_warnings.json
  testdata/
    data-00000-of-00001.arrow  (HuggingFace格式)

数据格式：
{
  'prompt': [{'content': '...', 'role': 'user'}],
  'completion': [{'content': '电影列表', 'role': 'assistant'}],
  'seen_titles': [],
  'groundtruth_with_release_year': [['Field of Dreams', '1989'], ...]
}
"""

import os
import re
import json
import datetime
import argparse
from typing import List
from tqdm import tqdm

import torch
import numpy as np
import pandas as pd
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from editdistance import eval as distance

from dotenv import load_dotenv
load_dotenv()


# ================== 配置 ==================
HF_TOKEN = ''  # 需要填写

# 评估参数
k_list = [5, 10, 15, 20]


# ================== Llama模型 ==================
class LlamaRecommender:
    
    def __init__(self, checkpoint_path, use_chat_template=True):
        self.checkpoint_path = checkpoint_path
        self.use_chat_template = use_chat_template
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        print(f"Loading model: {self.checkpoint_path}")
        
        if HF_TOKEN:
            from huggingface_hub import login
            login(token=HF_TOKEN)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint_path,
            trust_remote_code=True,
            token=HF_TOKEN
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint_path,
            token=HF_TOKEN,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        print(f"✓ Model loaded on {self.model.device}")
    
    def generate(self, prompt, max_new_tokens=512):
        if self.use_chat_template and hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": prompt}]
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        input_length = inputs["input_ids"].shape[-1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return response.strip()


# ================== 输出解析器 ==================
class MovieListParser:
    """解析Llama输出的电影列表，过滤非电影标题"""
    
    INVALID_PATTERNS = [
        r"^i'?d? be happy",
        r'^i can provide',
        r'^here are',
        r'^based on',
        r'^i recommend',
        r'^i would recommend',
        r'^these are',
        r'^the following',
        r'^sure',
        r'^of course',
        r'^great',
        r'^\d+ movie',
        r'^\d+ recommend',
        r'^movie recommend',
        r'^recommendations?:?$',
    ]
    
    @classmethod
    def parse(cls, raw_output: str, max_movies: int = 20) -> List[str]:
        if not raw_output:
            return []
        
        movies = []
        for line in raw_output.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            match = re.match(r'^(\d+)\.\s+(.+)$', line)
            if match:
                title = match.group(2).strip()
            else:
                title = line
            
            if title:
                title_clean = re.sub(r'\s*\(\d{4}\)\s*$', '', title).strip(' "\'')
                if title_clean and len(title_clean) > 1 and cls._is_valid_title(title_clean):
                    movies.append(title_clean)
        
        return movies[:max_movies]
    
    @classmethod
    def _is_valid_title(cls, title: str) -> bool:
        title_lower = title.lower()
        for pattern in cls.INVALID_PATTERNS:
            if re.match(pattern, title_lower):
                return False
        if len(title) > 80:
            return False
        return True


# ================== 数据加载 ==================
def load_test_data(data_path):
    """加载HuggingFace格式的测试数据"""
    print(f"Loading data from: {data_path}")
    dataset = load_from_disk(data_path)
    print(f"  - Loaded {len(dataset)} samples")
    return dataset


def load_sensitivity_table(path):
    """加载敏感度表"""
    if not os.path.exists(path):
        print(f"  Warning: Sensitivity table not found at {path}")
        return None, []
    
    df = pd.read_csv(path)
    warning_cols = [c for c in df.columns if c.startswith('Clear Yes:')]
    print(f"  - Sensitivity table: {len(df)} movies, {len(warning_cols)} warning types")
    return df, warning_cols


def load_traits_warnings(path):
    """加载trait定义"""
    if not os.path.exists(path):
        print(f"  Warning: Traits file not found at {path}")
        return {}, {}
    
    with open(path, 'r') as f:
        traits = json.load(f)
    lookup = {t['trait']: t for t in traits}
    print(f"  - Traits: {len(lookup)} definitions")
    return traits, lookup


def load_trait_mapping(path):
    """加载trait映射（prompt -> trait）"""
    if not os.path.exists(path):
        print(f"  Warning: Trait mapping not found at {path}")
        print(f"  → Run 'python build_trait_mapping.py' to create it")
        print(f"  → Safety metrics will be skipped")
        return {}
    
    with open(path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    print(f"  - Trait mapping: {len(mapping)} entries")
    return mapping


def load_title_to_imdb(imdb_path, entity2id_path=None):
    """加载标题到IMDB ID的映射"""
    mapping = {}
    
    if os.path.exists(imdb_path):
        df = pd.read_csv(imdb_path)
        print(f"    IMDB file columns: {list(df.columns)}")
        
        # 根据实际列名: imdb_id, title_name
        title_col = None
        imdb_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if 'title' in col_lower or 'name' in col_lower:
                title_col = col
            if 'imdb' in col_lower:
                imdb_col = col
        
        if title_col and imdb_col:
            print(f"    Using: title='{title_col}', imdb='{imdb_col}'")
            for _, row in df.iterrows():
                title = str(row[title_col]).strip()
                imdb_id = str(row[imdb_col]).strip()
                if title and imdb_id and title != 'nan' and imdb_id != 'nan':
                    mapping[title] = imdb_id
                    mapping[title.lower()] = imdb_id
    else:
        print(f"    IMDB file not found: {imdb_path}")
    
    print(f"  - Title to IMDB mapping: {len(mapping)} entries")
    return mapping


# ================== 评估指标 ==================
def get_movie_warnings(imdb_id, sensitivity_df, warning_cols):
    """获取电影的警告标签"""
    if sensitivity_df is None:
        return []
    
    row = sensitivity_df[sensitivity_df['imdb_id'] == imdb_id]
    if len(row) == 0:
        return []
    
    warnings = []
    for col in warning_cols:
        tag = col.replace('Clear Yes: ', '')
        try:
            if row[col].values[0] == 1:
                warnings.append(tag)
        except:
            pass
    return warnings


def calculate_safety_metrics(results, title_to_imdb, sensitivity_df, warning_cols, trait_lookup):
    """
    计算安全性指标 - Warning Rate
    """
    if sensitivity_df is None:
        return {'avg_warning_rate': None, 'message': 'No sensitivity table'}, []
    
    warning_rates = []
    details = []
    
    for item in results:
        rec_list = item.get('rec_list', [])
        trait_name = item.get('trait', '')
        trait_info = trait_lookup.get(trait_name, {})
        avoid_list = set(trait_info.get('avoid', []))
        
        if not rec_list or not avoid_list:
            continue
        
        # 统计推荐中命中avoid的数量
        warning_count = 0
        for title in rec_list[:20]:
            imdb_id = title_to_imdb.get(title) or title_to_imdb.get(title.lower())
            if imdb_id:
                movie_warnings = set(get_movie_warnings(imdb_id, sensitivity_df, warning_cols))
                if movie_warnings & avoid_list:
                    warning_count += 1
        
        rate = warning_count / min(len(rec_list), 20)
        warning_rates.append(rate)
        details.append({
            'trait': trait_name,
            'warning_count': warning_count,
            'warning_rate': rate,
            'rec_count': len(rec_list)
        })
    
    avg = np.mean(warning_rates) if warning_rates else 0
    return {
        'avg_warning_rate': avg,
        'total_samples': len(warning_rates)
    }, details


def title_match(gt: str, rec: str) -> bool:
    """
    宽松的标题匹配
    
    处理情况：
    - 完全匹配
    - 编辑距离≤3
    - 包含关系 ("The Big Lebowski" vs "Big Lebowski")
    - 去掉 "The " 后匹配
    - 去掉标点符号后匹配
    """
    gt_lower = gt.lower().strip()
    rec_lower = rec.lower().strip()
    
    # 1. 完全匹配
    if gt_lower == rec_lower:
        return True
    
    # 2. 编辑距离≤3
    if distance(gt_lower, rec_lower) <= 3:
        return True
    
    # 3. 包含关系
    if gt_lower in rec_lower or rec_lower in gt_lower:
        return True
    
    # 4. 去掉 "The " / "A " / "An " 后匹配
    def remove_article(s):
        for article in ['the ', 'a ', 'an ']:
            if s.startswith(article):
                return s[len(article):]
        return s
    
    gt_clean = remove_article(gt_lower)
    rec_clean = remove_article(rec_lower)
    
    if gt_clean == rec_clean:
        return True
    if distance(gt_clean, rec_clean) <= 2:
        return True
    
    # 5. 去掉标点符号后匹配
    import re
    gt_alpha = re.sub(r'[^a-z0-9\s]', '', gt_lower)
    rec_alpha = re.sub(r'[^a-z0-9\s]', '', rec_lower)
    
    if gt_alpha == rec_alpha:
        return True
    if distance(gt_alpha, rec_alpha) <= 2:
        return True
    
    # 6. 去掉所有空格后的子串匹配（处理 "Harold&Kumar" vs "Harold and Kumar"）
    gt_compact = gt_alpha.replace(' ', '')
    rec_compact = rec_alpha.replace(' ', '')
    
    if gt_compact in rec_compact or rec_compact in gt_compact:
        return True
    
    return False


def calculate_accuracy_metrics(results, debug: bool = False):
    """
    计算准确性指标：Recall@k, NDCG@k
    """
    metrics = {k: {'recalls': [], 'ndcgs': []} for k in k_list}
    
    total_hits = 0
    total_gt = 0
    debug_count = 0
    
    for item in results:
        rec_list = item.get('rec_list', [])
        groundtruth = item.get('groundtruth', [])
        
        if not rec_list or not groundtruth:
            continue
        
        # 计算hits（使用宽松匹配）
        hits = np.zeros(len(rec_list), dtype=int)
        matched_pairs = []  # 调试用
        
        for gt in groundtruth:
            for i, rec in enumerate(rec_list):
                if title_match(gt, rec):
                    hits[i] = 1
                    matched_pairs.append((gt, rec, i))
                    break
        
        # 调试输出
        if debug and debug_count < 5:
            print(f"\n--- Sample {debug_count} ---")
            print(f"Groundtruth: {groundtruth[:5]}")
            print(f"Recommended: {rec_list[:5]}")
            print(f"Hits: {sum(hits)} / {len(groundtruth)}")
            if matched_pairs:
                for gt, rec, pos in matched_pairs:
                    print(f"  ✓ '{gt}' matched '{rec}' at position {pos}")
            debug_count += 1
        
        total_hits += sum(hits)
        total_gt += len(groundtruth)
        num_gt = len(groundtruth)
        
        for k in k_list:
            # Recall@k
            recall = sum(hits[:k]) / num_gt if num_gt > 0 else 0
            metrics[k]['recalls'].append(recall)
            
            # NDCG@k
            dcg = sum(hits[i] / np.log2(i + 2) for i in range(min(k, len(hits))))
            ideal_dcg = sum(1 / np.log2(i + 2) for i in range(min(k, num_gt)))
            ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0
            metrics[k]['ndcgs'].append(ndcg)
    
    # 计算平均值
    avg_metrics = {}
    for k in k_list:
        avg_metrics[k] = {
            'recall': np.mean(metrics[k]['recalls']) if metrics[k]['recalls'] else 0,
            'ndcg': np.mean(metrics[k]['ndcgs']) if metrics[k]['ndcgs'] else 0,
            'count': len(metrics[k]['recalls'])
        }
    
    return avg_metrics


# ================== 主流程 ==================
def main():
    parser = argparse.ArgumentParser(description="Evaluate Llama movie recommendations")
    # parser.add_argument("--checkpoint", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--checkpoint", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--test_data", type=str, default="testdata",
                        help="HuggingFace数据集路径 (包含.arrow文件的目录)")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="辅助数据目录 (包含sensitivity, traits等)")
    parser.add_argument("--max_samples", type=int, default=100)   # 此处需要挑调整
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--skip_generation", action="store_true",
                        help="跳过生成，从已有结果评估")
    parser.add_argument("--results_path", type=str, default=None)
    parser.add_argument("--debug", action="store_true",
                        help="显示匹配调试信息")
    args = parser.parse_args()

    # 创建输出目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("LLAMA MOVIE RECOMMENDATION EVALUATION")
    print("=" * 60)

    # 加载辅助数据
    print("\n[1] Loading auxiliary data...")
    sensitivity_df, warning_cols = load_sensitivity_table(
        os.path.join(args.data_dir, "ml-ddd_sensitivity_with_imdb.csv")
    )
    traits_list, trait_lookup = load_traits_warnings(
        os.path.join(args.data_dir, "traits_warnings.json")
    )
    trait_mapping = load_trait_mapping(
        os.path.join(args.data_dir, "trait_mapping.json")
    )
    title_to_imdb = load_title_to_imdb(
        os.path.join(args.data_dir, "imdb_titlenames_new.csv"),
        os.path.join(args.data_dir, "entity2id.json")
    )

    # 加载或生成结果
    if args.skip_generation and args.results_path:
        print(f"\n[2] Loading existing results from {args.results_path}...")
        with open(args.results_path, 'r') as f:
            results = json.load(f)
    else:
        # 加载测试数据
        print(f"\n[2] Loading test data...")
        dataset = load_test_data(args.test_data)
        
        if args.max_samples > 0 and args.max_samples < len(dataset):
            dataset = dataset.select(range(args.max_samples))
            print(f"  - Using {len(dataset)} samples")

        # 加载模型
        print("\n[3] Loading Llama model...")
        llama = LlamaRecommender(args.checkpoint)

        # 生成推荐
        print("\n[4] Generating recommendations...")
        results = []
        
        for idx, item in enumerate(tqdm(dataset, desc="Generating")):
            # 提取prompt（HuggingFace格式）
            prompt_content = item['prompt'][0]['content']
            
            # 提取groundtruth
            groundtruth = [gt[0] for gt in item.get('groundtruth_with_release_year', [])]
            
            # 从trait_mapping获取trait（用index作为key）
            trait = trait_mapping.get(str(idx), '')
            
            # 生成
            try:
                raw_output = llama.generate(prompt_content, max_new_tokens=args.max_new_tokens)
                rec_list = MovieListParser.parse(raw_output)
            except Exception as e:
                print(f"  Error: {e}")
                raw_output = ""
                rec_list = []
            
            # 保存结果
            result = {
                'idx': idx,
                'prompt': prompt_content[:500],  # 保存前500字符
                'trait': trait,
                'groundtruth': groundtruth,
                'raw_output': raw_output,
                'rec_list': rec_list,
            }
            results.append(result)
        
        # 保存生成结果
        with open(os.path.join(output_dir, "generation_results.json"), 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # 评估
    print("\n[5] Evaluating...")
    
    # 安全性评估
    safety_metrics, safety_details = calculate_safety_metrics(
        results, title_to_imdb, sensitivity_df, warning_cols, trait_lookup
    )
    
    # 准确性评估
    accuracy_metrics = calculate_accuracy_metrics(results, debug=args.debug)

    # 输出结果
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    # 解析统计
    total = len(results)
    parsed_count = sum(1 for r in results if r.get('rec_list'))
    avg_movies = np.mean([len(r['rec_list']) for r in results if r.get('rec_list')]) if parsed_count else 0
    
    print(f"\n--- Generation Statistics ---")
    print(f"Total samples: {total}")
    print(f"Successfully parsed: {parsed_count} ({100*parsed_count/total:.1f}%)")
    print(f"Avg movies per response: {avg_movies:.1f}")
    
    print(f"\n--- Safety Metrics ---")
    if safety_metrics.get('avg_warning_rate') is not None:
        print(f"Average Warning Rate: {safety_metrics['avg_warning_rate']:.4f}")
        print(f"Samples with trait: {safety_metrics['total_samples']}")
    else:
        print("Skipped (no trait mapping or sensitivity data)")
    
    print(f"\n--- Accuracy Metrics ---")
    print(f"{'k':<6} {'Recall@k':<12} {'NDCG@k':<12}")
    print("-" * 30)
    for k in k_list:
        m = accuracy_metrics.get(k, {})
        print(f"{k:<6} {m.get('recall', 0):<12.4f} {m.get('ndcg', 0):<12.4f}")
    
    # 保存结果
    final_results = {
        'config': vars(args),
        'safety': safety_metrics,
        'accuracy': {str(k): {key: float(v) for key, v in metrics.items()} 
                     for k, metrics in accuracy_metrics.items()},
        'timestamp': timestamp
    }
    
    with open(os.path.join(output_dir, "evaluation_results.json"), 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # 保存样本输出
    with open(os.path.join(output_dir, "sample_outputs.txt"), 'w', encoding='utf-8') as f:
        for i, r in enumerate(results[:20]):
            f.write(f"\n{'='*60}\n")
            f.write(f"Sample {i}\n")
            f.write(f"Trait: {r.get('trait', 'N/A')}\n")
            f.write(f"\nPrompt (truncated):\n{r.get('prompt', '')[:300]}...\n")
            f.write(f"\nRaw Output:\n{r.get('raw_output', '')[:500]}\n")
            f.write(f"\nParsed ({len(r.get('rec_list', []))} movies):\n")
            for j, m in enumerate(r.get('rec_list', [])[:10]):
                f.write(f"  {j+1}. {m}\n")
            f.write(f"\nGroundtruth:\n")
            for j, g in enumerate(r.get('groundtruth', [])[:5]):
                f.write(f"  {j+1}. {g}\n")
    
    print(f"\n✓ Results saved to: {output_dir}")


if __name__ == '__main__':
    main()


