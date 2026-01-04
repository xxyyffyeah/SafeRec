"""
构建 trait_mapping.json

流程：
1. prompt → Llama → 电影推荐列表
2. 每部电影 → 查 sensitivity 表 → 获取触发的敏感标签
3. 敏感标签 → 映射到 traits_warnings 中的大类
4. 统计最多的大类 → 作为该prompt的trait
5. 用数据集index作为key

使用：
python build_trait_mapping.py --test_data test --checkpoint meta-llama/Llama-3.2-3B-Instruct
"""

import os
import re
import json
import argparse
from typing import List, Dict, Tuple
from collections import Counter
from tqdm import tqdm

import torch
import pandas as pd
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM

from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = ''


# ================== Llama模型 ==================
class LlamaRecommender:
    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
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
        print(f"✓ Model loaded")
    
    def generate(self, prompt, max_new_tokens=512):
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        
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


# ================== 电影解析器 ==================
class MovieListParser:
    """解析Llama输出的电影列表，过滤非电影标题"""
    
    # 非电影标题的模式
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
            
            # 格式: "1. Movie Title (2011)" 或 "Movie Title (2011)"
            match = re.match(r'^(\d+)\.\s+(.+)$', line)
            if match:
                title = match.group(2).strip()
            else:
                # 无编号格式
                title = line
            
            if title:
                # 去掉年份括号
                title_clean = re.sub(r'\s*\(\d{4}\)\s*$', '', title).strip(' "\'')
                
                # 过滤无效标题
                if title_clean and len(title_clean) > 1 and cls._is_valid_title(title_clean):
                    movies.append(title_clean)
        
        return movies[:max_movies]
    
    @classmethod
    def _is_valid_title(cls, title: str) -> bool:
        """检查是否是有效的电影标题"""
        title_lower = title.lower()
        
        # 检查无效模式
        for pattern in cls.INVALID_PATTERNS:
            if re.match(pattern, title_lower):
                return False
        
        # 太长的可能是句子而非标题
        if len(title) > 80:
            return False
        
        return True


# ================== 数据加载 ==================
def load_sensitivity_table(path: str) -> Tuple[pd.DataFrame, List[str]]:
    """加载敏感度表"""
    df = pd.read_csv(path)
    # 找到所有 "Clear Yes: xxx" 格式的列
    warning_cols = [c for c in df.columns if c.startswith('Clear Yes:')]
    print(f"  - Sensitivity table: {len(df)} movies, {len(warning_cols)} warning columns")
    return df, warning_cols


def load_traits_warnings(path: str) -> Tuple[List[Dict], Dict[str, List[str]]]:
    """
    加载traits_warnings.json，构建小标签→大类的反向映射
    
    traits_warnings格式:
    [
      {"trait": "depression", "avoid": ["Suicide", "Self-harm", ...]},
      ...
    ]
    
    返回:
      - traits_list: 原始列表
      - tag_to_traits: {小标签: [大类1, 大类2, ...]}
    """
    with open(path, 'r') as f:
        traits_list = json.load(f)
    
    # 构建反向映射: 小标签 → 大类列表
    tag_to_traits = {}
    for t in traits_list:
        trait_name = t['trait']
        avoid_tags = t.get('avoid', [])
        for tag in avoid_tags:
            if tag not in tag_to_traits:
                tag_to_traits[tag] = []
            tag_to_traits[tag].append(trait_name)
    
    print(f"  - Traits: {len(traits_list)} traits, {len(tag_to_traits)} tag mappings")
    return traits_list, tag_to_traits


def build_title_to_imdb(sensitivity_df: pd.DataFrame, imdb_titles_path: str = None) -> Dict[str, str]:
    """构建标题→IMDB ID映射"""
    mapping = {}
    
    # sensitivity表没有title，只能从imdb_titlenames加载
    if imdb_titles_path and os.path.exists(imdb_titles_path):
        df = pd.read_csv(imdb_titles_path)
        print(f"    IMDB titles file columns: {list(df.columns)}")
        
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
            print(f"    Using columns: title='{title_col}', imdb='{imdb_col}'")
            for _, row in df.iterrows():
                title = str(row[title_col]).strip()
                imdb_id = str(row[imdb_col]).strip()
                if title and imdb_id and title != 'nan' and imdb_id != 'nan':
                    mapping[title] = imdb_id
                    mapping[title.lower()] = imdb_id
        else:
            print(f"    Warning: Could not find title/imdb columns")
    else:
        print(f"    Warning: IMDB titles file not found: {imdb_titles_path}")
    
    print(f"  - Title to IMDB mapping: {len(mapping)} entries")
    
    # 打印几个样本
    if mapping:
        samples = list(mapping.items())[:3]
        print(f"    Samples: {samples}")
    
    return mapping


# ================== 核心逻辑 ==================
def get_movie_sensitivity_tags(
    movie_title: str,
    title_to_imdb: Dict[str, str],
    sensitivity_df: pd.DataFrame,
    warning_cols: List[str]
) -> List[str]:
    """
    获取一部电影触发的敏感标签
    
    返回: ["Violence", "Drug Use", ...] (去掉 "Clear Yes: " 前缀)
    """
    # 查找IMDB ID
    imdb_id = title_to_imdb.get(movie_title) or title_to_imdb.get(movie_title.lower())
    if not imdb_id:
        return []
    
    # 在sensitivity表中查找
    row = sensitivity_df[sensitivity_df['imdb_id'] == imdb_id]
    if len(row) == 0:
        return []
    
    # 收集值为1的标签
    triggered_tags = []
    for col in warning_cols:
        try:
            if row[col].values[0] == 1:
                tag = col.replace('Clear Yes: ', '')
                triggered_tags.append(tag)
        except:
            pass
    
    return triggered_tags


def determine_trait_from_movies(
    movie_list: List[str],
    title_to_imdb: Dict[str, str],
    sensitivity_df: pd.DataFrame,
    warning_cols: List[str],
    tag_to_traits: Dict[str, List[str]]
) -> Tuple[str, Dict, List[str]]:
    """
    根据推荐的电影列表，确定最可能的trait
    
    返回: (trait_name, stats_dict, unmatched_movies)
    """
    # 统计所有触发的大类
    trait_counter = Counter()
    tag_counter = Counter()
    unmatched = []
    matched_count = 0
    
    for movie in movie_list:
        # 先检查是否能找到IMDB ID
        imdb_id = title_to_imdb.get(movie) or title_to_imdb.get(movie.lower())
        if not imdb_id:
            unmatched.append(movie)
            continue
        
        matched_count += 1
        tags = get_movie_sensitivity_tags(movie, title_to_imdb, sensitivity_df, warning_cols)
        for tag in tags:
            tag_counter[tag] += 1
            # 映射到大类
            if tag in tag_to_traits:
                for trait in tag_to_traits[tag]:
                    trait_counter[trait] += 1
    
    # 找出触发最多的大类
    if trait_counter:
        most_common_trait = trait_counter.most_common(1)[0][0]
    else:
        most_common_trait = ''
    
    stats = {
        'trait_counts': dict(trait_counter),
        'tag_counts': dict(tag_counter),
        'num_movies': len(movie_list),
        'matched_count': matched_count
    }
    
    return most_common_trait, stats, unmatched


# ================== 主流程 ==================
def build_trait_mapping(
    test_data_path: str,
    checkpoint: str,
    sensitivity_path: str,
    traits_warnings_path: str,
    imdb_titles_path: str,
    output_path: str,
    max_samples: int = None,
    max_new_tokens: int = 512
):
    print("=" * 60)
    print("BUILD TRAIT MAPPING")
    print("=" * 60)
    
    # 1. 加载辅助数据
    print("\n[1] Loading auxiliary data...")
    sensitivity_df, warning_cols = load_sensitivity_table(sensitivity_path)
    traits_list, tag_to_traits = load_traits_warnings(traits_warnings_path)
    title_to_imdb = build_title_to_imdb(sensitivity_df, imdb_titles_path)
    
    # 2. 加载测试数据
    print("\n[2] Loading test data...")
    dataset = load_from_disk(test_data_path)
    print(f"  - Loaded {len(dataset)} samples")
    
    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
        print(f"  - Using first {max_samples} samples")
    
    # 3. 加载Llama模型
    print("\n[3] Loading Llama model...")
    llama = LlamaRecommender(checkpoint)
    
    # 4. 生成并分析
    print("\n[4] Generating recommendations and analyzing...")
    trait_mapping = {}
    all_stats = []
    unmatched_movies = set()  # 收集未匹配的电影
    
    for idx, item in enumerate(tqdm(dataset, desc="Processing")):
        prompt = item['prompt'][0]['content']
        
        # 生成推荐
        try:
            raw_output = llama.generate(prompt, max_new_tokens=max_new_tokens)
            movie_list = MovieListParser.parse(raw_output)
        except Exception as e:
            print(f"  Error at {idx}: {e}")
            movie_list = []
        
        # 分析trait
        trait, stats, unmatched = determine_trait_from_movies(
            movie_list,
            title_to_imdb,
            sensitivity_df,
            warning_cols,
            tag_to_traits
        )
        unmatched_movies.update(unmatched)
        
        # 保存（用index作为key）
        trait_mapping[str(idx)] = trait
        
        stats['index'] = idx
        stats['movie_list'] = movie_list[:5]  # 只保存前5部用于调试
        stats['trait'] = trait
        all_stats.append(stats)
    
    # 5. 保存结果
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(trait_mapping, f, indent=2, ensure_ascii=False)
    
    # 保存详细统计
    stats_path = output_path.replace('.json', '_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    
    # 6. 输出统计
    print(f"\n{'='*60}")
    print("STATISTICS")
    print(f"{'='*60}")
    
    trait_dist = Counter(trait_mapping.values())
    total = len(trait_mapping)
    with_trait = sum(1 for v in trait_mapping.values() if v)
    
    print(f"Total samples: {total}")
    print(f"Samples with trait: {with_trait} ({100*with_trait/total:.1f}%)")
    print(f"\nTrait distribution:")
    for trait, count in trait_dist.most_common():
        label = trait if trait else "(none)"
        print(f"  {label}: {count}")
    
    print(f"\n--- Sample results ---")
    for s in all_stats[:5]:
        print(f"\nIndex: {s['index']}")
        print(f"  Movies: {s['movie_list']}")
        print(f"  Matched: {s.get('matched_count', 0)}/{s['num_movies']}")
        print(f"  Top tags: {dict(Counter(s['tag_counts']).most_common(3))}")
        print(f"  Trait: {s['trait']}")
    
    # 显示未匹配的电影样本
    if unmatched_movies:
        print(f"\n--- Unmatched movies (sample of {min(20, len(unmatched_movies))}) ---")
        for m in list(unmatched_movies)[:20]:
            print(f"  - {m}")
    
    print(f"\n✓ Saved trait_mapping to: {output_path}")
    print(f"✓ Saved stats to: {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="Build trait mapping using Llama")
    parser.add_argument("--test_data", type=str, default="testdata",
                        help="HuggingFace数据集路径")
    # parser.add_argument("--checkpoint", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
    #                     help="Llama模型路径")
    parser.add_argument("--checkpoint", type=str, default="Qwen/Qwen2.5-0.5B",
                        help="qwen模型路径")
    parser.add_argument("--sensitivity", type=str, default="data/ml-ddd_sensitivity_with_imdb.csv",
                        help="敏感度表路径")
    parser.add_argument("--traits_warnings", type=str, default="data/traits_warnings.json",
                        help="traits_warnings.json路径")
    parser.add_argument("--imdb_titles", type=str, default="data/imdb_titlenames_new.csv",
                        help="IMDB标题映射文件路径")
    parser.add_argument("--output", type=str, default="data/trait_mapping.json",
                        help="输出路径")
    parser.add_argument("--max_samples", type=int, default=100,
                        help="最大处理样本数")   # 此处需要调整
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Llama生成最大token数")
    args = parser.parse_args()
    
    build_trait_mapping(
        test_data_path=args.test_data,
        checkpoint=args.checkpoint,
        sensitivity_path=args.sensitivity,
        traits_warnings_path=args.traits_warnings,
        imdb_titles_path=args.imdb_titles,
        output_path=args.output,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens
    )


if __name__ == '__main__':
    main()