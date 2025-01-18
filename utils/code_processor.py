import os
import json
from typing import List, Dict
import torch
from torch.utils.data import Dataset
from pathlib import Path
from config.code_pretrain_config import CodePretrainConfig

class CodeRepoProcessor:
    def __init__(self, config: CodePretrainConfig):
        self.config = config
        # 添加处理后数据的保存路径
        self.processed_data_dir = Path("./dataset/processed_code")
        self.processed_data_path = self.processed_data_dir / f"{config.repo_name}_processed.json"
        
        # 创建保存目录
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    def process_file(self, file_path: str) -> Dict:
        """处理单个代码文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            file_size = os.path.getsize(file_path)
            if not (self.config.min_file_size <= file_size <= self.config.max_file_size):
                return None
                
            # 获取相对路径，便于跨环境使用
            rel_path = os.path.relpath(file_path, self.config.code_dir)
                
            return {
                'path': rel_path,
                'content': content,
                'size': file_size,
                'type': os.path.splitext(file_path)[1],
                'repo': self.config.repo_name
            }
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")
            return None
    
    def process_repo(self) -> List[Dict]:
        """处理整个代码仓库"""
        # 如果已存在处理后的数据，直接加载
        if self.processed_data_path.exists():
            print(f"加载已处理的数据: {self.processed_data_path}")
            with open(self.processed_data_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        processed_files = []
        total_files = 0
        processed_count = 0
        
        for root, _, files in os.walk(self.config.code_dir):
            # 跳过排除的目录
            if any(x in root for x in self.config.excluded_dirs):
                continue
                
            for file in files:
                if os.path.splitext(file)[1] not in self.config.file_extensions:
                    continue
                    
                total_files += 1
                file_path = os.path.join(root, file)
                processed = self.process_file(file_path)
                if processed:
                    processed_files.append(processed)
                    processed_count += 1
                    
                if processed_count % 100 == 0:
                    print(f"已处理 {processed_count}/{total_files} 个文件...")
        
        # 保存处理后的数据
        print(f"保存处理后的数据到: {self.processed_data_path}")
        with open(self.processed_data_path, 'w', encoding='utf-8') as f:
            json.dump(processed_files, f, ensure_ascii=False, indent=2)
        
        # 保存处理统计信息
        stats = {
            'total_files': total_files,
            'processed_files': processed_count,
            'repo_name': self.config.repo_name,
            'file_types': {}
        }
        
        for file in processed_files:
            file_type = file['type']
            if file_type not in stats['file_types']:
                stats['file_types'][file_type] = 0
            stats['file_types'][file_type] += 1
        
        stats_path = self.processed_data_dir / f"{self.config.repo_name}_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
            
        return processed_files

class CodePretrainDataset(Dataset):
    def __init__(self, processed_files: List[Dict], tokenizer, config: CodePretrainConfig, max_length: int = 512):
        """
        Args:
            processed_files: 处理后的代码文件列表
            tokenizer: tokenizer实例
            config: CodePretrainConfig实例
            max_length: 最大序列长度
        """
        self.processed_files = processed_files
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.config = config
        
    def __len__(self):
        return len(self.processed_files)
        
    def __getitem__(self, idx):
        file_data = self.processed_files[idx]
        
        # 构建输入文本
        text = (
            f"{self.config.code_special_tokens['repo_start']}"
            f"{self.config.repo_name}\n"
            f"{self.config.code_special_tokens['file_start']}"
            f"{file_data['path']}\n"
            f"{file_data['content']}"
            f"{self.config.code_special_tokens['file_end']}"
            f"{self.config.code_special_tokens['repo_end']}"
        )
        
        # 编码文本
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
            pad_to_max_length=True
        )
        
        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze()
        } 