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
    def __init__(self, processed_files: List[Dict], tokenizer, config: CodePretrainConfig, max_length: int = 512, stride: int = 256):
        """
        Args:
            processed_files: 处理后的代码文件列表
            tokenizer: tokenizer实例
            config: CodePretrainConfig实例
            max_length: 最大序列长度
            stride: 滑动窗口的步长，决定相邻片段的重叠程度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.config = config
        self.stride = stride
        
        # 预处理所有文件，将长文件分割成多个片段
        self.segments = []
        
        for file_data in processed_files:
            # 构建完整输入文本
            text = self._construct_input_text(file_data)
            
            # 获取完整文本的token
            tokens = tokenizer(text, truncation=False)
            input_ids = tokens['input_ids']
            
            # 如果文本长度小于max_length，直接添加
            if len(input_ids) <= max_length:
                self.segments.append({
                    'input_ids': input_ids,
                    'file_info': file_data
                })
            else:
                # 使用滑动窗口分割长文本
                for start in range(0, len(input_ids), stride):
                    end = start + max_length
                    segment = input_ids[start:end]
                    
                    # 确保最后一个片段长度为max_length
                    if len(segment) < max_length:
                        if start > 0:  # 不是第一个片段
                            # 从末尾往前取max_length个token
                            segment = input_ids[-max_length:]
                        else:  # 文本总长度小于max_length
                            continue
                    
                    self.segments.append({
                        'input_ids': segment,
                        'file_info': {
                            **file_data,
                            'segment_start': start,
                            'is_full_file': len(input_ids) <= max_length
                        }
                    })
                   
                    # 如果这是最后一个片段，跳出循环
                    if end >= len(input_ids):
                        break
        # 输出segments 到文件
        with open('./dataset/pretrain_data.csv', 'w', encoding='utf-8') as f:
            for segment in self.segments:
                f.write(f"{segment['input_ids']}\n")
                f.write(f"{segment['file_info']}\n")
                
        print(f"总文件数: {len(processed_files)}, 生成片段数: {len(self.segments)}")
    
    def _construct_input_text(self, file_data):
        """构建输入文本"""
        return (
            f"{self.config.code_special_tokens['repo_start']}"
            f"{self.config.repo_name}\n"
            f"{self.config.code_special_tokens['file_start']}"
            f"{file_data['path']}\n"
            f"{file_data['content']}"
            f"{self.config.code_special_tokens['file_end']}"
            f"{self.config.code_special_tokens['repo_end']}"
        )
        
    def __len__(self):
        return len(self.segments)
        
    def __getitem__(self, idx):
        segment = self.segments[idx]
        input_ids = torch.tensor(segment['input_ids'])
        
        # 如果长度不足max_length，进行padding
        if len(input_ids) < self.max_length:
            padding_length = self.max_length - len(input_ids)
            input_ids = torch.cat([
                input_ids,
                torch.full((padding_length,), self.tokenizer.pad_token_id)
            ])
        
        # 创建attention mask
        attention_mask = torch.ones_like(input_ids)
        attention_mask[input_ids == self.tokenizer.pad_token_id] = 0
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        } 