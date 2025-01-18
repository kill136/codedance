from dataclasses import dataclass

@dataclass
class CodePretrainConfig:
    # 代码仓库相关配置
    repo_name: str = "nocobase"  
    repo_url: str = "https://github.com/nocobase/nocobase"
    code_dir: str = "./dataset/code_repos/nocobase"
    
    # 预处理配置
    max_file_size: int = 1024 * 1024  # 1MB
    min_file_size: int = 100  # 100B
    file_extensions: list = None
    excluded_dirs: list = None
    
    # 训练相关配置
    pretrain_batch_size: int = 4
    code_data_ratio: float = 0.3  # 每个batch中代码数据的比例
    code_special_tokens: dict = None
    
    def __post_init__(self):
        if self.file_extensions is None:
            self.file_extensions = ['.ts', '.tsx', '.js', '.jsx', '.md', '.json']
        if self.excluded_dirs is None:
            self.excluded_dirs = ['node_modules', 'dist', '.git']
        if self.code_special_tokens is None:
            self.code_special_tokens = {
                'repo_start': '<|repo_start|>',
                'repo_end': '<|repo_end|>',
                'file_start': '<|file_start|>',
                'file_end': '<|file_end|>'
            }