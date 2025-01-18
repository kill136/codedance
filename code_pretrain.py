import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from utils.code_processor import CodeRepoProcessor, CodePretrainDataset
from config.code_pretrain_config import CodePretrainConfig
from model.model import Transformer, LMConfig
from utils.logger import Logger
import argparse
from transformers import AutoTokenizer

def init_model_and_tokenizer(pretrained_path, device):
    """初始化模型和tokenizer"""
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 使用与1-pretrain.py相同的模型配置
    lm_config = LMConfig()
    
    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    
    # 设置padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        Logger(f"设置 pad_token 为 eos_token: {tokenizer.pad_token}")
    
    # 初始化模型
    model = Transformer(lm_config)
    
    # 加载预训练权重
    state_dict = torch.load(pretrained_path, map_location=device)
    # 处理可能的DDP前缀
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict, strict=False)
    Logger(f'成功加载预训练权重: {pretrained_path}')
    Logger(f'LLM总参数量：{count_parameters(model) / 1e6:.3f} 百万')
    
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description="Code Repository Specific Pretraining")
    parser.add_argument("--pretrained_path", type=str, default="./out/pretrain.pth",
                      help="预训练模型权重路径")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                  help="Learning rate for fine-tuning")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--save_dir", type=str, default="./out/code_pretrain")
    args = parser.parse_args()
    
    # 1. 加载配置
    config = CodePretrainConfig()
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 2. 初始化模型和tokenizer
    model, tokenizer = init_model_and_tokenizer(args.pretrained_path, args.device)
    model = model.to(args.device)
    
    # 3. 处理代码仓库
    processor = CodeRepoProcessor(config)
    processed_files = processor.process_repo()
    Logger(f'处理完成代码文件数量: {len(processed_files)}')
    
    # 4. 创建数据集和加载器
    dataset = CodePretrainDataset(
        processed_files=processed_files,
        tokenizer=tokenizer,
        config=config,
        max_length=5120
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # 5. 优化器和学习率设置
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=len(dataloader) * args.num_epochs
    )
    
    # 6. 训练循环
    model.train()
    best_loss = float('inf')
    
    # 添加混合精度训练
    scaler = torch.cuda.amp.GradScaler()
    ctx = torch.cuda.amp.autocast()
    
    for epoch in range(args.num_epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            
            # 前向传播
            with ctx:
                # 确保输入和标签的形状正确
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids  # 使用输入作为标签
                )
                
                # 获取损失
                loss = outputs.loss  # 这应该是一个标量值
                
                # 如果loss仍然是None，我们可以手动计算交叉熵损失
                if loss is None:
                    logits = outputs.logits
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = input_ids[..., 1:].contiguous()
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                                  shift_labels.view(-1))
                    
                    # 应用attention mask
                    if attention_mask is not None:
                        # 调整mask以匹配shifted序列
                        shift_mask = attention_mask[..., 1:].contiguous()
                        loss_mask = shift_mask.view(-1).float()
                        loss = torch.sum(loss * loss_mask) / loss_mask.sum()
            
            # 反向传播
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # 梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # 优化器步进
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                Logger(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(dataloader)
        Logger(f'Epoch {epoch} Average Loss: {avg_loss:.4f}')
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 
                      os.path.join(args.save_dir, 'code_pretrain_best.pth'))
        
        # 定期保存检查点
        if (epoch + 1) % 1 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(args.save_dir, f'code_pretrain_epoch_{epoch}.pth'))

if __name__ == "__main__":
    main() 