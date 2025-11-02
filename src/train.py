import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
from src.model import TransformerLM
from src.data import get_tokenizer_and_data, collate_fn
from src.utils import plot_loss_curves, plot_accuracy_table
import matplotlib.pyplot as plt

def train_with_accuracy(model, train_loader, val_loader, epochs=10, lr=3e-4, device='cuda'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        num_samples = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            preds = logits.argmax(dim=-1)
            masked_labels = labels[labels != -100]
            masked_preds = preds[labels != -100]
            if len(masked_labels) > 0:
                acc = (masked_preds == masked_labels).float().mean().item()
                epoch_acc += acc * len(masked_labels)
                num_samples += len(masked_labels)

        avg_train_loss = epoch_loss / len(train_loader)
        avg_train_acc = epoch_acc / num_samples if num_samples > 0 else 0
        train_losses.append(avg_train_loss)
        train_accs.append(avg_train_acc)

        model.eval()
        val_loss = 0
        val_acc = 0
        val_samples = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                logits = model(input_ids)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
                val_loss += loss.item()

                preds = logits.argmax(dim=-1)
                masked_labels = labels[labels != -100]
                masked_preds = preds[labels != -100]
                if len(masked_labels) > 0:
                    acc = (masked_preds == masked_labels).float().mean().item()
                    val_acc += acc * len(masked_labels)
                    val_samples += len(masked_labels)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / val_samples if val_samples > 0 else 0
        val_losses.append(avg_val_loss)
        val_accs.append(avg_val_acc)

        print(f"Epoch {epoch+1}, "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")

        scheduler.step()

    return train_losses, val_losses, train_accs, val_accs

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据
    tokenizer, train_data, val_data = get_tokenizer_and_data()
    collate = collate_fn(tokenizer)
    train_loader = DataLoader(train_data, batch_size=32, collate_fn=collate, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, collate_fn=collate)

    # 实验1：正弦位置编码
    print("🚀 Training with Sinusoidal Positional Encoding...")
    model_s = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        num_heads=8,
        num_layers=4,
        d_ff=1024,
        pos_encoding='sinusoidal'
    )
    train_losses_s, val_losses_s, train_accs_s, val_accs_s = train_with_accuracy(
        model_s, train_loader, val_loader, epochs=10, device=device
    )

    # 实验2：可学习位置编码
    print("🚀 Training with Learnable Positional Encoding...")
    model_l = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        num_heads=8,
        num_layers=4,
        d_ff=1024,
        pos_encoding='learnable'
    )
    train_losses_l, val_losses_l, train_accs_l, val_accs_l = train_with_accuracy(
        model_l, train_loader, val_loader, epochs=10, device=device
    )

    # 保存结果
    os.makedirs("results", exist_ok=True)
    plot_loss_curves(train_losses_s, val_losses_s, train_losses_l, val_losses_l)
    plt.savefig("results/loss_curve.png", dpi=300, bbox_inches='tight')
    plt.close()

    plot_accuracy_table(train_accs_s, val_accs_s, train_accs_l, val_accs_l)
    plt.savefig("results/accuracy_table.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 保存模型（可选）
    torch.save(model_s.state_dict(), "results/model_sinusoidal.pth")
    torch.save(model_l.state_dict(), "results/model_learnable.pth")

    print("✅ All experiments completed and results saved in ./results/")

if __name__ == "__main__":
    main()