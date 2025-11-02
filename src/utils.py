import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoTokenizer

def plot_loss_curves(train_losses_s, val_losses_s, train_losses_l, val_losses_l):
    epochs = range(1, len(train_losses_s) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses_s, 'bo-', label='Sinusoidal Train')
    plt.plot(epochs, val_losses_s, 'ro-', label='Sinusoidal Val')
    plt.title('Sinusoidal PE')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_losses_l, 'bo-', label='Learnable Train')
    plt.plot(epochs, val_losses_l, 'ro-', label='Learnable Val')
    plt.title('Learnable PE')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend()
    plt.tight_layout()

def plot_accuracy_table(train_accs_s, val_accs_s, train_accs_l, val_accs_l):
    epochs = list(range(1, 11))
    data = {
        "Epoch": epochs,
        "Sinusoidal - Train Acc": [f"{acc:.4f}" for acc in train_accs_s],
        "Sinusoidal - Val Acc": [f"{acc:.4f}" for acc in val_accs_s],
        "Learnable - Train Acc": [f"{acc:.4f}" for acc in train_accs_l],
        "Learnable - Val Acc": [f"{acc:.4f}" for acc in val_accs_l],
    }
    df = pd.DataFrame(data)
    print("📊 准确率表格：\n", df)
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight'); ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1.2, 1.6)
    plt.title("Accuracy per Epoch: Sinusoidal vs Learnable PE", y=1.1, fontsize=16)

def generate_text(model, tokenizer, prompt, max_length=50, temperature=1.0, top_k=50, top_p=0.95, device='cuda'):
    model.eval()
    model.to(device)
    tokens = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=max_length - 10)
    input_ids = tokens['input_ids'].to(device)
    with torch.no_grad():
        for _ in range(max_length - input_ids.size(1)):
            logits = model(input_ids)[:, -1, :] / temperature
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits = logits.masked_fill(indices_to_remove, float('-inf'))
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits = logits.masked_fill(indices_to_remove, float('-inf'))
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)