"""
OCR Error Detector — Character-Level GRU Training

Trains a small GRU language model on clean ground-truth text.
Exports weights in int8 quantized format for the Zig runtime.

Architecture:
  Input:  one-hot character (96 ASCII printable)
  Hidden: GRU with configurable hidden size (default 64)
  Output: softmax over 96 characters

Output: data/processed/gru_weights.bin
  Format: [input_size:u32][hidden_size:u32][scale_w_ih:f32][scale_w_hh:f32]
          [scale_b_ih:f32][scale_b_hh:f32][w_ih: int8[3*hs*is]]
          [w_hh: int8[3*hs*hs]][b_ih: int8[3*hs]][b_hh: int8[3*hs]]
          [output_w: int8[is*hs]][output_b: int8[is]]
          [scale_out_w:f32][scale_out_b:f32]
"""

import struct
import sys
from pathlib import Path

import numpy as np

# Try importing torch — give helpful error if missing
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("ERROR: PyTorch is required for GRU training.")
    print("Install it with: pip install torch")
    sys.exit(1)

# Character set: ASCII printable (32-127)
CHAR_OFFSET = 32
VOCAB_SIZE = 96  # 127 - 32 + 1

def char_to_idx(c: str) -> int:
    """Convert character to index, clamping to valid range."""
    v = ord(c) - CHAR_OFFSET
    return max(0, min(v, VOCAB_SIZE - 1))

def idx_to_char(i: int) -> str:
    return chr(i + CHAR_OFFSET)

class CharDataset(Dataset):
    """Sliding window character-level dataset."""
    
    def __init__(self, text: str, seq_len: int = 64):
        self.data = [char_to_idx(c) for c in text if 32 <= ord(c) < 128]
        self.seq_len = seq_len
    
    def __len__(self):
        return max(0, len(self.data) - self.seq_len)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
        return x, y

class CharGRU(nn.Module):
    """Character-level GRU language model."""
    
    def __init__(self, vocab_size=VOCAB_SIZE, hidden_size=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, vocab_size)
        # Initialize embedding as one-hot (identity matrix)
        self.embedding.weight.data = torch.eye(vocab_size)
        self.embedding.weight.requires_grad = False  # Keep one-hot
        
        self.gru = nn.GRU(vocab_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden=None):
        # x: (batch, seq_len) -> (batch, seq_len, vocab_size)
        emb = self.embedding(x)
        out, hidden = self.gru(emb, hidden)
        logits = self.output(out)
        return logits, hidden

def quantize_to_int8(tensor: torch.Tensor) -> tuple[np.ndarray, float]:
    """Quantize a float tensor to int8, returning (quantized, scale)."""
    t = tensor.detach().cpu().numpy().flatten()
    max_abs = max(np.abs(t).max(), 1e-10)
    scale = max_abs / 127.0
    quantized = np.clip(np.round(t / scale), -128, 127).astype(np.int8)
    return quantized, float(scale)

def export_weights(model: CharGRU, output_path: Path):
    """Export GRU weights in int8 quantized binary format."""
    hs = model.hidden_size
    is_ = model.vocab_size
    
    # GRU weights: w_ih (3*hs, is), w_hh (3*hs, hs), b_ih (3*hs), b_hh (3*hs)
    w_ih = model.gru.weight_ih_l0.data
    w_hh = model.gru.weight_hh_l0.data
    b_ih = model.gru.bias_ih_l0.data
    b_hh = model.gru.bias_hh_l0.data
    
    # Output weights
    out_w = model.output.weight.data  # (is, hs)
    out_b = model.output.bias.data    # (is,)
    
    # Quantize
    q_w_ih, s_w_ih = quantize_to_int8(w_ih)
    q_w_hh, s_w_hh = quantize_to_int8(w_hh)
    q_b_ih, s_b_ih = quantize_to_int8(b_ih)
    q_b_hh, s_b_hh = quantize_to_int8(b_hh)
    q_out_w, s_out_w = quantize_to_int8(out_w)
    q_out_b, s_out_b = quantize_to_int8(out_b)
    
    with open(output_path, 'wb') as f:
        # Header
        f.write(struct.pack('<II', is_, hs))
        # Scales
        f.write(struct.pack('<ffffff', s_w_ih, s_w_hh, s_b_ih, s_b_hh, s_out_w, s_out_b))
        # Weights
        f.write(q_w_ih.tobytes())   # 3*hs*is bytes
        f.write(q_w_hh.tobytes())   # 3*hs*hs bytes
        f.write(q_b_ih.tobytes())   # 3*hs bytes
        f.write(q_b_hh.tobytes())   # 3*hs bytes
        f.write(q_out_w.tobytes())  # is*hs bytes
        f.write(q_out_b.tobytes())  # is bytes
    
    total_bytes = (2*4 + 6*4 + 3*hs*is_ + 3*hs*hs + 3*hs + 3*hs + is_*hs + is_)
    print(f"Exported weights to {output_path}")
    print(f"  Input size: {is_}, Hidden size: {hs}")
    print(f"  Total: {total_bytes:,} bytes ({total_bytes/1024:.1f} KB)")
    print(f"  Scales: w_ih={s_w_ih:.6f} w_hh={s_w_hh:.6f}")
    print(f"          b_ih={s_b_ih:.6f} b_hh={s_b_hh:.6f}")
    print(f"          out_w={s_out_w:.6f} out_b={s_out_b:.6f}")

def train():
    # Config
    hidden_size = 64
    seq_len = 64
    batch_size = 128
    epochs = 5
    lr = 0.002
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load training text
    text_path = Path("data/processed/gru_train.txt")
    if not text_path.exists():
        print("ERROR: Run preprocess.py first to generate training data.")
        sys.exit(1)
    
    text = text_path.read_text(encoding='utf-8')
    print(f"Training text: {len(text):,} characters")
    
    # Create dataset and dataloader
    dataset = CharDataset(text, seq_len)
    print(f"Dataset: {len(dataset):,} samples (seq_len={seq_len})")
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                       num_workers=0, drop_last=True)
    
    # Model
    model = CharGRU(VOCAB_SIZE, hidden_size).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {total_params:,} trainable parameters")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            
            logits, _ = model(x)
            # logits: (batch, seq_len, vocab) -> (batch*seq_len, vocab)
            loss = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / max(n_batches, 1)
        perplexity = np.exp(avg_loss) if avg_loss < 10 else float('inf')
        print(f"Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f}, Perplexity: {perplexity:.1f}")
    
    # Export
    out_dir = Path("data/processed")
    export_weights(model, out_dir / "gru_weights.bin")
    
    # Also save the full model for later use
    torch.save(model.state_dict(), out_dir / "gru_model.pt")
    print(f"Saved PyTorch model to {out_dir / 'gru_model.pt'}")

if __name__ == '__main__':
    train()
