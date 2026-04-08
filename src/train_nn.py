import h5py
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=50):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class SpeechDetectorTransformer(nn.Module):
    def __init__(
        self,
        num_landmarks=80,
        coords=2,
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.1,
    ):
        super().__init__()
        self.input_dim = num_landmarks * coords

        self.embedding = nn.Linear(self.input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=38)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        batch_size, seq_len, num_points, dims = x.size()

        x = x.view(batch_size, seq_len, num_points * dims)

        src_key_padding_mask = x.abs().sum(dim=-1) == 0

        x = self.embedding(x)
        x = self.pos_encoder(x)

        out = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        valid_mask = (~src_key_padding_mask).unsqueeze(-1).float()

        sum_valid = valid_mask.sum(dim=1).clamp(min=1e-9)

        pooled = (out * valid_mask).sum(dim=1) / sum_valid

        logits = self.classifier(pooled)
        return logits


data_path = Path("data/lip_landmarks_2d.h5")
model_save_path = Path("models/speech_detector_transformer_2d_weights.pth")
model_save_path.parent.mkdir(parents=True, exist_ok=True)


def run_training(X_train, y_train, X_val, y_val):

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpeechDetectorTransformer().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimiser = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    best_val_loss = float("inf")

    epochs = 200
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0

        for batch_X, batch_y in train_dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimiser.zero_grad()
            predictions = model(batch_X)

            loss = criterion(predictions, batch_y)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

            train_loss += loss.item() * batch_X.size(0)

            preds = (torch.sigmoid(predictions) > 0.5).float()
            train_correct += (preds == batch_y).sum().item()

        avg_train_loss = train_loss / len(train_dataset)
        avg_train_acc = train_correct / len(train_dataset)

        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for v_batch_X, v_batch_y in val_dataloader:
                v_batch_X, v_batch_y = v_batch_X.to(device), v_batch_y.to(device)

                v_predictions = model(v_batch_X)

                v_loss = criterion(v_predictions, v_batch_y)

                val_loss += v_loss.item() * v_batch_X.size(0)

                v_preds = (torch.sigmoid(v_predictions) > 0.5).float()
                val_correct += (v_preds == v_batch_y).sum().item()

        avg_val_loss = val_loss / len(val_dataset)
        avg_val_acc = val_correct / len(val_dataset)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)

        print(f"Epoch [{epoch + 1}/{epochs}]")
        print(f"  Train Loss: {avg_train_loss:.4f} | Acc: {avg_train_acc:.2%}")
        print(f"  Val Loss:     {avg_val_loss:.4f} | Acc: {avg_val_acc:.2%}")
        print("-" * 30)


def run_testing(X_test, y_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpeechDetectorTransformer().to(device)
    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    total_correct = 0
    with torch.no_grad():
        for batch_X, batch_y in test_dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            predictions = model(batch_X)

            preds = (torch.sigmoid(predictions) > 0.5).float()
            total_correct += (preds == batch_y).sum().item()

    print(f"\nFinal Accuracy on Test Set: {total_correct / len(test_dataset):.2%}")


if __name__ == "__main__":
    with h5py.File(data_path, "r") as rf:
        print(list(rf.keys()))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)

        X_train = rf["x_train"][:]
        y_train = rf["y_train"][:]

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
        )
        run_training(X_train, y_train, X_val, y_val)

        X_test = rf["x_test"][:]
        y_test = rf["y_test"][:]

        run_testing(X_test, y_test)
