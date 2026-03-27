import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path


class SpeechDetectorNN(nn.Module):
    def __init__(self, input_size=160, hidden_size=64, num_layers=2, dropout=0.2):
        super(SpeechDetectorNN, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        # self.fc1 = nn.Linear(hidden_size, 32)
        # self.relu = nn.ReLU()
        # self.fc2 = nn.Linear(32, 1)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x input shape: (batch_size, 38, 80, 2)
        batch_size, seq_len, num_points, dims = x.size()

        # Flatten to (batch_size, 38, 160)
        x = x.view(batch_size, seq_len, num_points * dims)

        output, (hn, cn) = self.lstm(x)
        final_frame_output = output[:, -1, :]

        logits = self.fc(final_frame_output)
        return logits


data_path = Path("data/lip_landmarks.h5")
model_save_path = Path("models/speech_detector_weights.pth")
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
    model = SpeechDetectorNN().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimiser = optim.Adam(model.parameters(), lr=0.001)

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
    model = SpeechDetectorNN().to(device)
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
