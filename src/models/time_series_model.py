import torch
import torch.nn as nn
import numpy as np

# --- 1) Sequence Preparation ---
def create_sequences(data, look_back=3):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        y.append(data[i + look_back])
    return np.array(X), np.array(y).reshape(-1, 1)


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=False)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        device = input_seq.device
        batch_size = input_seq.size(1)

        h0 = torch.zeros(1, batch_size, self.hidden_layer_size, device=device)
        c0 = torch.zeros(1, batch_size, self.hidden_layer_size, device=device)

        lstm_out, _ = self.lstm(input_seq, (h0, c0))   
        last_t = lstm_out[-1]                          
        out = self.linear(last_t)                       
        return out.view(-1)                             



def train_model(model, train_loader, epochs=100, lr=0.001, device=None, clip_grad=1.0):

    if device is None:
        device = next(model.parameters()).device
    else:
        model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()

    for epoch in range(epochs):
        running_loss, n_batches = 0.0, 0

        for xb, yb in train_loader:
            xb = xb.to(device)                          
            yb = yb.view(-1).to(device)                 

            xb = xb.permute(1, 0, 2)                    

            optimizer.zero_grad()
            y_pred = model(xb)                          
            loss = criterion(y_pred, yb)

            loss.backward()
            if clip_grad is not None:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        avg_loss = running_loss / max(n_batches, 1)
        if (epoch + 1) % 25 == 0 or epoch == 0 or (epoch + 1) == epochs:
            print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.6f}")

    return model


def predict_future(model, data, steps=7, look_back=3, device=None):
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    future_predictions = []

    last_seq = torch.FloatTensor(data[-look_back:]).view(look_back, 1, 1).to(device)  

    with torch.no_grad():
        for _ in range(steps):
            pred = model(last_seq)                     
            future_predictions.append(pred.item())
            last_seq = torch.cat((last_seq[1:], pred.view(1, 1, 1)), dim=0)

    return future_predictions
