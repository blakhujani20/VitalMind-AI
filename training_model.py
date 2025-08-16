import os
import pickle
import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from src.preprocessing.data_loader import load_raw_data
from src.preprocessing.data_cleaning import clean_and_merge_data
from src.preprocessing.feature_engineering import create_features
from src.models.time_series_model import create_sequences, LSTMModel, train_model

# Config
LOOK_BACK = 7
EPOCHS = 150
LR = 0.001
BATCH_SIZE = 32
HIDDEN_SIZE = 50
SEED = 42

def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DATA_PATH = os.path.join(os.getcwd(), 'data')
    activity_df, sleep_df = load_raw_data(DATA_PATH)

    if activity_df is None or sleep_df is None:
        return

    cleaned_df = clean_and_merge_data(activity_df, sleep_df)
    final_df = create_features(cleaned_df)

    data = final_df['Steps'].values.astype(float)

    train_size = int(len(data) * 0.8)
    train_data, val_data = data[:train_size], data[train_size:]
    print(f"Training data size: {len(train_data)}, Validation data size: {len(val_data)}")

    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_norm = scaler.fit_transform(train_data.reshape(-1, 1)).flatten()
    full_norm = scaler.transform(data.reshape(-1, 1)).flatten()

    X_train, y_train = create_sequences(train_norm, look_back=LOOK_BACK)  
    X_train_t = torch.FloatTensor(X_train).unsqueeze(-1)                  
    y_train_t = torch.FloatTensor(y_train).view(-1, 1)                    

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    model = LSTMModel(input_size=1, hidden_layer_size=HIDDEN_SIZE, output_size=1).to(device)
    trained_model = train_model(model, train_loader, epochs=EPOCHS, lr=LR, device=device)


    val_start_idx = train_size - LOOK_BACK
    val_segment = full_norm[val_start_idx:]                     
    X_val_sw, y_val_sw = create_sequences(val_segment, LOOK_BACK)

    preds_sw = []
    trained_model.eval()
    with torch.no_grad():
        for seq in torch.FloatTensor(X_val_sw):
            seq = seq.view(LOOK_BACK, 1, 1).to(device)         
            pred = trained_model(seq)                       
            preds_sw.append(pred.item())

    preds_sw_inv = scaler.inverse_transform(np.array(preds_sw).reshape(-1, 1)).flatten()
    n_eval = min(len(preds_sw_inv), len(val_data))
    mse_sw = mean_squared_error(val_data[:n_eval], preds_sw_inv[:n_eval])
    rmse_sw = np.sqrt(mse_sw)
    print(f"✅ One-step (sliding-window) Validation MSE: {mse_sw:.2f}")
    print(f"   → RMSE ≈ {rmse_sw:.0f} steps")

    val_inputs = full_norm[train_size - LOOK_BACK:train_size].tolist()
    preds_ar = []
    with torch.no_grad():
        for _ in range(len(val_data)):
            seq = torch.FloatTensor(val_inputs[-LOOK_BACK:]).view(LOOK_BACK, 1, 1).to(device)
            pred = trained_model(seq)                           
            preds_ar.append(pred.item())
            val_inputs.append(pred.item())

    preds_ar_inv = scaler.inverse_transform(np.array(preds_ar).reshape(-1, 1)).flatten()
    mse_ar = mean_squared_error(val_data, preds_ar_inv)
    rmse_ar = np.sqrt(mse_ar)
    print(f"ℹ️  Autoregressive (multi-step) Validation MSE: {mse_ar:.2f}")
    print(f"   → RMSE ≈ {rmse_ar:.0f} steps (will typically be higher due to error compounding)")

    if len(val_data) >= 2:
        naive_pred = val_data[:-1]
        naive_true = val_data[1:]
        mse_naive = mean_squared_error(naive_true, naive_pred)
        rmse_naive = np.sqrt(mse_naive)
        print(f"🧪 Naive baseline RMSE (predict yesterday): {rmse_naive:.0f} steps")


    MODEL_DIR = os.path.join(os.getcwd(), 'models')
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = os.path.join(MODEL_DIR, 'lstm_steps_model.pth')
    scaler_path = os.path.join(MODEL_DIR, 'scaler_steps.pkl')

    torch.save(trained_model.state_dict(), model_path)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)


if __name__ == "__main__":
    main()
