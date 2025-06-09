# test.py
import torch
import numpy as np

from data_preprocess import load_data, load_config
import torch.nn.functional as F
from visualization.confusion_matrix import plot_confusion
from utils.save_results import save_team_results, evaluate_results
from model.best_ecapa import build_model
from torch.utils.data import TensorDataset, DataLoader

def test():
    cfg = load_config()
    
    # --------- Device -------------
    """
    Defalut : mps
    If you wanna use cuda, modify config.yaml cuda: True
    """
    if cfg.get('cuda'):
        device = torch.device('cuda' if torch.backends.cuda.is_available() else 'cpu')
        
    else:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print("device:", device)
    # ----------- Dataset -----------
    
    cfg['use_specaugment']=False
    X_test, y_test, fnames = load_data(
        cfg['test_metadata_path'], cfg['test_data_path'],
        cfg.get('sr', 16000), cfg.get('n_mfcc', 20), cfg
    )

    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.int64)
    X_tensor = torch.from_numpy(X_test).unsqueeze(1).to(device) 
    test_dataset = TensorDataset(X_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(">>> X_test.shape (numpy):", X_test.shape)
    print(">>> X_tensor.shape (torch):", X_tensor.shape) 
    # shape: (num_samples, 1, n_mels, time_steps)


    # --------- Model ------------
    n_mels = X_tensor.size(2)
    dropout_p = cfg.get('dropout_p')
    depth = cfg.get('depth')
    model = build_model(
        input_dim=n_mels,
        hidden1=cfg.get('ecapa_channels', 512),
        hidden2=cfg.get('ecapa_emb_dim', 192),
        num_classes=cfg.get('num_classes', 2),
        dropout_p=dropout_p,
        depth=depth
    ).to(device)
    model.load_state_dict(
        torch.load(cfg.get('model_path_for_test', 'model.pth'), map_location=device)
    )

    # -------- Test ---------
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            xb = batch[0].to(device)  # shape: (batch_size, 1, n_mels, time_steps)
            _, logits = model(xb)
            prob_real = logits.softmax(1)[:, 1].cpu()
            batch_preds = (prob_real >= 0.5).long().numpy()
            all_preds.append(batch_preds)

    preds = np.concatenate(all_preds, axis=0)

    # Confusion matrix visualization
    plot_confusion(
        y_test, preds,
        labels=cfg.get('label_names', ['fake', 'real']),
        save_path=cfg.get('cm_path', None)
    )

    # Save and evaluate results
    result_txt = cfg.get('result_txt', 'team_test_result.txt')
    save_team_results(fnames, preds, cfg.get('label_names', ['fake','real']), result_txt)
    evaluate_results(
        cfg.get('eval_script_path', './2501ml_data/eval.pl'),
        result_txt,
        cfg['test_metadata_path']
    )

if __name__ == '__main__':
    test()
