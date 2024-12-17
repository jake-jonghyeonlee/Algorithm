import optuna
from optimize import train_model
import torch
from evaluation import SegmentationMetrics
import numpy as np

def objective(trial):
    # 기존 하이퍼파라미터 옵션
    lr_options = [1e-4, 5e-4, 1e-3, 5e-3]
    batch_size_options = [2, 4, 8, 16]
    num_workers_options = [0, 2, 4]
    decay_rate_options = [0.9, 0.95, 0.99]
    epoch_options = [30, 50, 70, 100]
    
    # Loss function 하이퍼파라미터 옵션
    weight_options = [0.0, 0.5, 1.0]
    focal_alpha_options = [0.25, 0.5, 0.75]
    focal_gamma_options = [2.0, 3.0, 4.0]
    tversky_alpha_options = [0.3, 0.5, 0.7]
    tversky_beta_options = [0.3, 0.5, 0.7]
    dice_smooth_options = [0.5, 1.0, 1.5]
    
    # 파라미터 선택
    learning_rate = trial.suggest_categorical('learning_rate', lr_options)
    batch_size = trial.suggest_categorical('batch_size', batch_size_options)
    num_workers = trial.suggest_categorical('num_workers', num_workers_options)
    decay_rate = trial.suggest_categorical('decay_rate', decay_rate_options)
    num_epochs = trial.suggest_categorical('num_epochs', epoch_options)
    
    # Loss function 파라미터 선택
    loss_weights = {
        'bce': trial.suggest_categorical('bce_weight', weight_options),
        'focal': trial.suggest_categorical('focal_weight', weight_options),
        'dice': trial.suggest_categorical('dice_weight', weight_options),
        'tversky': trial.suggest_categorical('tversky_weight', weight_options)
    }
    
    focal_alpha = trial.suggest_categorical('focal_alpha', focal_alpha_options)
    focal_gamma = trial.suggest_categorical('focal_gamma', focal_gamma_options)
    tversky_alpha = trial.suggest_categorical('tversky_alpha', tversky_alpha_options)
    tversky_beta = trial.suggest_categorical('tversky_beta', tversky_beta_options)
    dice_smooth = trial.suggest_categorical('dice_smooth', dice_smooth_options)
    
    try:
        optimizer_params = {
            'lr': learning_rate,
            'weight_decay': decay_rate
        }
        
        # Loss function 파라미터 설정
        loss_params = {
            'weights': loss_weights,
            'focal_params': {'alpha': focal_alpha, 'gamma': focal_gamma},
            'tversky_params': {'alpha': tversky_alpha, 'beta': tversky_beta},
            'dice_params': {'smooth': dice_smooth}
        }
        
        model = train_model(
            data_dir=data_dir,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_workers=num_workers,
            optimizer_params=optimizer_params,
            loss_params=loss_params  # Loss 파라미터 추가
        )
        
        checkpoint = torch.load('best_model.pkl')
        val_metrics = checkpoint['val_metrics']
        dice_score = val_metrics['Dice Coefficient']
        
        return dice_score
        
    except Exception as e:
        print(f"Trial failed: {e}")
        return float('-inf')

def run_optimization(n_trials=None):
    # 모든 가능한 조합의 수 계산 (Loss 파라미터 추가)
    total_combinations = (4 * 4 * 3 * 3 * 4 *  # 기존 파라미터
                        3 * 3 * 3 * 3 *        # loss weights
                        3 * 3 * 3 * 3 * 3)     # loss function 파라미터
    
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.GridSampler({
            'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3],
            'batch_size': [2, 4, 8, 16],
            'num_workers': [0, 2, 4],
            'decay_rate': [0.9, 0.95, 0.99],
            'num_epochs': [30, 50, 70, 100],
            # Loss weights
            'bce_weight': [0.0, 0.5, 1.0],
            'focal_weight': [0.0, 0.5, 1.0],
            'dice_weight': [0.0, 0.5, 1.0],
            'tversky_weight': [0.0, 0.5, 1.0],
            # Loss function 파라미터
            'focal_alpha': [0.25, 0.5, 0.75],
            'focal_gamma': [2.0, 3.0, 4.0],
            'tversky_alpha': [0.3, 0.5, 0.7],
            'tversky_beta': [0.3, 0.5, 0.7],
            'dice_smooth': [0.5, 1.0, 1.5]
        })
    )
    
    # 모든 조합을 시도
    study.optimize(objective, n_trials=total_combinations)
    
    print("Best trial:")
    trial = study.best_trial
    
    print(f"  Value (Dice Score): {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # 결과 저장
    study.trials_dataframe().to_csv('grid_search_results.csv')

if __name__ == "__main__":
    run_optimization() 