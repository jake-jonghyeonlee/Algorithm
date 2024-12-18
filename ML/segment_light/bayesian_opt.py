import optuna
from optimize import train_model
import torch
from evaluation import SegmentationMetrics
import numpy as np
from visualization import save_optimization_plots

def objective(trial):
    # 하이퍼파라미터 정의
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_int('batch_size', 2, 16)
    num_workers = trial.suggest_int('num_workers', 0, 4)
    decay_rate = trial.suggest_float('decay_rate', 0.9, 0.99)
    num_epochs = trial.suggest_int('num_epochs', 30, 100)  # 에폭 범위 추가
    
    # Loss function 하이퍼파라미터
    loss_weights = {
        'bce': trial.suggest_float('bce_weight', 0.0, 1.0),
        'focal': trial.suggest_float('focal_weight', 0.0, 1.0),
        'dice': trial.suggest_float('dice_weight', 0.0, 1.0),
        'tversky': trial.suggest_float('tversky_weight', 0.0, 1.0)
    }
    
    # Focal Loss 파라미터
    focal_alpha = trial.suggest_float('focal_alpha', 0.1, 0.9)
    focal_gamma = trial.suggest_float('focal_gamma', 1.0, 5.0)
    
    # Tversky Loss 파라미터
    tversky_alpha = trial.suggest_float('tversky_alpha', 0.1, 0.9)
    tversky_beta = trial.suggest_float('tversky_beta', 0.1, 0.9)
    
    # Dice Loss 파라미터
    dice_smooth = trial.suggest_float('dice_smooth', 0.1, 2.0)
    
    # 데이터 경로
    data_dir = 'data'  # 실제 데이터 경로로 수정 필요
    
    try:
        # 학습률 스케줄러를 위한 파라미터 설정
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
        
        # 모델 학습
        model = train_model(
            data_dir=data_dir,
            num_epochs=num_epochs,  # 최적화된 에폭 수 사용
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_workers=num_workers,
            optimizer_params=optimizer_params,
            loss_params=loss_params  # Loss 파라미터 추가
        )
        
        # 검증 세트에서의 Dice Score를 반환
        checkpoint = torch.load('best_model.pkl')
        val_metrics = checkpoint['val_metrics']
        dice_score = val_metrics['Dice Coefficient']
        
        return dice_score
        
    except Exception as e:
        print(f"Trial failed: {e}")
        return float('-inf')

def run_optimization(n_trials=100):
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(),  # Tree-structured Parzen Estimators
        # TPESampler: 베이지안 최적화 기반 샘플러
        # - 이전 시도들의 결과를 바탕으로 다음 파라미터를 선택
        # - 성능이 좋은 영역에 더 집중하여 탐색
        # - 탐색(exploration)과 활용(exploitation)의 균형을 자동으로 조절
        
        pruner=optuna.pruners.MedianPruner()
        # MedianPruner: 중간값 기반 가지치기
        # - 현재 trial의 중간 성능이 이전 trials의 중간값보다 낮으면 조기 종료
        # - 학습 시간을 절약하고 비효율적인 파라미터 조합을 빠르게 제거
    )
    
    study.optimize(objective, n_trials=n_trials)
    
    print("Best trial:")
    trial = study.best_trial
    
    print(f"  Value (Dice Score): {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # 결과 저장
    study.trials_dataframe().to_csv('bayesian_optimization_results.csv')
    
    # 시각화 결과 저장
    save_optimization_plots(study, 'bayesian_optimization_plots')

if __name__ == "__main__":
    run_optimization() 