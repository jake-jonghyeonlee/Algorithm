import optuna
import matplotlib.pyplot as plt
import os

def save_optimization_plots(study, output_dir='optimization_plots'):
    """
    최적화 결과를 시각화하여 저장하는 함수
    
    Args:
        study: optuna study 객체
        output_dir: 결과를 저장할 디렉토리
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Parameter Importance
    try:
        importance = optuna.importance.get_param_importances(study)
        plt.figure(figsize=(10, 6))
        importance_values = list(importance.values())
        importance_labels = list(importance.keys())
        
        plt.barh(range(len(importance_values)), importance_values)
        plt.yticks(range(len(importance_labels)), importance_labels)
        plt.xlabel('Importance')
        plt.title('Hyperparameter Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'parameter_importance.png'))
        plt.close()
    except Exception as e:
        print(f"Failed to plot parameter importance: {e}")
    
    # 2. Optimization History
    plt.figure(figsize=(10, 6))
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.title('Optimization History')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'optimization_history.png'))
    plt.close()
    
    # 3. Slice Plot
    try:
        for param_name in study.best_trial.params.keys():
            plt.figure(figsize=(10, 6))
            optuna.visualization.matplotlib.plot_slice(study, params=[param_name])
            plt.title(f'Slice Plot: {param_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'slice_plot_{param_name}.png'))
            plt.close()
    except Exception as e:
        print(f"Failed to create slice plots: {e}")
    
    # 4. Contour Plots
    try:
        param_names = list(study.best_trial.params.keys())
        n_params = len(param_names)
        
        # 상위 5개 중요 파라미터만 선택
        if n_params > 5:
            importance = optuna.importance.get_param_importances(study)
            param_names = list(importance.keys())[:5]
        
        # 모든 가능한 파라미터 쌍에 대해 Contour Plot 생성
        for i in range(len(param_names)):
            for j in range(i + 1, len(param_names)):
                plt.figure(figsize=(10, 8))
                optuna.visualization.matplotlib.plot_contour(
                    study,
                    params=[param_names[i], param_names[j]]
                )
                plt.title(f'Contour Plot: {param_names[i]} vs {param_names[j]}')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 
                    f'contour_plot_{param_names[i]}_{param_names[j]}.png'))
                plt.close()
    except Exception as e:
        print(f"Failed to create contour plots: {e}") 