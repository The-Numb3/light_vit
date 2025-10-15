"""
XGBoost 모델의 성능을 종합적으로 평가하는 모듈
"""

import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

plt.rcParams['font.family'] = ['Malgun Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class XGBoostEvaluator:
    def __init__(self, model_dir: str = "./xgb_results"):
        self.model_dir = model_dir
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """저장된 XGBoost 모델들을 로드"""
        print(f"📁 모델 로딩: {self.model_dir}")
        
        if not os.path.exists(self.model_dir):
            print(f"❌ 모델 디렉토리가 없습니다: {self.model_dir}")
            return
        
        model_files = [f for f in os.listdir(self.model_dir) if f.startswith('xgb_') and f.endswith('.pkl')]
        
        for model_file in model_files:
            target_name = model_file.replace('xgb_', '').replace('.pkl', '')
            model_path = os.path.join(self.model_dir, model_file)
            
            try:
                self.models[target_name] = joblib.load(model_path)
                print(f"  ✅ {target_name} 모델 로드 완료")
            except Exception as e:
                print(f"  ❌ {target_name} 모델 로드 실패: {e}")
        
        print(f"총 {len(self.models)}개 모델 로드 완료\n")
    
    def evaluate_predictions(self, predictions_path: str = "./xgb_predictions.csv") -> Dict:
        """예측 결과를 평가"""
        print("=== XGBoost 예측 성능 평가 ===\n")
        
        if not os.path.exists(predictions_path):
            print(f"❌ 예측 파일이 없습니다: {predictions_path}")
            return {}
        
        df = pd.read_csv(predictions_path)
        print(f"📊 예측 데이터: {len(df)}개 샘플")
        
        # 타겟별 성능 계산
        evaluation_results = {}
        
        for target in self.models.keys():
            true_col = f"true_{target}"
            pred_col = f"pred_{target}"
            
            if true_col in df.columns and pred_col in df.columns:
                y_true = df[true_col].values
                y_pred = df[pred_col].values
                
                # 결측치 제거
                mask = ~(np.isnan(y_true) | np.isnan(y_pred))
                y_true_clean = y_true[mask]
                y_pred_clean = y_pred[mask]
                
                if len(y_true_clean) > 0:
                    # 성능 메트릭 계산
                    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
                    mae = mean_absolute_error(y_true_clean, y_pred_clean)
                    r2 = r2_score(y_true_clean, y_pred_clean)
                    
                    # 추가 메트릭
                    mape = np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100
                    residuals = y_true_clean - y_pred_clean
                    
                    evaluation_results[target] = {
                        'n_samples': len(y_true_clean),
                        'rmse': rmse,
                        'mae': mae,
                        'r2': r2,
                        'mape': mape,
                        'mean_true': np.mean(y_true_clean),
                        'std_true': np.std(y_true_clean),
                        'mean_pred': np.mean(y_pred_clean),
                        'std_pred': np.std(y_pred_clean),
                        'residuals_mean': np.mean(residuals),
                        'residuals_std': np.std(residuals)
                    }
                    
                    print(f"🎯 {target}:")
                    print(f"   RMSE: {rmse:.4f}")
                    print(f"   MAE:  {mae:.4f}")
                    print(f"   R²:   {r2:.4f}")
                    print(f"   MAPE: {mape:.2f}%")
                    print(f"   샘플: {len(y_true_clean)}개\n")
        
        return evaluation_results
    
    def create_performance_plots(self, predictions_path: str = "./xgb_predictions.csv", 
                               output_dir: str = "./xgb_evaluation"):
        """성능 시각화 플롯 생성"""
        print("=== 성능 시각화 생성 ===\n")
        
        os.makedirs(output_dir, exist_ok=True)
        
        if not os.path.exists(predictions_path):
            print(f"❌ 예측 파일이 없습니다: {predictions_path}")
            return
        
        df = pd.read_csv(predictions_path)
        
        for target in self.models.keys():
            true_col = f"true_{target}"
            pred_col = f"pred_{target}"
            
            if true_col in df.columns and pred_col in df.columns:
                y_true = df[true_col].values
                y_pred = df[pred_col].values
                
                # 결측치 제거
                mask = ~(np.isnan(y_true) | np.isnan(y_pred))
                y_true_clean = y_true[mask]
                y_pred_clean = y_pred[mask]
                
                if len(y_true_clean) > 0:
                    # 플롯 생성
                    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                    fig.suptitle(f'{target} 예측 성능 분석', fontsize=16, fontweight='bold')
                    
                    # 1. 실제값 vs 예측값 산점도
                    axes[0, 0].scatter(y_true_clean, y_pred_clean, alpha=0.6, s=50)
                    axes[0, 0].plot([y_true_clean.min(), y_true_clean.max()], 
                                   [y_true_clean.min(), y_true_clean.max()], 'r--', lw=2)
                    axes[0, 0].set_xlabel('실제값')
                    axes[0, 0].set_ylabel('예측값')
                    axes[0, 0].set_title('실제값 vs 예측값')
                    
                    # R² 표시
                    r2 = r2_score(y_true_clean, y_pred_clean)
                    axes[0, 0].text(0.05, 0.95, f'R² = {r2:.3f}', 
                                   transform=axes[0, 0].transAxes, 
                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    # 2. 잔차 플롯
                    residuals = y_true_clean - y_pred_clean
                    axes[0, 1].scatter(y_pred_clean, residuals, alpha=0.6, s=50)
                    axes[0, 1].axhline(y=0, color='r', linestyle='--')
                    axes[0, 1].set_xlabel('예측값')
                    axes[0, 1].set_ylabel('잔차 (실제값 - 예측값)')
                    axes[0, 1].set_title('잔차 플롯')
                    
                    # 3. 잔차 히스토그램
                    axes[1, 0].hist(residuals, bins=20, alpha=0.7, edgecolor='black')
                    axes[1, 0].axvline(x=0, color='r', linestyle='--')
                    axes[1, 0].set_xlabel('잔차')
                    axes[1, 0].set_ylabel('빈도')
                    axes[1, 0].set_title('잔차 분포')
                    
                    # 4. 예측값 분포 비교
                    axes[1, 1].hist(y_true_clean, bins=15, alpha=0.7, label='실제값', color='blue')
                    axes[1, 1].hist(y_pred_clean, bins=15, alpha=0.7, label='예측값', color='red')
                    axes[1, 1].set_xlabel('값')
                    axes[1, 1].set_ylabel('빈도')
                    axes[1, 1].set_title('실제값 vs 예측값 분포')
                    axes[1, 1].legend()
                    
                    plt.tight_layout()
                    
                    # 저장
                    plot_path = os.path.join(output_dir, f"{target}_performance.png")
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print(f"📈 {target} 플롯 저장: {plot_path}")
    
    def feature_importance_analysis(self, output_dir: str = "./xgb_evaluation"):
        """피처 중요도 분석"""
        print("\n=== 피처 중요도 분석 ===\n")
        
        os.makedirs(output_dir, exist_ok=True)
        
        all_importance = {}
        
        for target, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                all_importance[target] = importance
                
                # 상위 20개 피처 플롯
                top_indices = np.argsort(importance)[-20:]
                top_importance = importance[top_indices]
                top_features = [f'f{i}' for i in top_indices]
                
                plt.figure(figsize=(10, 8))
                plt.barh(range(len(top_features)), top_importance)
                plt.yticks(range(len(top_features)), top_features)
                plt.xlabel('Feature Importance')
                plt.title(f'{target} - Top 20 Feature Importance')
                plt.tight_layout()
                
                importance_path = os.path.join(output_dir, f"{target}_feature_importance.png")
                plt.savefig(importance_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"📊 {target} 피처 중요도 저장: {importance_path}")
        
        return all_importance
    
    def comprehensive_evaluation_report(self, embedding_path: str = "./runs_vit_light/test_embeddings.parquet",
                                      predictions_path: str = "./xgb_predictions.csv",
                                      output_dir: str = "./xgb_evaluation"):
        """종합 평가 리포트 생성"""
        print("=== 종합 평가 리포트 생성 ===\n")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 성능 평가
        evaluation_results = self.evaluate_predictions(predictions_path)
        
        # 2. 시각화 생성
        self.create_performance_plots(predictions_path, output_dir)
        
        # 3. 피처 중요도 분석
        feature_importance = self.feature_importance_analysis(output_dir)
        
        # 4. 종합 리포트 생성
        report = {
            'evaluation_results': evaluation_results,
            'summary': {
                'n_targets': len(evaluation_results),
                'targets': list(evaluation_results.keys()),
                'avg_r2': np.mean([r['r2'] for r in evaluation_results.values()]) if evaluation_results else 0,
                'avg_rmse': np.mean([r['rmse'] for r in evaluation_results.values()]) if evaluation_results else 0
            }
        }
        
        # 리포트 저장
        report_path = os.path.join(output_dir, "comprehensive_evaluation_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"📋 종합 리포트 저장: {report_path}")
        
        # 요약 출력
        print(f"\n=== 성능 요약 ===")
        print(f"평가 타겟 수: {len(evaluation_results)}")
        if evaluation_results:
            print(f"평균 R²: {report['summary']['avg_r2']:.4f}")
            print(f"평균 RMSE: {report['summary']['avg_rmse']:.4f}")
            
            print(f"\n타겟별 R² 순위:")
            sorted_targets = sorted(evaluation_results.items(), key=lambda x: x[1]['r2'], reverse=True)
            for i, (target, metrics) in enumerate(sorted_targets, 1):
                print(f"  {i}. {target}: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}")
        
        return report

def main():
    """실험 실행"""
    print("🧪 XGBoost 성능 평가 실험 시작\n")
    
    # 평가기 초기화
    evaluator = XGBoostEvaluator()
    
    if not evaluator.models:
        print("❌ 로드된 모델이 없습니다. 먼저 XGBoost 학습을 실행하세요.")
        return
    
    # 종합 평가 실행
    report = evaluator.comprehensive_evaluation_report()
    
    print(f"\n🎉 평가 완료! 결과는 './xgb_evaluation' 폴더를 확인하세요.")

if __name__ == "__main__":
    main()