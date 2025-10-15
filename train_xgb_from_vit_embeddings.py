"""
ViT에서 추출한 CLS 토큰 임베딩을 XGBoost로 학습하는 모듈
"""

import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import xgboost as xgb
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def load_embeddings(parquet_path: str) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    ViT 임베딩 parquet 파일을 로드하고 피처/라벨 컬럼을 분리
    """
    print(f"📁 임베딩 파일 로드: {parquet_path}")
    
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"임베딩 파일이 없습니다: {parquet_path}")
    
    df = pd.read_parquet(parquet_path)
    print(f"   데이터 크기: {df.shape}")
    
    # 피처 컬럼 (f0, f1, f2, ...)과 라벨 컬럼 (y_*) 분리
    feature_cols = [col for col in df.columns if col.startswith('f') and col[1:].isdigit()]
    label_cols = [col for col in df.columns if col.startswith('y_')]
    
    print(f"   피처 차원: {len(feature_cols)}개")
    print(f"   라벨 개수: {len(label_cols)}개")
    print(f"   라벨들: {label_cols}")
    
    return df, feature_cols, label_cols

def train_xgboost_single_target(X: np.ndarray, y: np.ndarray, 
                               groups: np.ndarray = None,
                               xgb_params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    단일 타겟에 대해 XGBoost 학습 (K-Fold CV)
    """
    if xgb_params is None:
        xgb_params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 2025,
            'n_estimators': 100
        }
    
    # K-Fold 설정
    if groups is not None:
        # 그룹 수 확인
        unique_groups = len(np.unique(groups))
        n_splits = min(5, unique_groups)
        
        if unique_groups >= 2:
            # 그룹 기반 CV (같은 이력번호는 같은 fold)
            kf = GroupKFold(n_splits=n_splits)
            splits = list(kf.split(X, y, groups))
            print(f"     GroupKFold 사용: {n_splits}개 fold, {unique_groups}개 그룹")
        else:
            # 그룹이 1개뿐이면 일반 K-Fold 사용
            print(f"     그룹이 {unique_groups}개뿐이므로 일반 K-Fold 사용")
            kf = KFold(n_splits=5, shuffle=True, random_state=2025)
            splits = list(kf.split(X, y))
    else:
        # 일반 K-Fold
        kf = KFold(n_splits=5, shuffle=True, random_state=2025)
        splits = list(kf.split(X, y))
    
    cv_scores = []
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(splits):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # XGBoost 학습
        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X_train, y_train, 
                 eval_set=[(X_val, y_val)],
                 verbose=False)
        
        # 예측 및 평가
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)
        
        cv_scores.append({'fold': fold, 'rmse': rmse, 'r2': r2})
        models.append(model)
        
        print(f"     Fold {fold+1}: RMSE={rmse:.4f}, R²={r2:.4f}")
    
    # CV 결과 정리
    avg_rmse = np.mean([s['rmse'] for s in cv_scores])
    avg_r2 = np.mean([s['r2'] for s in cv_scores])
    std_rmse = np.std([s['rmse'] for s in cv_scores])
    std_r2 = np.std([s['r2'] for s in cv_scores])
    
    return {
        'models': models,
        'cv_scores': cv_scores,
        'avg_rmse': avg_rmse,
        'std_rmse': std_rmse,
        'avg_r2': avg_r2,
        'std_r2': std_r2
    }

def train_xgboost_multi_target(embedding_path: str, output_dir: str = "./xgb_results"):
    """
    ViT 임베딩을 사용해 다중 타겟 XGBoost 학습
    """
    print("=== ViT → XGBoost 학습 시작 ===\n")
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 임베딩 로드
    df, feature_cols, label_cols = load_embeddings(embedding_path)
    
    # 2. 데이터 준비
    X = df[feature_cols].values
    sample_ids = df['sample_id'].values
    
    # 그룹 정보 (이력번호 추출)
    groups = np.array([sid.split('_')[0] for sid in sample_ids])
    
    print(f"\n📊 데이터 준비 완료:")
    print(f"   샘플 수: {len(X)}")
    print(f"   피처 차원: {X.shape[1]}")
    print(f"   고유 그룹 수: {len(np.unique(groups))}")
    
    # 3. XGBoost 하이퍼파라미터
    xgb_params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 2025,
        'n_estimators': 100
    }
    
    # 4. 각 타겟별로 학습
    all_results = {}
    
    for label_col in label_cols:
        target_name = label_col.replace('y_', '')
        print(f"\n🎯 타겟: {target_name}")
        
        y = df[label_col].values
        
        # 결측치 처리
        valid_mask = ~np.isnan(y)
        if not valid_mask.all():
            print(f"   결측치 {np.sum(~valid_mask)}개 제거")
            X_clean = X[valid_mask]
            y_clean = y[valid_mask]
            groups_clean = groups[valid_mask]
        else:
            X_clean, y_clean, groups_clean = X, y, groups
        
        # XGBoost 학습
        result = train_xgboost_single_target(X_clean, y_clean, groups_clean, xgb_params)
        all_results[target_name] = result
        
        print(f"   평균 RMSE: {result['avg_rmse']:.4f} ± {result['std_rmse']:.4f}")
        print(f"   평균 R²: {result['avg_r2']:.4f} ± {result['std_r2']:.4f}")
        
        # 최고 성능 모델 저장
        best_fold = np.argmin([s['rmse'] for s in result['cv_scores']])
        best_model = result['models'][best_fold]
        
        model_path = os.path.join(output_dir, f"xgb_{target_name}.pkl")
        joblib.dump(best_model, model_path)
        print(f"   모델 저장: {model_path}")
    
    # 5. 전체 결과 정리
    summary = {
        'embedding_source': embedding_path,
        'feature_dim': X.shape[1],
        'n_samples': len(X),
        'n_groups': len(np.unique(groups)),
        'xgb_params': xgb_params,
        'results': {}
    }
    
    print(f"\n=== 전체 결과 요약 ===")
    for target_name, result in all_results.items():
        summary['results'][target_name] = {
            'avg_rmse': float(result['avg_rmse']),
            'std_rmse': float(result['std_rmse']),
            'avg_r2': float(result['avg_r2']),
            'std_r2': float(result['std_r2'])
        }
        
        print(f"{target_name:>15}: RMSE={result['avg_rmse']:.4f}±{result['std_rmse']:.4f}, R²={result['avg_r2']:.4f}±{result['std_r2']:.4f}")
    
    # 결과 저장
    summary_path = os.path.join(output_dir, "xgb_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 결과 저장: {summary_path}")
    print(f"🎉 XGBoost 학습 완료!")
    
    return all_results, summary

def predict_with_xgboost(embedding_path: str, model_dir: str = "./xgb_results", 
                        output_path: str = "./xgb_predictions.csv"):
    """
    학습된 XGBoost 모델로 예측
    """
    print("=== XGBoost 예측 시작 ===\n")
    
    # 임베딩 로드
    df, feature_cols, label_cols = load_embeddings(embedding_path)
    X = df[feature_cols].values
    sample_ids = df['sample_id'].values
    
    predictions = {'sample_id': sample_ids}
    
    # 각 타겟별 예측
    for label_col in label_cols:
        target_name = label_col.replace('y_', '')
        model_path = os.path.join(model_dir, f"xgb_{target_name}.pkl")
        
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            pred = model.predict(X)
            predictions[f"pred_{target_name}"] = pred
            
            # 실제 라벨도 포함
            if label_col in df.columns:
                predictions[f"true_{target_name}"] = df[label_col].values
            
            print(f"✅ {target_name} 예측 완료")
        else:
            print(f"❌ 모델 없음: {model_path}")
    
    # 결과 저장
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(output_path, index=False)
    
    print(f"\n💾 예측 결과 저장: {output_path}")
    return pred_df

if __name__ == "__main__":
    # 사용 예시
    embedding_file = "./runs_vit_light/test_embeddings.parquet"
    
    if os.path.exists(embedding_file):
        print("🚀 ViT 임베딩을 XGBoost로 학습합니다...")
        results, summary = train_xgboost_multi_target(embedding_file)
        
        print("\n🔮 예측도 실행합니다...")
        predictions = predict_with_xgboost(embedding_file)
        
    else:
        print(f"❌ 임베딩 파일이 없습니다: {embedding_file}")
        print("먼저 ViT 학습을 완료하세요: python train_vit_light.py --config config_vit_light_fixed.yaml")