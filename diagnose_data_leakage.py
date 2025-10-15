"""
데이터 누출 및 과적합 문제를 진단하는 스크립트
"""

import pandas as pd
import numpy as np
import json

def analyze_data_leakage():
    print("=== 데이터 누출 및 과적합 진단 ===\n")
    
    # 1. ViT 학습 시 데이터 분할 확인
    print("1️⃣ ViT 학습 시 데이터 분할 상황")
    
    # ViT 메트릭 파일 확인
    with open('./runs_vit_light/metrics.json', 'r') as f:
        vit_metrics = json.load(f)
    
    print(f"   Train 샘플: {vit_metrics.get('n_train', 'N/A')}개")
    print(f"   Valid 샘플: {vit_metrics.get('n_val', 'N/A')}개") 
    print(f"   Test 샘플: {vit_metrics.get('n_test', 'N/A')}개")
    
    # 2. XGBoost에서 사용된 데이터 확인
    print(f"\n2️⃣ XGBoost에서 사용된 데이터")
    
    # XGBoost가 사용한 임베딩 데이터
    df_embedding = pd.read_parquet('./runs_vit_light/test_embeddings.parquet')
    print(f"   XGBoost 학습 데이터: {len(df_embedding)}개")
    print(f"   샘플 ID들: {df_embedding['sample_id'].tolist()[:5]}...")
    
    # 그룹 분석
    groups = [sid.split('_')[0] for sid in df_embedding['sample_id']]
    unique_groups = set(groups)
    print(f"   고유 그룹 수: {len(unique_groups)}")
    print(f"   그룹들: {unique_groups}")
    
    # 3. 핵심 문제 진단
    print(f"\n3️⃣ 문제 진단")
    
    # 문제 1: XGBoost가 ViT의 test set만 사용했는가?
    if vit_metrics.get('n_test') == len(df_embedding):
        print(f"   ⚠️  XGBoost가 ViT의 TEST SET만 사용함!")
        print(f"       - ViT test: {vit_metrics.get('n_test')}개")
        print(f"       - XGB 데이터: {len(df_embedding)}개")
        print(f"   ❌ 이는 XGBoost에서 train/val/test 분할이 없음을 의미")
    
    # 문제 2: 그룹이 1개뿐인 문제
    if len(unique_groups) == 1:
        print(f"   ⚠️  모든 데이터가 같은 그룹 ({list(unique_groups)[0]})")
        print(f"   ❌ GroupKFold가 제대로 작동하지 않음")
    
    # 문제 3: 과적합 가능성
    print(f"\n4️⃣ 과적합 가능성 분석")
    
    # XGBoost 결과 확인
    with open('./xgb_results/xgb_summary.json', 'r') as f:
        xgb_summary = json.load(f)
    
    print(f"   데이터 크기: {len(df_embedding)}개 (매우 작음)")
    print(f"   피처 차원: 192개 (데이터 대비 매우 높음)")
    print(f"   피처/샘플 비율: {192/len(df_embedding):.1f}:1")
    
    if 192 / len(df_embedding) > 1:
        print(f"   ❌ 피처 수가 샘플 수보다 많음 → 과적합 위험 극대")
    
    # 5. 해결 방안 제시
    print(f"\n5️⃣ 해결 방안")
    print(f"   1. ViT 학습 시 모든 데이터 사용하여 임베딩 생성")
    print(f"   2. 전체 임베딩에서 train/val/test 새로 분할")
    print(f"   3. PCA 등으로 차원 축소")
    print(f"   4. 정규화 강화 (L1/L2 regularization)")
    
    return {
        'vit_test_size': vit_metrics.get('n_test', 0),
        'xgb_data_size': len(df_embedding),
        'n_groups': len(unique_groups),
        'feature_to_sample_ratio': 192 / len(df_embedding),
        'is_problematic': vit_metrics.get('n_test') == len(df_embedding)
    }

def check_actual_performance():
    """실제 성능을 더 엄격하게 평가"""
    print(f"\n=== 실제 성능 재평가 ===")
    
    df_pred = pd.read_csv('./xgb_predictions.csv')
    
    for target in ['Marbling', 'Meat Color', 'Texture', 'Surface Moisture', 'Total']:
        true_col = f'true_{target}'
        pred_col = f'pred_{target}'
        
        if true_col in df_pred.columns and pred_col in df_pred.columns:
            y_true = df_pred[true_col].values
            y_pred = df_pred[pred_col].values
            
            # 실제 값의 분포 확인
            print(f"\n🎯 {target}:")
            print(f"   실제값 범위: {y_true.min():.3f} ~ {y_true.max():.3f}")
            print(f"   실제값 표준편차: {y_true.std():.3f}")
            print(f"   예측값 범위: {y_pred.min():.3f} ~ {y_pred.max():.3f}")
            print(f"   예측값 표준편차: {y_pred.std():.3f}")
            
            # 완전히 동일한 값들이 있는지 확인
            exact_matches = np.sum(np.abs(y_true - y_pred) < 1e-6)
            print(f"   완전 일치 개수: {exact_matches}/{len(y_true)}")
            
            if exact_matches > len(y_true) * 0.5:
                print(f"   ⚠️  50% 이상이 완전 일치 → 의심스러움")

if __name__ == "__main__":
    diagnosis = analyze_data_leakage()
    check_actual_performance()
    
    print(f"\n=== 최종 진단 ===")
    if diagnosis['is_problematic']:
        print(f"❌ 심각한 데이터 누출 문제 발견!")
        print(f"   XGBoost가 ViT의 test set에서만 학습함")
        print(f"   실제로는 train/val/test 분할이 없었음")
        print(f"🔧 전체 데이터를 다시 분할해서 학습해야 함")
    else:
        print(f"✅ 데이터 분할은 올바름")
        
    if diagnosis['feature_to_sample_ratio'] > 5:
        print(f"⚠️  과적합 위험: 피처/샘플 비율 {diagnosis['feature_to_sample_ratio']:.1f}:1")