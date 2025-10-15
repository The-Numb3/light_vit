"""
train_vit_light.py와 config_vit_light.yaml의 호환성을 확인하는 스크립트
"""

import pandas as pd
import os

def check_compatibility():
    print("=== train_vit_light.py와 config 호환성 검토 ===\n")
    
    # 1. CSV 파일 확인
    csv_file = "complete_merged_data_with_file_paths.csv"
    if not os.path.exists(csv_file):
        print(f"❌ CSV 파일이 없습니다: {csv_file}")
        return False
    
    df = pd.read_csv(csv_file)
    print(f"📋 CSV 파일: {csv_file}")
    print(f"   행 수: {len(df)}")
    print(f"   컬럼 수: {len(df.columns)}")
    
    # 2. config에서 요구하는 컬럼들 확인
    print(f"\n=== config_vit_light.yaml 요구사항 확인 ===")
    
    # label_cols 확인
    label_cols = ["Marbling", "Meat Color", "Texture", "Surface Moisture", "Total"]
    print(f"📊 Label 컬럼들:")
    missing_labels = []
    for col in label_cols:
        exists = col in df.columns
        print(f"   {col}: {'✓' if exists else '❌'}")
        if not exists:
            missing_labels.append(col)
    
    # path_cols 확인 (config에서는 잘못된 이름 사용)
    print(f"\n📁 Path 컬럼들:")
    config_path_cols = ["file_path_430", "file_path_540", "file_path_580"]  # config의 잘못된 이름
    actual_path_cols = ["file_path_430nm", "file_path_540nm", "file_path_580nm"]  # CSV의 실제 이름
    
    for i, (config_col, actual_col) in enumerate(zip(config_path_cols, actual_path_cols)):
        config_exists = config_col in df.columns
        actual_exists = actual_col in df.columns
        print(f"   config: {config_col} {'✓' if config_exists else '❌'}")
        print(f"   actual: {actual_col} {'✓' if actual_exists else '❌'}")
        print()
    
    # sample_id 또는 샘플번호 확인
    print(f"📝 그룹/샘플 컬럼들:")
    group_cols = ["sample_id", "샘플번호", "이력번호"]
    for col in group_cols:
        exists = col in df.columns
        print(f"   {col}: {'✓' if exists else '❌'}")
    
    # 3. 실제 컬럼 목록
    print(f"\n=== 실제 CSV 컬럼들 (처음 20개) ===")
    for i, col in enumerate(df.columns[:20]):
        print(f"   {i+1:2d}. {col}")
    
    if len(df.columns) > 20:
        print(f"   ... 및 {len(df.columns) - 20}개 더")
    
    # 4. 호환성 문제점 분석
    print(f"\n=== 호환성 문제점 및 해결 방안 ===")
    
    issues = []
    solutions = []
    
    # Label 컬럼 문제
    if missing_labels:
        issues.append(f"❌ 누락된 label 컬럼: {missing_labels}")
        solutions.append("→ config에서 실제 존재하는 컬럼명으로 수정")
    else:
        print("✅ 모든 label 컬럼이 존재함")
    
    # Path 컬럼 문제
    path_issue = not all(col in df.columns for col in config_path_cols)
    if path_issue:
        issues.append(f"❌ path 컬럼명 불일치: config={config_path_cols}, actual={actual_path_cols}")
        solutions.append("→ config의 path_cols를 올바른 이름으로 수정")
    else:
        print("✅ 모든 path 컬럼이 일치함")
    
    # sample_id 문제
    has_sample_id = "sample_id" in df.columns
    has_sample_num = "샘플번호" in df.columns
    if not has_sample_id and has_sample_num:
        issues.append(f"❌ sample_id 컬럼이 없음 (샘플번호는 존재)")
        solutions.append("→ CSV에 sample_id 컬럼 추가 또는 코드에서 샘플번호 사용")
    elif has_sample_id:
        print("✅ sample_id 컬럼 존재")
    
    # 5. 결론
    print(f"\n=== 호환성 검토 결과 ===")
    if issues:
        print("⚠️  호환성 문제 발견:")
        for issue in issues:
            print(f"   {issue}")
        print("\n🔧 해결 방안:")
        for solution in solutions:
            print(f"   {solution}")
        return False
    else:
        print("🎉 완벽한 호환성!")
        print("✅ CSV 파일이 학습 스크립트와 완전히 호환됩니다.")
        return True

if __name__ == "__main__":
    is_compatible = check_compatibility()
    
    if is_compatible:
        print("\n🚀 학습 실행 준비 완료!")
    else:
        print("\n🔧 수정이 필요합니다.")