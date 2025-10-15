"""
CSV 파일에 sample_id 컬럼을 추가하여 학습 스크립트와 완전히 호환되도록 만드는 스크립트
"""

import pandas as pd

def add_sample_id():
    print("=== sample_id 컬럼 추가 ===")
    
    # CSV 파일 읽기
    csv_file = "complete_merged_data_with_file_paths.csv"
    df = pd.read_csv(csv_file)
    
    print(f"원본 파일: {len(df)}행 × {len(df.columns)}컬럼")
    
    # sample_id 컬럼 생성 (이력번호_샘플번호 조합)
    df['sample_id'] = df['이력번호'].astype(str) + '_' + df['샘플번호'].astype(str)
    
    print(f"수정 후: {len(df)}행 × {len(df.columns)}컬럼")
    print(f"추가된 컬럼: sample_id")
    
    # 몇 개 예시 출력
    print(f"\n=== sample_id 예시 ===")
    for i in range(min(5, len(df))):
        print(f"  {i+1}. {df.iloc[i]['이력번호']}_{df.iloc[i]['샘플번호']} → {df.iloc[i]['sample_id']}")
    
    # 컬럼 순서 조정 (sample_id를 앞쪽으로)
    cols = df.columns.tolist()
    cols.remove('sample_id')
    cols.insert(2, 'sample_id')  # 이력번호, 샘플번호 다음에 배치
    df = df[cols]
    
    # 저장
    output_file = "complete_merged_data_ready_for_training.csv"
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n=== 저장 완료 ===")
    print(f"파일명: {output_file}")
    print(f"총 컬럼: {len(df.columns)}")
    
    # 최종 호환성 확인
    print(f"\n=== 최종 호환성 확인 ===")
    required_cols = ["sample_id", "Marbling", "Meat Color", "Texture", "Surface Moisture", "Total", 
                    "file_path_430nm", "file_path_540nm", "file_path_580nm", "이력번호"]
    
    all_good = True
    for col in required_cols:
        exists = col in df.columns
        print(f"  {col}: {'✓' if exists else '❌'}")
        if not exists:
            all_good = False
    
    if all_good:
        print(f"\n🎉 완벽한 호환성! 학습 준비 완료!")
    else:
        print(f"\n⚠️  여전히 문제가 있습니다.")
    
    return output_file, all_good

if __name__ == "__main__":
    output_file, is_ready = add_sample_id()
    
    if is_ready:
        print(f"\n🚀 {output_file}을 사용하여 학습을 시작할 수 있습니다!")
    else:
        print(f"\n🔧 추가 수정이 필요합니다.")