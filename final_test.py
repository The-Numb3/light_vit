"""
실제 학습 실행 전 최종 테스트
"""

import pandas as pd
import os
import torch
import yaml

def final_test():
    print("=== 학습 실행 준비 최종 점검 ===\n")
    
    # 1. 파일 존재 확인
    csv_file = "complete_merged_data_ready_for_training.csv"
    config_file = "config_vit_light_fixed.yaml"
    train_script = "train_vit_light.py"
    
    files = [csv_file, config_file, train_script]
    print("📁 필수 파일 확인:")
    for file in files:
        exists = os.path.exists(file)
        print(f"   {file}: {'✓' if exists else '❌'}")
        if not exists:
            return False
    
    # 2. CSV 데이터 확인
    print(f"\n📊 데이터 확인:")
    df = pd.read_csv(csv_file)
    print(f"   총 레코드: {len(df)}개")
    print(f"   총 컬럼: {len(df.columns)}개")
    
    # 3. 파일 경로 실제 존재 확인
    print(f"\n📷 이미지 파일 존재 확인:")
    path_cols = ["file_path_430nm", "file_path_540nm", "file_path_580nm"]
    for col in path_cols:
        exists_count = sum(1 for path in df[col] if os.path.exists(path))
        total_count = len(df)
        percentage = (exists_count / total_count) * 100
        print(f"   {col}: {exists_count}/{total_count} ({percentage:.1f}%)")
    
    # 4. Config 파일 확인
    print(f"\n⚙️  Config 파일 확인:")
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"   CSV 경로: {config['data']['csv_path']}")
    print(f"   Label 컬럼: {config['data']['label_cols']}")
    print(f"   Path 컬럼: {config['data']['path_cols']}")
    print(f"   배치 크기: {config['train']['batch_size']}")
    print(f"   에포크: {config['train']['epochs']}")
    
    # 5. 하드웨어 확인
    print(f"\n🖥️  하드웨어 확인:")
    print(f"   CUDA 사용 가능: {'✓' if torch.cuda.is_available() else '❌'}")
    if torch.cuda.is_available():
        print(f"   GPU 개수: {torch.cuda.device_count()}")
        print(f"   GPU 이름: {torch.cuda.get_device_name(0)}")
    
    # 6. 예상 실행 명령
    print(f"\n🚀 실행 명령:")
    print(f"   conda activate vitxgb")
    print(f"   cd c:\\kkm\\deeplant_2025\\dataset\\result_3dim")
    print(f"   python train_vit_light.py --config {config_file}")
    
    print(f"\n✅ 모든 준비가 완료되었습니다!")
    return True

if __name__ == "__main__":
    ready = final_test()
    
    if ready:
        print(f"\n🎯 학습을 시작할 수 있습니다!")
        print(f"💡 예상 학습 시간: 약 10-20분 (GPU 기준)")
        print(f"💡 결과는 './runs_vit_light' 폴더에 저장됩니다.")
    else:
        print(f"\n⚠️  추가 수정이 필요합니다.")