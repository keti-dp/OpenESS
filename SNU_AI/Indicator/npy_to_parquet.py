import numpy as np
import pandas as pd
import argparse
import os

def npy_to_parquet(arr):
    """
    .npy 파일을 읽어 DataFrame으로 변환한 뒤,
    세 번째 컬럼 이름을 'BANK_SOC'로 설정하여 .parquet으로 저장합니다.
    """
    # 1) NumPy 로드
    # arr = np.load(input_npy)
    if arr.ndim != 2:
        raise ValueError(f"입력 배열 차원(ndim)이 2가 아닙니다: {arr.ndim}")

    # 2) DataFrame 생성
    df = pd.DataFrame(arr)

    # 3) 컬럼명 설정 (0,1번 컬럼은 그대로, 2번 컬럼만 BANK_SOC)
    cols = list(df.columns)
    if len(cols) < 3:
        raise ValueError("입력 배열에 최소 3개 이상의 컬럼이 필요합니다.")
    cols[2] = "BANK_SOC"
    df.columns = cols

    # # 4) 출력 폴더가 없으면 생성
    # out_dir = os.path.dirname(output_parquet)
    # if out_dir and not os.path.exists(out_dir):
    #     os.makedirs(out_dir, exist_ok=True)

    # # 5) Parquet 저장
    # df.to_parquet(output_parquet, index=False)
    # print(f"Saved {input_npy} → {output_parquet}")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert .npy to .parquet and rename 3rd column to BANK_SOC"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="입력 .npy 파일 경로"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="출력 .parquet 파일 경로"
    )
    args = parser.parse_args()
    npy_to_parquet(args.input, args.output)