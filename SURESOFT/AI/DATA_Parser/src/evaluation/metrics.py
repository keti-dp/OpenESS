import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def get_score(
    df_gt: pd.DataFrame,
    df_score: pd.DataFrame,
    thres: float = 0.5,
    timestamp_col_gt: str = "timestamp",
    timestamp_col_pred: str = "TIMESTAMP",
    tolerance: str = "1s",
    timezone: str = "Asia/Seoul",
    print_result: bool = True,
):
    """
    Ground truth df와 모델 score df를 받아서
    - threshold 기준으로 Pred 라벨 생성
    - timestamp 기준 merge_asof
    - Accuracy / Precision / Recall / F1 계산

    Parameters
    ----------
    df_gt : pd.DataFrame
        정답 라벨이 포함된 데이터프레임 (timestamp_col_gt, 'Answer' 포함)
    df_score : pd.DataFrame
        모델 점수가 포함된 데이터프레임 (timestamp_col_pred, 'VALUE' 포함)
    thres : float
        anomaly score를 1/0으로 나눌 threshold
    timestamp_col_gt : str
        정답 쪽 timestamp 컬럼명 (기본: 'timestamp')
    timestamp_col_pred : str
        score 쪽 timestamp 컬럼명 (기본: 'TIMESTAMP')
    tolerance : str
        merge_asof에서 허용할 최대 시간 차이 (ex: '1s', '30s', '1min')
    timezone : str
        타임존 (기본: 'Asia/Seoul')
    print_result : bool
        True면 결과를 print, False면 반환만 함

    Returns
    -------
    df_merged : pd.DataFrame
        Answer와 Pred가 매칭된 데이터프레임
    metrics : dict
        accuracy, precision, recall, f1 값 딕셔너리
    """

    # 원본 안 건드리게 복사
    df_gt = df_gt.copy()
    df_score = df_score.copy()

    # 1) threshold 기준 1/0 라벨링
    df_score["Pred"] = (df_score["VALUE"] >= thres).astype(int)

    # 2) timestamp 타임존/타입 정리 (둘 다 같은 timezone & 타입으로 맞추기)
    df_gt[timestamp_col_gt] = (
        pd.to_datetime(df_gt[timestamp_col_gt], utc=True)
        .dt.tz_convert(timezone).astype('datetime64[ns, UTC]')
    )

    df_score[timestamp_col_pred] = (
        pd.to_datetime(df_score[timestamp_col_pred], utc=True)
        .dt.tz_convert(timezone).astype('datetime64[ns, UTC]')
    )

    # 3) merge_asof 로 시간 매칭
    df_merged = pd.merge_asof(
        df_score.sort_values(timestamp_col_pred),
        df_gt[[timestamp_col_gt, "Answer"]].sort_values(timestamp_col_gt),
        left_on=timestamp_col_pred,
        right_on=timestamp_col_gt,
        direction="nearest",
        tolerance=pd.Timedelta(tolerance),
    )

    # timestamp_col_gt는 더 이상 필요 없으면 제거
    df_merged = df_merged.drop(columns=[timestamp_col_gt])

    # 매칭 실패한 행 제거
    df_merged = df_merged.dropna(subset=["Answer"])

    # 4) 성능 평가
    y_true = df_merged["Answer"]
    y_pred = df_merged["Pred"]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }

    if print_result:
        print("📊 Model Evaluation Results")
        print(f"Threshold : {thres:.4f}")
        print(f"Accuracy  : {acc:.4f}")
        print(f"Precision : {prec:.4f}")
        print(f"Recall    : {rec:.4f}")
        print(f"F1 Score  : {f1:.4f}")

    return df_merged, metrics
