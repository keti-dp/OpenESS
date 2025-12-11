# python soc_slope_synthesis.py --data_dir /data/ess/data/TestBed/241213_scenario/preprocessed_normal/1/20241113.parquet --intensity 0 --output_dir /data/ess/data/TestBed/241213_scenario/preprocessed_abnormal_soc_slope

# python soc_slope_synthesis.py --data_dir /data/ess/data/TestBed/241213_scenario/preprocessed_normal/1/20241113.parquet --intensity 0.1 --output_dir /data/ess/data/TestBed/241213_scenario/preprocessed_abnormal_soc_slope

# python soc_slope_synthesis.py --data_dir /data/ess/data/TestBed/241213_scenario/preprocessed_normal/1/20241113.parquet --intensity 0.2 --output_dir /data/ess/data/TestBed/241213_scenario/preprocessed_abnormal_soc_slope

# python soc_slope_synthesis.py --data_dir /data/ess/data/TestBed/241213_scenario/preprocessed_normal/1/20241113.parquet --intensity 0.3 --output_dir /data/ess/data/TestBed/241213_scenario/preprocessed_abnormal_soc_slope

# python soc_slope_synthesis.py --data_dir /data/ess/data/TestBed/241213_scenario/preprocessed_normal/1/20241113.parquet --intensity 0.4 --output_dir /data/ess/data/TestBed/241213_scenario/preprocessed_abnormal_soc_slope

# python soc_slope_synthesis.py --data_dir /data/ess/data/TestBed/241213_scenario/preprocessed_normal/1/20241113.parquet --intensity 0.5 --output_dir /data/ess/data/TestBed/241213_scenario/preprocessed_abnormal_soc_slope

# python soc_slope_synthesis.py --data_dir /data/ess/data/TestBed/241213_scenario/preprocessed_normal/1/20241113.parquet --intensity 0.6 --output_dir /data/ess/data/TestBed/241213_scenario/preprocessed_abnormal_soc_slope

# python soc_slope_synthesis.py --data_dir /data/ess/data/TestBed/241213_scenario/preprocessed_normal/1/20241113.parquet --intensity 0.7 --output_dir /data/ess/data/TestBed/241213_scenario/preprocessed_abnormal_soc_slope

# python soc_slope_synthesis.py --data_dir /data/ess/data/TestBed/241213_scenario/preprocessed_normal/1/20241113.parquet --intensity 0.8 --output_dir /data/ess/data/TestBed/241213_scenario/preprocessed_abnormal_soc_slope

# python soc_slope_synthesis.py --data_dir /data/ess/data/TestBed/241213_scenario/preprocessed_normal/1/20241113.parquet --intensity 0.9 --output_dir /data/ess/data/TestBed/241213_scenario/preprocessed_abnormal_soc_slope

# python soc_slope_synthesis.py --data_dir /data/ess/data/TestBed/241213_scenario/preprocessed_normal/1/20241113.parquet --intensity 1.0 --output_dir /data/ess/data/TestBed/241213_scenario/preprocessed_abnormal_soc_slope

# # --------------------------------------------------------------------------------------------------------

# python soc_slope_synthesis.py --data_dir /home/sky3alfory/NCIA_ESS/Forecasting/Clean_panli/20230321.parquet --intensity 0 --output_dir /home/sky3alfory/NCIA_ESS/Forecasting/SoC_synthesis

# python soc_slope_synthesis.py --data_dir /home/sky3alfory/NCIA_ESS/Forecasting/Clean_panli/20230321.parquet --intensity 0.1 --output_dir /home/sky3alfory/NCIA_ESS/Forecasting/SoC_synthesis

# python soc_slope_synthesis.py --data_dir /home/sky3alfory/NCIA_ESS/Forecasting/Clean_panli/20230321.parquet --intensity 0.2 --output_dir /home/sky3alfory/NCIA_ESS/Forecasting/SoC_synthesis

# python soc_slope_synthesis.py --data_dir /home/sky3alfory/NCIA_ESS/Forecasting/Clean_panli/20230321.parquet --intensity 0.3 --output_dir /home/sky3alfory/NCIA_ESS/Forecasting/SoC_synthesis

# python soc_slope_synthesis.py --data_dir /home/sky3alfory/NCIA_ESS/Forecasting/Clean_panli/20230321.parquet --intensity 0.4 --output_dir /home/sky3alfory/NCIA_ESS/Forecasting/SoC_synthesis

# python soc_slope_synthesis.py --data_dir /home/sky3alfory/NCIA_ESS/Forecasting/Clean_panli/20230321.parquet --intensity 0.5 --output_dir /home/sky3alfory/NCIA_ESS/Forecasting/SoC_synthesis

# python soc_slope_synthesis.py --data_dir /home/sky3alfory/NCIA_ESS/Forecasting/Clean_panli/20230321.parquet --intensity 0.6 --output_dir /home/sky3alfory/NCIA_ESS/Forecasting/SoC_synthesis

# python soc_slope_synthesis.py --data_dir /home/sky3alfory/NCIA_ESS/Forecasting/Clean_panli/20230321.parquet --intensity 0.7 --output_dir /home/sky3alfory/NCIA_ESS/Forecasting/SoC_synthesis

# python soc_slope_synthesis.py --data_dir /home/sky3alfory/NCIA_ESS/Forecasting/Clean_panli/20230321.parquet --intensity 0.8 --output_dir /home/sky3alfory/NCIA_ESS/Forecasting/SoC_synthesis

# python soc_slope_synthesis.py --data_dir /home/sky3alfory/NCIA_ESS/Forecasting/Clean_panli/20230321.parquet --intensity 0.9 --output_dir /home/sky3alfory/NCIA_ESS/Forecasting/SoC_synthesis

# python soc_slope_synthesis.py --data_dir /home/sky3alfory/NCIA_ESS/Forecasting/Clean_panli/20230321.parquet --intensity 1 --output_dir /home/sky3alfory/NCIA_ESS/Forecasting/SoC_synthesis

# # --------------------------------------------------------------------------------------------------------

# set -euo pipefail

IN_DIR="/home/sky3alfory/NCIA_ESS/Forecasting/Panli_type2"
OUT_ROOT="/home/sky3alfory/NCIA_ESS/Forecasting/SoC_synthesis"

# 모든 parquet 파일 순회
for f in "$IN_DIR"/*.parquet; do
  [ -e "$f" ] || continue
  base="$(basename "$f" .parquet)"
  outdir="$OUT_ROOT/$base"           # 파일명과 동일한 폴더
  mkdir -p "$outdir"

  # intensity 0.0 ~ 1.0 (0.1 step)
  for n in $(LC_ALL=C seq 0 0.1 1); do
    n=$(printf "%.1f" "$n")
    echo "Processing: $base  intensity=$n"
    python soc_slope_synthesis.py \
      --data_dir "$f" \
      --intensity "$n" \
      --output_dir "$outdir"
  done
done
