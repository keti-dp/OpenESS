"""
독립 실행 GCP 데이터 다운로더
GCP Storage에서 데이터를 다운로드하여 Parquet 파일로 저장
Airflow와 독립적으로 실행 가능
"""
import os
from datetime import datetime, timedelta
from pathlib import Path
from google.cloud import storage
import pandas as pd
import argparse


class GCPDataDownloader:
    """GCP 버킷에서 데이터를 다운로드하고 Parquet 형식으로 변환"""

    def __init__(self, credential_path, bucket_name, save_dir=None, data_types=None,
                 dest_bucket_name=None, dest_blob_prefix=None):
        """
        GCP 데이터 다운로더 초기화

        Args:
            credential_path: Path to GCP 인증 JSON 파일
            bucket_name: 소스 GCP 버킷 이름 (데이터를 가져올 버킷)
            save_dir: 로컬 임시 저장 디렉토리 (옵션, dest_bucket_name이 있으면 임시 사용)
            data_types: 다운로드할 데이터 타입 리스트 (기본값: ['rack'])
                       예: ['rack'], ['bank', 'rack'], ['rack', 'pcs', 'etc', 'bank']
            dest_bucket_name: 목적지 GCP 버킷 이름 (변환된 파일을 업로드할 버킷)
            dest_blob_prefix: 목적지 버킷 내 경로 (예: 'rack-ori-data/site-a/')
        """
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credential_path
        self.bucket_name = bucket_name
        self.data_types = data_types or ['rack']  # 기본적으로 rack만 다운로드

        # 로컬 저장 디렉토리 (임시 사용 또는 최종 저장)
        if save_dir:
            self.save_dir = Path(save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)
        else:
            # 임시 디렉토리 사용
            import tempfile
            self.save_dir = Path(tempfile.mkdtemp())

        # GCS 업로드 설정
        self.dest_bucket_name = dest_bucket_name
        self.dest_blob_prefix = dest_blob_prefix or ''
        if self.dest_blob_prefix and not self.dest_blob_prefix.endswith('/'):
            self.dest_blob_prefix += '/'

        # GCP 스토리지 클라이언트 초기화
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(bucket_name)

        # 목적지 버킷 (GCS 업로드용)
        if self.dest_bucket_name:
            self.dest_bucket = self.storage_client.bucket(dest_bucket_name)
        else:
            self.dest_bucket = None

    def download_date_data(self, year, month, day):
        """
        특정 날짜의 데이터 다운로드 및 GCS 업로드

        Args:
            year: 년도 (e.g., 2025)
            month: 월 (1-12)
            day: 일 (1-31)
        """
        # Format: YYYY/MM/DD/
        blob_prefix = f"{year}/{month:02d}/{day:02d}/"
        date_str = f"{year}{month:02d}{day:02d}"

        print(f"데이터 다운로드 중: {blob_prefix}...")

        # 접두사로 모든 blob 나열
        blobs = self.bucket.list_blobs(prefix=blob_prefix)

        downloaded_files = []
        for blob in blobs:
            file_name = blob.name.split("/")[-1]

            # CSV 파일과 빈 파일명 건너뛰기
            if not file_name or file_name.endswith('.csv'):
                continue

            # 데이터 타입 필터링 (rack, bank, pcs, etc)
            file_name_lower = file_name.lower()
            if not any(data_type.lower() in file_name_lower for data_type in self.data_types):
                continue

            # 1. parquet 파일 우선 처리
            if file_name.endswith('.parquet'):
                # 목적지 GCS에 이미 있는지 체크
                if self.dest_bucket:
                    dest_blob_path = f"{self.dest_blob_prefix}{file_name}"
                    dest_blob = self.dest_bucket.blob(dest_blob_path)
                    if dest_blob.exists():
                        print(f"  ✓ 이미 목적지 GCS에 존재 (스킵): {dest_blob_path}")
                        downloaded_files.append(file_name)
                        continue

                # parquet 다운로드
                parquet_path = self.save_dir / file_name
                print(f"  다운로드 중: {file_name}")
                blob.download_to_filename(str(parquet_path))

                try:
                    # 목적지 GCS에 업로드
                    if self.dest_bucket:
                        dest_blob_path = f"{self.dest_blob_prefix}{file_name}"
                        dest_blob = self.dest_bucket.blob(dest_blob_path)
                        dest_blob.upload_from_filename(str(parquet_path))
                        print(f"  ✓ GCS 업로드 완료: gs://{self.dest_bucket_name}/{dest_blob_path}")

                        # 로컬 파일 삭제 (GCS 업로드 후)
                        parquet_path.unlink()
                    else:
                        print(f"  ✓ 로컬 저장 완료: {parquet_path}")

                    downloaded_files.append(file_name)
                except Exception as e:
                    print(f"  ✗ 업로드 오류: {file_name}: {e}")
                    if parquet_path.exists():
                        parquet_path.unlink()

            # 2. feather/ft 파일 처리 (parquet가 없을 때)
            elif file_name.endswith(('.feather', '.ft')):
                parquet_filename = file_name.replace('.feather', '.parquet').replace('.ft', '.parquet')

                # 목적지 GCS에 parquet 파일이 이미 있는지 체크
                if self.dest_bucket:
                    dest_blob_path = f"{self.dest_blob_prefix}{parquet_filename}"
                    dest_blob = self.dest_bucket.blob(dest_blob_path)
                    if dest_blob.exists():
                        print(f"  ✓ 이미 목적지 GCS에 존재 (스킵): {dest_blob_path}")
                        downloaded_files.append(parquet_filename)
                        continue

                # feather 다운로드 → 변환 → 업로드
                temp_file_path = self.save_dir / file_name
                parquet_path = self.save_dir / parquet_filename

                print(f"  다운로드 중: {file_name}")
                blob.download_to_filename(str(temp_file_path))

                try:
                    # Read feather file and save as parquet
                    df = pd.read_feather(temp_file_path)
                    df.to_parquet(parquet_path, index=False, engine='pyarrow')
                    print(f"  Parquet으로 변환 완료: {parquet_filename}")

                    # 목적지 GCS에 업로드
                    if self.dest_bucket:
                        dest_blob_path = f"{self.dest_blob_prefix}{parquet_filename}"
                        dest_blob = self.dest_bucket.blob(dest_blob_path)
                        dest_blob.upload_from_filename(str(parquet_path))
                        print(f"  ✓ GCS 업로드 완료: gs://{self.dest_bucket_name}/{dest_blob_path}")

                        # 로컬 파일 삭제 (GCS 업로드 후)
                        parquet_path.unlink()
                    else:
                        print(f"  ✓ 로컬 저장 완료: {parquet_path}")

                    # 임시 feather 파일 삭제
                    temp_file_path.unlink()

                    downloaded_files.append(parquet_filename)
                except Exception as e:
                    print(f"  ✗ 변환/업로드 오류: {file_name}: {e}")
                    # 오류 발생 시 임시 파일 정리
                    if temp_file_path.exists():
                        temp_file_path.unlink()
                    if parquet_path.exists():
                        parquet_path.unlink()

        return downloaded_files

    def download_month_data(self, year, month):
        """
        전체 월 데이터 다운로드

        Args:
            year: 년도 (e.g., 2025)
            month: 월 (1-12)
        """
        from calendar import monthrange

        _, num_days = monthrange(year, month)

        print(f"\n{'='*60}")
        print(f"데이터 다운로드 중: {year}-{month:02d}")
        print(f"데이터 타입: {', '.join(self.data_types)}")
        print(f"{'='*60}\n")

        all_files = []
        for day in range(1, num_days + 1):
            try:
                files = self.download_date_data(year, month, day)
                all_files.extend(files)
            except Exception as e:
                print(f"데이터 다운로드 오류: {year}-{month:02d}-{day:02d}: {e}")

        print(f"\n총 다운로드 파일 수: {year}-{month:02d}: {len(all_files)}")
        return all_files

    def download_date_range(self, start_year, start_month, end_year, end_month):
        """
        날짜 범위의 데이터 다운로드

        Args:
            start_year: 시작 년도
            start_month: 시작 월 (1-12)
            end_year: 종료 년도
            end_month: 종료 월 (1-12)
        """
        current_year = start_year
        current_month = start_month

        while (current_year < end_year) or (current_year == end_year and current_month <= end_month):
            self.download_month_data(current_year, current_month)

            # 다음 월로 이동
            current_month += 1
            if current_month > 12:
                current_month = 1
                current_year += 1


def main():
    """독립 실행용 메인 함수"""
    parser = argparse.ArgumentParser(description='Download GCP data and convert to Parquet')
    parser.add_argument('--credential-path', type=str,
                       default=os.getenv('GCP_CREDENTIAL_PATH'),
                       help='Path to GCP 인증 JSON 파일 (환경변수 GCP_CREDENTIAL_PATH로 설정 가능)')
    parser.add_argument('--bucket-name', type=str,
                       default=os.getenv('SOURCE_BUCKET_NAME', 'source-data-bucket'),
                       help='소스 GCP 버킷 이름 (환경변수 SOURCE_BUCKET_NAME으로 설정 가능)')
    parser.add_argument('--save-dir', type=str,
                       default=os.getenv('LOCAL_DATA_DIR', '/tmp/data'),
                       help='로컬 임시 저장 디렉토리 (환경변수 LOCAL_DATA_DIR로 설정 가능)')
    parser.add_argument('--data-types', type=str, nargs='+', default=['rack'],
                       help='다운로드할 데이터 타입 (기본값: rack). 예: --data-types rack bank pcs')
    parser.add_argument('--dest-bucket-name', type=str,
                       default=os.getenv('DEST_BUCKET_NAME'),
                       help='목적지 GCS 버킷 이름 (지정하면 GCS에 업로드, 미지정시 로컬 저장)')
    parser.add_argument('--dest-blob-prefix', type=str,
                       default=os.getenv('DEST_BLOB_PREFIX', ''),
                       help='목적지 버킷 내 경로 (환경변수 DEST_BLOB_PREFIX로 설정 가능)')
    parser.add_argument('--start-year', type=int, default=2025,
                       help='시작 년도')
    parser.add_argument('--start-month', type=int, default=1,
                       help='시작 월 (1-12)')
    parser.add_argument('--end-year', type=int, default=2025,
                       help='종료 년도')
    parser.add_argument('--end-month', type=int, default=11,
                       help='종료 월 (1-12)')

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"GCP Data Downloader")
    print(f"{'='*60}")
    print(f"Credential: {args.credential_path}")
    print(f"Source Bucket: {args.bucket_name}")
    print(f"Save directory: {args.save_dir}")
    print(f"Data types: {', '.join(args.data_types)}")
    if args.dest_bucket_name:
        print(f"Destination: gs://{args.dest_bucket_name}/{args.dest_blob_prefix}")
    else:
        print(f"Destination: Local storage only")
    print(f"Date range: {args.start_year}/{args.start_month:02d} - {args.end_year}/{args.end_month:02d}")
    print(f"{'='*60}\n")

    # Create downloader instance
    downloader = GCPDataDownloader(
        credential_path=args.credential_path,
        bucket_name=args.bucket_name,
        save_dir=args.save_dir,
        data_types=args.data_types,
        dest_bucket_name=args.dest_bucket_name,
        dest_blob_prefix=args.dest_blob_prefix
    )

    # Download data
    downloader.download_date_range(
        start_year=args.start_year,
        start_month=args.start_month,
        end_year=args.end_year,
        end_month=args.end_month
    )

    print("\n" + "="*60)
    print("다운로드 완료!")
    print("="*60)


if __name__ == "__main__":
    main()
