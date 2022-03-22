import sys

sys.path.append("./")
import google_storage_class


_PROJECT_NAME = "KETI-IISRC"
_BUCKET_NAME = "ess-bucket-1"
_CREDENTIAL_JSON_FILE_PATH = "./Data_Ingestion/gcp_apikey/keti-iisrc-40dc8930fc3b.json"
# _CREDENTIAL_JSON_FILE_PATH = (
#     "/home/keti_iisrc/test/gcp_apikey/keti-iisrc-40dc8930fc3b.json"
# )


def upload_2bucket(gcs_file_path, local_file_path):

    # GCS 객체 생성
    GCP = google_storage_class.google_cloud_storage(
        _PROJECT_NAME, _BUCKET_NAME, _CREDENTIAL_JSON_FILE_PATH
    )
    GCP.upload_to_bucket(gcs_file_path, local_file_path)


if __name__ == "__main__":
    upload_2bucket()
