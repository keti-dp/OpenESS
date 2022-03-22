#-*- coding: utf-8 -*-
import pandas as pd
from google.cloud import storage
import os
from datetime import datetime

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]= currentpath + "/lib/keti-iisrc-40dc8930fc3b.json"

if __name__ == "__main__":

    etc_file_dir = currentpath+'/etc/{}.csv'.format(datetime.now().strftime('%Y%m%d-%H:%M'))
    dataset = dataset.to_csv(etc_file_dir, sep=',')

    storage_client = storage.Client()

    bucket_name = 'ess_temporary_client_bucket'
    source_blob_name = etc_file_dir
    destination_blob_name = etc_file_dir
    print(' bucket:{}'.format(bucket_name))

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    print(' blob(dir):{}'.format(destination_blob_name))

    blob.upload_from_filename(etc_file_dir)
    print(' dataset upload done')