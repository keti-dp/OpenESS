#-*- coding: utf-8 -*-
import pandas as pd
from google.cloud import storage
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]= currentpath + "/lib/keti-iisrc-40dc8930fc3b.json"

if __name__ == "__main__":
    storage_client = storage.Client()
    buckets = list(storage_client.list_buckets())
    print(' bucket_list')
    print(' ',buckets)

    bucket_name = 'ess-bucket-1'
    source_blob_name = '2021/11/01/20211101_bank.csv'
    destination_file_name = currentpath+'/etc/2021_11_01_20211101_bank.csv'

    print(' dataset download')
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    dataset = pd.read_csv(destination_file_name)
    #pd.set_option('display.max_rows', None)
    #pd.set_option('display.max_columns', None)
    #pd.set_option('display.width', None)
    print(' dataset load')
    print(dataset.head(10),'\n')
    print(' ', len(dataset),'rows & ', len(dataset.columns),'cols loaded')