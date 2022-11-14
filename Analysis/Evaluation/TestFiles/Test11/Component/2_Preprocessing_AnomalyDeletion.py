#-*- coding: utf-8 -*-

import psycopg2
import pandas.io.sql as psql
import yaml

def read_yaml(dir):
    """
    yaml 파일 읽어서 딕셔너리 형태로 반환
    """
    with open(dir, 'r') as f:
        return yaml.safe_load(f)

if __name__ == "__main__":

    mean = read_yaml(currentpath+'/lib/outlier_3sigma_mean.yaml')
    std = read_yaml(currentpath+'/lib/outlier_3sigma_std.yaml')

    result = dataset
    for col, item in dataset.iteritems():
        print(' check_\'',col,'\'')
        try:
            max = float(mean[col]) + 3*float(std[col])
            min = float(mean[col]) - 3*float(std[col])
            idx_max = result[result[col] > max].index
            idx_min = result[result[col] < min].index
            result = result.drop(idx_max)
            result = result.drop(idx_min)
        except:
            pass

    dataset = result
    print(' 3sigma outlier deleted')

