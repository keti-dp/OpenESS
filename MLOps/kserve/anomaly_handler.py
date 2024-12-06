import torch
import sys
import json
import psycopg2
import time
import pandas as pd
from datetime import datetime, timedelta
from ts.torch_handler.base_handler import BaseHandler
import anomaly_transformer

def dictfetchall(cursor):
    '''
    커서의 모든행을 dict으로 반환하는 함수
        Args:
            cursor (str): a value

        Retruns:

    '''
    columns = [col[0] for col in cursor.description]

    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def scaling (site_id, data):
    num_modules = {
        "1": 17, 
        "2": 20,
        "3": 17,
        "4": 20
    }

    ylims = {
        "BANK_DC_VOLT": [3.2, 4.2],
        "BANK_DC_CURRENT": [-3.5, 3.5],
        "BANK_SOC": [0, 100],
        "MAX_CELL_TEMPERATURE_OF_BANK": [5, 45],
        "VOLTAGE_GAP": [0, 0.4],
    }

    copy_df = data.copy()
    
    for col in copy_df.columns:
        if col == "BANK_DC_VOLT" or col == "BANK_DC_CURRENT":
            copy_df[col] = copy_df[col] / (num_modules[str(site_id)] * 12)
        
        if col in ylims:
            min_val, max_val = ylims[col]
            copy_df[col] = (copy_df[col] - min_val) / (max_val - min_val)

    return copy_df


class AnomalyHandler(BaseHandler):
    def initialize(self, context):
        self.batch_size = 1
        # self.window_sliding = 512
        # self.check_count = None
        # self.dataset = 'ESS_sionyu'
        self.device = 'cpu'
        self.post_activation = torch.nn.Sigmoid().to(self.device)

        model_dir = context.system_properties.get("model_dir", "./")
    
        try:
            #self.model = torch.load(f"{model_dir}/model.pt", map_location=self.device)
            self.model = anomaly_transformer.get_anomaly_transformer(input_d_data=5,
                                                    output_d_data=1,
                                                    patch_size=90,
                                                    d_embed=512,
                                                    hidden_dim_rate=4.,
                                                    max_seq_len=512,
                                                    positional_encoding=None,
                                                    relative_position_embedding=True,
                                                    transformer_n_layer=6,
                                                    transformer_n_head=8,
                                                    dropout=0.1
                                                    )
            self.model.load_state_dict(torch.load(f"{model_dir}/state_dict_step_150000.pt", map_location=self.device))
            self.model.eval()
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
 

    def preprocess(self, data):
        try:
            input_dict = data[0]
            #body = input_dict.get('body', {})

            date = input_dict['date']
            site_id = input_dict['operating_site']
            bank_id = input_dict['bank_id']

            if site_id == 1:
                site = ''
            elif site_id == 2:
                site = ''
            elif site_id == 3:
                site = ''
            elif site_id == 4:
                site = ''
            elif site_id == 5:
                site = ''
            else:
                raise KeyError(f"Invalid site_id: {site_id}. Site ID should be between 1 and 5.")


            conn = psycopg2.connect(host='',
                                    dbname=site,
                                    user='',
                                    password='',
                                    port='')
            
            time_delta = timedelta(hours=12, minutes=48)
            start_time = datetime.strptime(date, "%Y-%m-%d %H:%M:%S") - time_delta
            start = time.time()

            
            queries = f"""
                        select
                            "TIMESTAMP",
                            "BANK_DC_VOLT",
                            "BANK_DC_CURRENT",
                            "BANK_SOC",
                            "MAX_CELL_TEMPERATURE_OF_BANK",
                            ("MAX_CELL_VOLTAGE_OF_BANK" - "MIN_CELL_VOLTAGE_OF_BANK") AS "VOLTAGE_GAP",
                            "BANK_ID"
                        from bank
                        where 
                            "TIMESTAMP" >= '{start_time.strftime("%Y-%m-%d %H:%M:%S")}' 
                            and "TIMESTAMP" < '{date}' 
                            and "BANK_ID" = {bank_id}
                            order by "TIMESTAMP" desc
                        """

            print(queries)
            
            with conn.cursor() as cur:
                cur.execute(queries)
                result = dictfetchall(cur)
            df = pd.DataFrame(result)
    
            if df.empty:
                raise ValueError("No data returned from the database.")
            
            df = df.drop("BANK_ID", axis=1)
            df = df.drop_duplicates(subset='TIMESTAMP')
    
            df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
            df['TIMESTAMP'] = df['TIMESTAMP'].dt.tz_localize(None)
    
            full_time_range = pd.date_range(start=start_time, end=datetime.strptime(date, "%Y-%m-%d %H:%M:%S"), freq='S')[:-1]
    
            full_df = pd.DataFrame(full_time_range, columns=['TIMESTAMP'])
            full_df = pd.merge(full_df, df, on='TIMESTAMP', how='left')
            full_df.fillna(method='ffill', inplace=True)

            full_df = full_df.drop(['TIMESTAMP'], axis=1)

            df_scaled = scaling(site_id, full_df)

            input_data = df_scaled.to_numpy()
            input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(self.device)

            return input_tensor
        
        except Exception as e:
            raise ValueError(f"Preprocessing failed: {e}")


    def inference(self, inputs):
        try:
            with torch.no_grad():
                output = self.post_activation(self.model(inputs))
        
            return output

        except Exception as e:
            raise RuntimeError(f"Inference failed: {e}")





    def postprocess(self, inference_output):
        try:
            output = inference_output.view(-1).tolist()
            
            return output

        except Exception as e:
            raise ValueError(f"Postprocessing failed: {e}")

    
    def handle(self, data, context):
        print(f"Handling data: {data}")
        try:
            model_input = self.preprocess(data)
        
            inference_output = self.inference(model_input)
            
            # return inference_output

            result = self.postprocess(inference_output)
            
            return [json.dumps({"prediction": result})]

        except Exception as e:
            print(f"Error in handle method: {e}")
            raise



