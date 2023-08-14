import json
import warnings

import dash
from dash import dcc,html, clientside_callback,dash_table

from dash.dependencies import Input, Output,State
import plotly.graph_objs as go
import pandas as pd
import datetime
from dash_extensions import WebSocket

warnings.filterwarnings(action='ignore')

range_dict = {
    'CPU':[0,100],
   #'Temperature':[30,120],
   'Stack usage':[0,2000]
}





usage_data = pd.DataFrame(columns = 'timestamp,CPU,Stack usage,Heap current,Heap max,Temperature,Time stamp,classification,DeepAnt,USAD'.split(','))
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(title='ESS DEMO', external_stylesheets=external_stylesheets,update_title= None)

app.layout = html.Div ([

    WebSocket(id = 'ws', url = 'ws://10.10.10.61:1223/data'),
        html.Div([
            html.H1('ESS DEMO',style={'text-align':'center','font-size':'50px','padding':'20px'}),
            
            html.Div(
            html.Div(className= 'row', children=[
            html.Div(children=[
                html.Label(['Time Range'], style={'font-weight': 'bold', "text-align": "center"}),
                dcc.Dropdown(
                    id='dropdown-time_ragne',
                    options=[
                        {'label': 'Last 1 Day', 'value': '1_day'},
                        {'label': 'Last 3 Days', 'value': '3_day'},
                        {'label': 'Last 1 Week', 'value': '1_week'}

                    ],
                    value='3_day',
                    searchable=False,
                    clearable=False,
                    style={'justify-content':'center','text-align':'center',},
                    ),
               
                 ],style={'justify-content':'center','width':'25%','padding':'25px',}),
                 html.Div(children=[
                        html.Label(['Period'],style={'font-weight': 'bold', "text-align": "center"}),
                        dcc.Dropdown(
                            id='dropdown_period',
                            options=[
                                {'label': '10 Sec', 'value': '10 sec'},
                                {'label': '30 Sec', 'value': '30 sec'},
                                {'label': '1 Min', 'value': '1 min'}

                            ],
                            value='30 sec',
                            searchable=False,
                            clearable=False,
                            style={'justify-content':'center','text-align':'center'},
                            ),   
                        #label과 dropdown을 같이 묶고 스타일 적용
                    ],style={'justify-content':'center','width':'25%','padding':'25px',}), #dict(width='15%',padding='25px')
            ],style={'display':'flex','text-align':'center','justify-content':'center','align-items':'center'}), #style=dict(width='33.33%')
),

                html.Div([
                
                html.Div(children=[
                html.Div(children=[
                html.Label('Live data',style={'display':'flex','font-size':'25px','justify-content':'center','padding':'15px'}),
                dcc.Dropdown(id='graph_dd',
                            options=[{'label':'CPU usage','value':'CPU'},
                                    {'label':'Stack usage','value':'Stack usage'}],
                            value='CPU',
                            searchable=False,
                            clearable=False,    

                            style ={'display':'flex','justify-content':'right','text-align':'center','font-size':'15px','padding':'15px','width':'250px','align-items':'center',},
                                                                                                        
                            ),
                ],style={'display':'flex','justify-content':'space-between','width':'auto'}),
                dcc.Graph(id='live_figure',style={'display':'flex','justify-content':'center','align-items':'center','box-shadow':'0 0 10px 1px gray'}),
                html.Hr(), 
                    html.Div(children=[
                    html.Label('Anomaly history',style={'display':'flex','font-size':'25px','justify-content':'left','padding':'15px'}),
                    dash_table.DataTable(
                        id='log_data',
                        columns=[{'name':col, 'id':col} for col in usage_data.keys()],
                        data=[],
                        style_header={'font-size':'17px','text-align':'center','width':'auto','height':'auto'}, 
                        style_cell={'font-size':'15px','font-family':'Nanum gotihc','width':'auto','height':'auto','overflow':'hidden','textOverflow':'ellipsis','maxWidth':0,},
                        style_data={
                            'whiteSpace':'normal',
                            'height':'auto',
                            'width':'auto',
                        }
                        )
                    ],style={'display':'flex-inline','justify-content':'center','text-align':'center',
                             'font-size':'15px','padding':'15px','align-items':'center','width':'auto','height':'auto',}),

                
            ],style={'display':'flex-inline','justify-content':'center','align-items':'center',
                    'border':'0.5px solid #C8C8C8','font-size':'30px','width':'75%','padding':'20px'}),
                    ],style={'display':'flex','justify-content':'center','align-items':'center','width':'auto'}),
            
                     
            
        ],style = {'display':'flex-inline','flex-flow':'column wrap','width':'85%','height':'95%','background':'#FFFFFF', 'padding':'5px 5px 5px 5px','box-shadow':'0 0 10px 5px gray'}),
        
        
        dcc.Store(id='dataframe_memory',
                  data = usage_data.to_dict()),
        dcc.Store(id='figure_data'),
        
        
], style = {'display':'flex', 'justify-content':'center','background':'#F8F8F8'})#'box-shadow':'0 0 10px 5px gray', 'width':'98vw','height':'98vh'

@app.callback(#Output('log_data','data'),
              Output('figure_data','data'),
              Output('dataframe_memory','data'),
              Input('ws','message'),
              Input('graph_dd','value'), 
              State('dataframe_memory','data'),
              
              
              
              prevent_initial_call=True
              )
def update_graph(message,graph_dd, usage_data):
    # 데이터 저장 틀
    usage_data = pd.DataFrame(usage_data)
    # 결과 메시지
    message = json.loads(message['data'])
    
    data = message['data']
    deep_ant_res = message['DeepAnt']
    usad_res = message['USAD']
    
    # 데이터 추가
    usage_data = pd.concat([usage_data.iloc[-399:], pd.DataFrame(data)]).reset_index(drop = True)
    usage_data.timestamp = pd.to_datetime(usage_data.timestamp)
    
    if deep_ant_res != None:
        for key, value in deep_ant_res.items():
            usage_data.loc[usage_data['timestamp'] == key,'DeepAnt'] = value
    if usad_res != None:        
        if list(usad_res.values())[0] == True:
            idx = usage_data.loc[usage_data.timestamp == list(usad_res.keys())[0]].index
            idx = list(idx)[0]
            usage_data.loc[idx:idx+128,'USAD'] = True
        

    usage_data.fillna(False, inplace = True)
    
    
    
    
    # 그래프 생성
    fig = go.Figure()
    layout = go.Layout(title=f'Real-Time {graph_dd} Visualization', xaxis=dict(title='Time'), yaxis=dict(title='Value')) 
    
    
    
    # 원본 데이터
    fig.add_trace(
        go.Scatter(
        x=usage_data['timestamp'],
        y=usage_data[graph_dd],
        mode='lines',
        name = 'Normal Data',
        line=dict(shape='linear')
    ))
    
    # usad model
    usad_data = usage_data.loc[usage_data.USAD != False]
    if len(usad_data):
        usad_data.loc[:,'idx'] = usad_data.loc[:,'timestamp'].diff().gt(datetime.timedelta(seconds = 1)).cumsum()
        for __,data in  usad_data.groupby('idx'):
            fig.add_trace(
                go.Scatter(
                x = data['timestamp'],
                y = data[graph_dd], 
                mode = 'lines',
                name = 'USAD_Anomaly',
                
                line = dict(shape = 'linear', color = 'red')
                
            ))
    
    # DeepAnt 모델
    fig.add_trace(
        go.Scatter(
            
            x = usage_data.loc[usage_data.DeepAnt != False,'timestamp'],
            y = usage_data.loc[usage_data.DeepAnt != False,graph_dd],
            mode = 'markers',
            name = 'DeepAnt_Anomaly',
            marker=dict(
            color='red',  # 일정 값 이상일 때 색 변경
            size=10
                    )
        )
    )
    
    
    fig.update_yaxes(range = range_dict[graph_dd])
    fig.update_xaxes(range = [usage_data.at[0,'timestamp'], usage_data.at[len(usage_data)-1,'timestamp']])
    
    
    

    fig.update_layout(layout)
    anomaly_data =  usage_data.loc[usage_data.DeepAnt != False]
    client_data = {}
    
    client_data['figure'] = fig
    client_data['anomaly'] = anomaly_data.to_dict('records')
    
    return client_data,usage_data.to_dict()



clientside_callback(
    """
    function(data){
        if (data === undefined){
            
            return {'data':[],'layout':{}};
        } 
        const fig = Object.assign({}, data['figure'])
        const anomaly = Object.assign([], data['anomaly'])
       
        
        
        return [fig, anomaly];
    }
    
    
    """,
    Output('live_figure', 'figure'),
    Output('log_data','data'),
    
    
    Input('figure_data','data')
      
    
)



if __name__ == '__main__':
    app.run(host='10.10.10.61',port=2600,debug=False,use_reloader=False,dev_tools_silence_routes_logging=True,threaded = True)
