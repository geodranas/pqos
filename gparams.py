import os

_ROOT=os.path.dirname(__file__)

gparam={
	'Dir':{
		'Root': _ROOT,
		'Datasets': os.path.join(_ROOT, 'datasets'),
		'Logs': os.path.join(_ROOT, 'logs'),
		'Plots': os.path.join(_ROOT, 'plots'),
		'MLmodels':os.path.join(_ROOT, 'ml_models'),
        'AIsim':os.path.join(_ROOT, 'ai_sim'),
        'OMNETsim':os.path.join(_ROOT, 'omnet_sim'),
        'OMNETcsv': os.path.join(_ROOT, 'omnet_sim','csvs'),
        'OMNETpickles': os.path.join(_ROOT, 'omnet_sim', 'pickles'),
	},
    'File': {
    },
    'Name':{
        'CurrDataset':'asym_4bsod4_full_cl26_1hz.csv'
    },
    'Schema':{
        'Delimiter':';',
        'LoggerCsv':['repetition',
                      'curr_dataset_name',
                      'valid_client_pool_per_round',
                      'clients_per_round_train',
                      'clients_per_round_infer',
                      'ml',
                      'learning_rate',
                      'uni_multi',
                      'sliding_window',
                      'hidden_features',
                      'batch_size',
                      'horizon',
                      'algo_param_1',
                      'algo_param_2',
                      'algo_param_3',
                      'algo_param_4',
                      'curr_ddm_result',
                      'curr_round',
                      'client_bw_ul',
                      'client_bw_dl',
                      'client_energy_ul',
                      'client_energy_dl',
                      'client_energy_proc',
                      'cloud_energy',
                      'network_energy_ul',
                      'network_energy_dl',
                      'ml_rmse',
                      'n_rmse',
                      'baseline_rmse',
                      'saved_loss',
                      'ml_smape',
                      'ml_mae',
                      'ml_r2',
                     ],
    }
}

gparam['File']['LoggerTemp']=os.path.join(gparam['Dir']['Logs'],'temp_logger.txt')
gparam['File']['LoggerCsv']=os.path.join(gparam['Dir']['Logs'],'csv_logger.txt')
gparam['File']['CurrDataset']=os.path.join(gparam['Dir']['Datasets'],gparam['Name']['CurrDataset'])


_REPETITION=0

_VALID_CLIENT_POOL_PER_ROUND = 25
_CLIENTS_PER_ROUND_TRAIN = 5
_CLIENTS_PER_ROUND_INFER=20
_MIN_RECORDS_PER_USER = 20
_MIN_RECORDS_PER_ROUND= 500
_SPLIT_METHOD=['hetero_rand',0.8,0.2,1]
_PLOT_CLIENTS=[1,9,19]
_NAIVE_PREDICTION='single'
_DATASET_STEP=1
_T_START=0
_T_PREV = 0
_T_END=36000
_SPLIT_ROUND_TYPE='time'
_ROUND_DURATION=1200
_COL_PREDICTION=0
_RMSE_ZERO_GROUND= 0.02
_KPI_SIZE=1

_ML='CL'
_UNI_MULTI = 8
_SCALER='min_max_minus'
_FLAG_NORMALIZE=True
_LEARNING_RATE=0.01
_DROPOUT=0.8
_SLIDING_WINDOW=5
_HIDDEN_FEATURES = 50
_EPOCHS=200
_STACKED_LSTM=1
_BATCH_SIZE=256
_BIDIRECTIONAL=False
_STATEFUL=False
_OPTIMIZER='adam'
_WEIGHT_DECAY=0
_ACTIVATION='relu'
_ML_MODEL='MLss'
_LOSS_FUNC='MSE'
_HORIZON=2
_EARLY_STOP_MIN_EPOCHS=30
_EARLY_STOP_LOOKTHROUGH_EPOCHS=20
_LOSS_INFINITE_VALUE=99
_ROUNDSTOP=3


_MIN_THROUGHPUT=120000
_POWER_LTE_UL=float(2.5)
_POWER_LTE_DL=float(3.5)
_POWER_TRAIN_UE_LSTM=float(1.8)
_POWER_TRAIN_CLOUD_LSTM= float(125)
_POWER_INF_CLOUD_LSTM=_POWER_TRAIN_CLOUD_LSTM/3
_POWER_AGGREGATION_CLOUD_LSTM = float(15)
_POWER_NUMERICAL_CALC_CLOUD = _POWER_AGGREGATION_CLOUD_LSTM/3
_SPEED_TRAIN_UE_LSTM=300
_SPEED_TRAIN_CLOUD_LSTM=40000
_SPEED_AGGREGATION_CLOUD_LSTM=0.4


_DDM_CLIENTS_MONITORED=10
_DDM_CONV_SAMPLES_MIN=2
_DDM_CONV_SAMPLES_NUM=3
_DDM_DRIFT_SAMPLES_MIN=20
_DDM_DRIFT_WARN_LEVEL=float(2)
_DDM_DRIFT_OUT_LEVEL=float(3)
_DDM_DRIFT_ENHANCE_THRESHOLD_PERC=5

_ADAPT_BETA1 = 0.5
_ADAPT_BETA2 = 0.5
_ADAPT_BETA3 = 0.5

_FLARE_A=8
_FLARE_B=0.3
_FLARE_F=0.35
_FLARE_KEEP_CLIENT=[]

_PERSON_TOTAL_ROUNDS=0
