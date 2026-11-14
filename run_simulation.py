from itertools import product
import gparams
from simulation import Simulation
import time
from gparams import gparam
import random
def run_experiment(sim_id,exp_id):

        tic=time.time()


        gparams._REPETITION = sim_id

        gparams._VALID_CLIENT_POOL_PER_ROUND = 25
        gparams._CLIENTS_PER_ROUND_TRAIN = _CLIENTS_PER_ROUND_TRAIN
        gparams._CLIENTS_PER_ROUND_INFER = _CLIENTS_PER_ROUND_INFER
        gparams._MIN_RECORDS_PER_USER = 20
        gparams._MIN_RECORDS_PER_ROUND = 1150
        gparams._SPLIT_METHOD = ['hetero_rand',0.8,0.2,1]
        gparams._PLOT_CLIENTS = _PLOT_CLIENTS
        gparams._NAIVE_PREDICTION = 'double'
        gparams._DATASET_STEP = _DATASET_STEP
        gparams._T_START = int(1170*_DATASET_STEP)
        gparams._T_PREV=0
        gparams._T_END = _T_END
        gparams._SPLIT_ROUND_TYPE = 'time'
        gparams._ROUND_DURATION = int(1185*_DATASET_STEP)
        gparams._COL_PREDICTION=_COL_PREDICTION

        gparams._ML= _ML
        gparams._UNI_MULTI = _UNI_MULTI
        gparams._SCALER = 'standard'
        gparams._FLAG_NORMALIZE = True
        gparams._LEARNING_RATE = _LEARNING_RATE
        gparams._DROPOUT = 0.8
        gparams._SLIDING_WINDOW = _SLIDING_WINDOW
        gparams._HIDDEN_FEATURES = _HIDDEN_FEATURES
        gparams._EPOCHS = 50
        gparams._STACKED_LSTM = 1
        gparams._BATCH_SIZE = _BATCH_SIZE
        gparams._BIDIRECTIONAL = False
        gparams._STATEFUL = False
        gparams._OPTIMIZER = 'adam'
        gparams._WEIGHT_DECAY = 1e-5
        gparams._ACTIVATION = 'relu'
        gparams._ML_MODEL = 'LSTM_bibatch'
        gparams._LOSS_FUNC = 'MSE'
        gparams._HORIZON = _HORIZON
        gparams._EARLY_STOP_MIN_EPOCHS = 15
        gparams._EARLY_STOP_LOOKTHROUGH_EPOCHS = 5
        gparams._LOSS_INFINITE_VALUE = 99
        gparams._ROUNDSTOP=_ROUNDSTOP

        gparams._MIN_THROUGHPUT = 120000
        gparams._POWER_LTE_UL = float(2.5)
        gparams._POWER_LTE_DL = float(3.5)
        gparams._POWER_TRAIN_UE_LSTM = float(50)
        gparams._POWER_TRAIN_CLOUD_LSTM = float(225)
        gparams._POWER_AGGREGATION_CLOUD_LSTM = float(100)
        gparams._SPEED_TRAIN_UE_LSTM = 25
        gparams._SPEED_TRAIN_CLOUD_LSTM = 900
        gparams._SPEED_AGGREGATION_CLOUD_LSTM = 0.1


        gparams._DDM_CLIENTS_MONITORED = gparams._VALID_CLIENT_POOL_PER_ROUND
        gparams._DDM_CONV_SAMPLES_MIN = 5
        gparams._DDM_CONV_SAMPLES_NUM = 5
        gparams._DDM_DRIFT_SAMPLES_MIN = 5*gparams._DDM_CLIENTS_MONITORED
        gparams._DDM_DRIFT_WARN_LEVEL = _DDM_DRIFT_WARN_LEVEL
        gparams._DDM_DRIFT_OUT_LEVEL = _DDM_DRIFT_OUT_LEVEL
        gparams._DDM_DRIFT_ENHANCE_THRESHOLD_PERC = _DDM_DRIFT_ENHANCE_THRESHOLD_PERC


        gparams._ADAPT_BETA1 = _ADAPT_BETA1
        gparams._ADAPT_BETA2 = _ADAPT_BETA2
        gparams._ADAPT_BETA3 = _ADAPT_BETA3


        gparams._FLARE_A=_FLARE_A
        gparams._FLARE_B=_FLARE_B
        gparams._FLARE_F=_FLARE_F
        gparams._FLARE_KEEP_CLIENT=_FLARE_KEEP_CLIENT


        gparams._PERSON_TOTAL_ROUNDS=_PERSON_TOTAL_ROUNDS

        simulation=Simulation(id=sim_id,exp_id=exp_id)
        simulation.run()

        toc=time.time()
        mystr='Finised exp=' + str(sim_id) +',ml='+str(gparams._ML)+',at time='+str(toc-tic)+ '\n'+'-----------'
        with open(gparam['File']['LoggerTemp'], mode='a') as file:
            file.write(mystr+'\n')

exp_id=1
for repett in range(1,10000):
        print('Running repet='+str(repett))
        _T_END= 72000
        _PLOT_CLIENTS=[]

        for _ML, (train, infer) in product(
            ['FL_ddm_all','FL_oneshot_with_conv_stop','FL_forever'],
            [(5, 20),]
        ):


                """
                if repett == 15:
                        if _ML != "Personalized" or (_ML == "Personalized" and (train, infer) == (5, 20)):
                                print(f"Skipping repet={repett}, _ML={_ML}, train={train}, infer={infer}")
                                continue
                """
                _CLIENTS_PER_ROUND_TRAIN = train
                _CLIENTS_PER_ROUND_INFER = infer

                _ROUNDSTOP=1
                _DATASET_STEP=1
                _UNI_MULTI=11
                _COL_PREDICTION=0
                _SLIDING_WINDOW= 75
                _HIDDEN_FEATURES=50
                _HORIZON=int(20/_DATASET_STEP)

                if _CLIENTS_PER_ROUND_TRAIN == 5 and _CLIENTS_PER_ROUND_INFER == 20:
                        _DDM_DRIFT_ENHANCE_THRESHOLD_PERC = 5
                        _FLARE_A=16
                        _FLARE_B=0.1
                        _FLARE_F=0.15
                elif _CLIENTS_PER_ROUND_TRAIN == 10 and _CLIENTS_PER_ROUND_INFER == 10:
                        _DDM_DRIFT_ENHANCE_THRESHOLD_PERC = 10
                        _FLARE_A=16
                        _FLARE_B=0.1
                        _FLARE_F=0.15
                elif _CLIENTS_PER_ROUND_TRAIN == 20 and _CLIENTS_PER_ROUND_INFER == 5:
                        _DDM_DRIFT_ENHANCE_THRESHOLD_PERC = 10
                        _FLARE_A=16
                        _FLARE_B=0.3
                        _FLARE_F=0.15
                else:
                        _DDM_DRIFT_ENHANCE_THRESHOLD_PERC = 10
                        _FLARE_A=16
                        _FLARE_B=0.1
                        _FLARE_F=0.15

                _DDM_DRIFT_WARN_LEVEL = float(2)
                _DDM_DRIFT_OUT_LEVEL = float(3)
                _PERSON_TOTAL_ROUNDS=60
                _ADAPT_BETA1 = 0.7
                _ADAPT_BETA2 = 0.7
                _ADAPT_BETA3 = 0.7

                if _ML == 'CL':
                        _FLARE_KEEP_CLIENT=[]
                        _BATCH_SIZE=128
                        _LEARNING_RATE=1e-4
                elif _ML=='Flare':
                        steady_list=list(range(gparams._VALID_CLIENT_POOL_PER_ROUND))
                        steady_selection = random.sample(steady_list, _CLIENTS_PER_ROUND_TRAIN)
                        _FLARE_KEEP_CLIENT = steady_selection
                        _BATCH_SIZE = 64
                        _LEARNING_RATE = 1e-5
                else:
                        _FLARE_KEEP_CLIENT = []
                        _BATCH_SIZE = 64
                        _LEARNING_RATE = 1e-5

                run_experiment(sim_id=repett,exp_id=exp_id)
                exp_id=exp_id+1
