import gparams
from worker import Clients,Cloud
from dataset import Dataset_OMNET,Dataset_BERLIN, Dataset_HIDRIVE
from core import Core
from gparams import gparam
import time
class Simulation:
    def __init__(self,id,exp_id):
        self.id=id
        self.curr_round=1
        self.curr_time=gparams._T_START
        self.prev_time=gparams._T_PREV
        self.t_end=gparams._T_END
        self.exp_id=exp_id

        dataset=Dataset_OMNET()
        network=Core(hops=3)
        self.clients=Clients(subdatasets=dataset.get_subdatasets(),network=network)
        self.cloud=Cloud(id=9999,subdataset=dataset.get_schema())

    def run(self):
        while self.curr_time<self.t_end:

            if gparams._SPLIT_ROUND_TYPE=='data':
                # todo
                exit(401)


            elif gparams._SPLIT_ROUND_TYPE=='time':
                is_time = (self.curr_time-self.prev_time>=gparams._ROUND_DURATION)
                have_enough_clients=self.clients.create_availability_list(min_time=self.prev_time,
                                    max_time=self.curr_time,
                                    min_records_per_round=gparams._MIN_RECORDS_PER_ROUND,
                                    valid_clients_pool=gparams._VALID_CLIENT_POOL_PER_ROUND)
                are_clients_ready=(is_time) and (have_enough_clients)

            if not are_clients_ready:
                self.curr_time = self.curr_time + gparams._DATASET_STEP
                print('Clients not ready, lets move')
                continue

            print('-- Entering round='+str(self.curr_round)+', at time='+str(self.curr_time))

            self.clients.client_selection(train_clients=gparams._CLIENTS_PER_ROUND_TRAIN,
                                          infer_clients=gparams._CLIENTS_PER_ROUND_INFER,
                                          strategy=gparams._SPLIT_METHOD,
                                          min_time=self.prev_time,
                                          max_time=self.curr_time)
            self.prev_time=self.curr_time
            if gparams._ML=='CL':
                list_of_client_curr_raw_train,max_time1=self.clients.upload_curr_raw_train(selected=self.clients.curr_train_clients_list,curr_time=self.curr_time)
                list_of_client_curr_raw_valid,max_time2 = self.clients.upload_curr_raw_valid(selected=self.clients.curr_train_clients_list,curr_time=self.curr_time)

                this_rounds_model=self.cloud.central_train(train_list=list_of_client_curr_raw_train,
                                                           valid_list=list_of_client_curr_raw_valid)

                self.clients.infer(model='existing', selected=self.clients.curr_infer_clients_list,plot_list=gparams._PLOT_CLIENTS,curr_round=self.curr_round)
                self.clients.download_model(model=this_rounds_model,
                                            curr_time=self.curr_time)  # todo: check optimizers etc if need to copy + model.eval model.train etc
            elif gparams._ML=='SL':
                curr_raw_train=self.clients.upload_curr_raw_train(selected=self.clients.curr_train_clients_list,curr_time=self.curr_time)
                curr_raw_valid = self.clients.upload_curr_raw_valid(selected=self.clients.curr_train_clients_list,curr_time=self.curr_time)
                try:
                    list_of_client_curr_raw_train.extend(curr_raw_train)
                    list_of_client_curr_raw_valid.extend(curr_raw_valid)
                except:
                    list_of_client_curr_raw_train=curr_raw_train
                    list_of_client_curr_raw_valid=curr_raw_valid

                self.cloud.reset_model()
                this_rounds_model=self.cloud.central_train(train_list=list_of_client_curr_raw_train,
                                                           valid_list=list_of_client_curr_raw_valid)

                self.clients.infer(model='existing', selected=self.clients.curr_infer_clients_list)
                self.clients.download_model(model=this_rounds_model,curr_time=self.curr_time)
            elif gparams._ML=='CL_roundstop':
                self.cloud.curr_ddm_result=gparams._ROUNDSTOP
                if self.curr_round<gparams._ROUNDSTOP:
                    list_of_client_curr_raw_train,max_time1=self.clients.upload_curr_raw_train(selected=self.clients.curr_train_clients_list,curr_time=self.curr_time)
                    list_of_client_curr_raw_valid,max_time2 = self.clients.upload_curr_raw_valid(selected=self.clients.curr_train_clients_list,curr_time=self.curr_time)

                    this_rounds_model=self.cloud.central_train(train_list=list_of_client_curr_raw_train,
                                                               valid_list=list_of_client_curr_raw_valid)

                    self.clients.infer(model='existing', selected=self.clients.curr_infer_clients_list,plot_list=gparams._PLOT_CLIENTS,curr_round=self.curr_round)
                    self.clients.download_model(model=this_rounds_model,
                                                curr_time=self.curr_time)  # todo: check optimizers etc if need to copy + model.eval model.train etc
                else:
                    self.clients.infer(model='existing', selected=self.clients.curr_infer_clients_list)
            elif gparams._ML=='SL_roundstop':
                self.cloud.curr_ddm_result = gparams._ROUNDSTOP
                if self.curr_round<gparams._ROUNDSTOP:
                    curr_raw_train=self.clients.upload_curr_raw_train(selected=self.clients.curr_train_clients_list,curr_time=self.curr_time)
                    curr_raw_valid = self.clients.upload_curr_raw_valid(selected=self.clients.curr_train_clients_list,curr_time=self.curr_time)
                    try:
                        list_of_client_curr_raw_train.extend(curr_raw_train)
                        list_of_client_curr_raw_valid.extend(curr_raw_valid)
                    except:
                        list_of_client_curr_raw_train=curr_raw_train
                        list_of_client_curr_raw_valid=curr_raw_valid

                    self.cloud.reset_model()
                    this_rounds_model=self.cloud.central_train(train_list=list_of_client_curr_raw_train,
                                                               valid_list=list_of_client_curr_raw_valid)

                    self.clients.infer(model='existing', selected=self.clients.curr_infer_clients_list)
                    self.clients.download_model(model=this_rounds_model,curr_time=self.curr_time)
                else:
                    self.clients.infer(model='existing', selected=self.clients.curr_infer_clients_list)
            elif gparams._ML=='XGBoost':
                list_of_client_curr_raw_train,max_time1=self.clients.upload_curr_raw_train(selected=self.clients.curr_train_clients_list,curr_time=self.curr_time)
                list_of_client_curr_raw_valid,max_time2 = self.clients.upload_curr_raw_valid(selected=self.clients.curr_train_clients_list,curr_time=self.curr_time)

                this_rounds_model=self.cloud.central_train(train_list=list_of_client_curr_raw_train,
                                                           valid_list=list_of_client_curr_raw_valid)

                self.clients.infer(model='existing', selected=self.clients.curr_infer_clients_list,plot_list=gparams._PLOT_CLIENTS,curr_round=self.curr_round)
                self.clients.download_model(model=this_rounds_model,
                                            curr_time=self.curr_time)  # todo: check optimizers etc if need to copy + model.eval model.train etc
            elif gparams._ML=='FL_forever':
                if self.curr_round==1:
                    self.clients.download_model(model=self.cloud.model,curr_time=self.curr_time)
                all_models=self.clients.local_train_and_upload_models(selected=self.clients.curr_train_clients_list,curr_time=self.curr_time)
                self.clients.infer(model='existing',selected=self.clients.curr_infer_clients_list,plot_list=gparams._PLOT_CLIENTS,curr_round=self.curr_round)
                this_rounds_model = self.cloud.aggregate(all_models)
                self.clients.download_model(model=this_rounds_model,curr_time=self.curr_time)
            elif gparams._ML=='Adapt_FL':

                if self.curr_round==1:
                    self.clients.download_model(model=self.cloud.model,curr_time=self.curr_time)
                if self.curr_round>1:
                    self.clients.update_lr(lr=self.cloud.curr_ddm_result,selected=self.clients.curr_train_clients_list,curr_time=self.curr_time)
                all_models=self.clients.local_train_and_upload_models(selected=self.clients.curr_train_clients_list,curr_time=self.curr_time)
                self.clients.infer(model='existing',selected=self.clients.curr_infer_clients_list,plot_list=gparams._PLOT_CLIENTS,curr_round=self.curr_round)
                this_rounds_model = self.cloud.aggregate(all_models)
                self.cloud.adaptive_calc(this_rounds_model,self.curr_round)
                self.clients.download_model(model=this_rounds_model,curr_time=self.curr_time)
            elif gparams._ML=='FL_freq_5rounds':
                if self.curr_round%5==0 or self.curr_round==1:
                    if self.curr_round == 1:
                        self.clients.download_model(model=self.cloud.model, curr_time=self.curr_time)
                    all_models=self.clients.local_train_and_upload_models(selected=self.clients.curr_train_clients_list,curr_time=self.curr_time)
                    self.clients.infer(model='existing', selected=self.clients.curr_infer_clients_list)
                    this_rounds_model = self.cloud.aggregate(all_models)
                    self.clients.download_model(model=this_rounds_model,curr_time=self.curr_time)
                else:
                    self.clients.infer(model='existing', selected=self.clients.curr_infer_clients_list)
            elif gparams._ML=='FL_freq_3rounds':
                if self.curr_round%3==0 or self.curr_round==1:
                    if self.curr_round == 1:
                        self.clients.download_model(model=self.cloud.model, curr_time=self.curr_time)
                    all_models=self.clients.local_train_and_upload_models(selected=self.clients.curr_train_clients_list,curr_time=self.curr_time)
                    self.clients.infer(model='existing', selected=self.clients.curr_infer_clients_list)
                    this_rounds_model = self.cloud.aggregate(all_models)
                    self.clients.download_model(model=this_rounds_model,curr_time=self.curr_time)
                else:
                    self.clients.infer(model='existing', selected=self.clients.curr_infer_clients_list)
            elif gparams._ML=='FL_roundstop':
                self.cloud.curr_ddm_result = gparams._ROUNDSTOP
                if self.curr_round<gparams._ROUNDSTOP:
                    if self.curr_round==1:
                        self.clients.download_model(model=self.cloud.model,curr_time=self.curr_time)
                    all_models=self.clients.local_train_and_upload_models(selected=self.clients.curr_train_clients_list,curr_time=self.curr_time)
                    self.clients.infer(model='existing', selected=self.clients.curr_infer_clients_list)
                    this_rounds_model = self.cloud.aggregate(all_models)
                    self.clients.download_model(model=this_rounds_model,curr_time=self.curr_time)
                else:
                    self.clients.infer(model='existing', selected=self.clients.curr_infer_clients_list)
            elif gparams._ML=='FL_ddm_all' or gparams._ML=='FL_oneshot_with_conv_stop':
                if gparams._ML=='FL_ddm_all':
                    check_drift=True
                else:
                    check_drift=False

                if self.curr_round==1:
                    self.clients.download_model(model=self.cloud.model,curr_time=self.curr_time)

                calc_start=time.time()
                res=self.cloud.is_dead_round(check_drift=check_drift,check_converge=True,curr_round=self.curr_round)
                calc_end = time.time()
                self.cloud.energy = self.cloud.energy + (calc_end - calc_start) * gparams._POWER_NUMERICAL_CALC_CLOUD
                if not res:
                    all_models=self.clients.local_train_and_upload_models(selected=self.clients.curr_train_clients_list,curr_time=self.curr_time)

                    self.clients.infer(model='existing', selected=self.clients.curr_infer_clients_list)

                    kpi_list=self.clients.upload_stats(selected=self.clients.curr_infer_clients_list,curr_time=self.curr_time)
                    self.cloud.curr_client_kpi_list = kpi_list
                    this_rounds_model = self.cloud.aggregate(all_models)
                    self.clients.download_model(model=this_rounds_model,curr_time=self.curr_time)
                else:

                    self.clients.infer(model='existing', selected=self.clients.curr_infer_clients_list)

                    kpi_list=self.clients.upload_stats(selected=self.clients.curr_infer_clients_list,curr_time=self.curr_time)
                    self.cloud.curr_client_kpi_list = kpi_list
            elif gparams._ML=='Flare':
                if self.curr_round==1:
                    self.clients.download_model(model=self.cloud.model,curr_time=self.curr_time)

                all_models=self.clients.local_train_and_upload_models_flare(selected=self.clients.curr_train_clients_list,
                                                                      curr_time=self.curr_time)

                self.clients.infer(model='existing', selected=self.clients.curr_infer_clients_list)

                if len(all_models)>0:
                    this_rounds_model = self.cloud.aggregate(all_models)
                    self.clients.download_model(model=this_rounds_model,curr_time=self.curr_time)
            elif gparams._ML=='Personalized':

                self.clients.download_model(model=self.cloud.model, curr_time=self.curr_time,client_list=self.clients.curr_train_clients_list)
                all_models=self.clients.local_train_and_upload_models_person(selected=self.clients.curr_train_clients_list,curr_time=self.curr_time,curr_round=self.curr_round)


                outdated_tester_list=self.clients.person_get_list_outdated_clients(selected=self.clients.curr_infer_clients_list,curr_round=self.curr_round)
                self.clients.download_model(model=self.cloud.model, curr_time=self.curr_time,client_list=outdated_tester_list)
                self.clients.infer_person(model=self.cloud.model,selected=self.clients.curr_infer_clients_list,plot_list=gparams._PLOT_CLIENTS,curr_round=self.curr_round)


                self.cloud.aggregate(all_models)


                self.clients.person_save_data()
            else:
                print('Cannot find ML scheme')
                print('ml_scheme='+str(gparams._ML))
                exit(0)

            self.write_round_log()
            self.curr_round=self.curr_round+1

            self.curr_time=self.curr_time+gparams._ROUND_DURATION      # todo: calculate time passed by transmission params

    def get_algo_param(self):
        if gparams._ML == 'Adapt_FL':
            return gparams._ADAPT_BETA1, gparams._ADAPT_BETA2, gparams._ADAPT_BETA3, -1
        elif gparams._ML == 'Flare':
            return gparams._FLARE_A, gparams._FLARE_B, gparams._FLARE_F, -1
        elif gparams._ML == 'Personalized':
            return gparams._PERSON_TOTAL_ROUNDS, -1, -1, -1
        elif gparams._ML == 'FL_ddm_all':
            return (
                gparams._DDM_DRIFT_ENHANCE_THRESHOLD_PERC,
                gparams._DDM_DRIFT_WARN_LEVEL,
                gparams._DDM_DRIFT_OUT_LEVEL,
                gparams._DDM_CONV_SAMPLES_NUM,
            )
        elif gparams._ML=='FL_oneshot_with_conv_stop':
            return -1, -1, -1,gparams._DDM_CONV_SAMPLES_NUM


        return None, None, None, None

    def write_round_log(self):

        ml_rmse,baseline_rmse,n_rmse,saved_loss,ml_smape,ml_mae,ml_r2=self.clients.get_round_ml_stats()

        client_bw_UL=self.clients.get_client_bw_UL()
        client_bw_DL = self.clients.get_client_bw_DL()
        client_energy_UL = self.clients.get_client_energy_UL()
        client_energy_DL = self.clients.get_client_energy_DL()
        client_energy_PROC=self.clients.get_client_energy_PROC()
        cloud_energy=self.cloud.get_cloud_energy()
        network_energy_UL=self.clients.get_network_energy_UL()
        network_energy_DL=self.clients.get_network_energy_DL()
        al1,al2,al3,al4=self.get_algo_param()

        mystr=''
        results_list=[gparams._REPETITION,
                      gparam['Name']['CurrDataset'],
                      gparams._VALID_CLIENT_POOL_PER_ROUND,
                      gparams._CLIENTS_PER_ROUND_TRAIN,
                      gparams._CLIENTS_PER_ROUND_INFER,
                      gparams._ML,
                      gparams._LEARNING_RATE,
                      gparams._UNI_MULTI,
                      gparams._SLIDING_WINDOW,
                      gparams._HIDDEN_FEATURES,
                      gparams._BATCH_SIZE,
                      gparams._HORIZON,
                      al1,
                      al2,
                      al3,
                      al4,
                      self.cloud.curr_ddm_result,
                      self.curr_round,
                      client_bw_UL,
                      client_bw_DL,
                      client_energy_UL,
                      client_energy_DL,
                      client_energy_PROC,
                      cloud_energy,
                      network_energy_UL,
                      network_energy_DL,
                      ml_rmse,
                      n_rmse,
                      baseline_rmse,
                      saved_loss,
                      ml_smape,
                      ml_mae,
                      ml_r2
                      ]
        for el in results_list:
            mystr=mystr+str(el)+gparam['Schema']['Delimiter']
        mystr = mystr[:-1]+'\n'


        myheader=''
        for feat in gparam['Schema']['LoggerCsv']:
            myheader=myheader+feat+gparam['Schema']['Delimiter']
        myheader=myheader[:-1]+'\n'

        with open(gparam['File']['LoggerCsv'], mode='a') as file:
            if self.curr_round==1 and self.id==1 and self.exp_id==1:
                file.write(myheader)
            file.write(mystr)
