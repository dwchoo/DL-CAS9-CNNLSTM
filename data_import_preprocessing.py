import pandas as pd
import numpy as np
from copy import deepcopy
import os
from sklearn.model_selection import train_test_split

class import_data_preprocessing:

    num_mismatch_max = 5
    num_indel_class = 11


    seq_one_hot_encoder = {
        'A': np.array([1, 0, 0, 0], dtype= np.int),
        'C': np.array([0, 1, 0, 0], dtype= np.int),
        'G': np.array([0, 0, 1, 0], dtype= np.int),
        'T': np.array([0, 0, 0, 1], dtype= np.int),
        'N': np.array([0, 0, 0, 0], dtype= np.int)
    }

    mismatch_one_hot_encoder = {}
    for i in range(num_mismatch_max):
        initial_list = [0] * num_mismatch_max
        initial_list[i] = int(1)
        mismatch_one_hot_encoder[i] = initial_list

    indel_class_one_hot_encoder = {}
    for i in range(num_indel_class):
        rate = i /10
        initial_list = [0] * num_indel_class
        initial_list[i] = int(1)
        indel_class_one_hot_encoder[rate] = initial_list



    def __init__(self,
                 train_data_file_name = 'HT_Cas9_train',
                 test_data_file_name = None,
                 cut_off_read_count = None,
                 split_train_val = 0.2,
                 RD_seed = 1234
                 ):
        self.train_data_file_name = train_data_file_name
        self.test_data_file_name = test_data_file_name
        self.cut_off_read_count = cut_off_read_count
        self.split_train_val = split_train_val
        self.RD_seed = RD_seed

        # CSV file columns check
        self.train_data_file_sample = pd.read_csv(train_data_file_name, nrows=1)
        self.train_data_file_sample_column = list(self.train_data_file_sample.columns)
        self.test_data_file_sample = None
        self.test_data_file_sample_column = None
        if test_data_file_name is not None:
            self.test_data_file_sample = pd.read_csv(test_data_file_name, nrows=1)
            self.test_data_file_sample_column = list(self.test_data_file_sample.columns)


    def Read_csv_file(self, file_path,
                      sgRNA_column = 'sgRNA',
                      targetDNA_column = None,
                      indel_column = 'indel',
                      read_cnt_column = None,
                      num_mis_column = None,
                      ):
        ###
        '''
        
        '''
        ###
        
        
        # Remove None column
        column_list = [sgRNA_column, targetDNA_column, indel_column, read_cnt_column, num_mis_column]
        usecols = list(filter(None, column_list))
        
    
        # Read CSV file
        selected_df = pd.read_csv(filepath_or_buffer=file_path,
                                  usecols=usecols)
        
        # Exclude less than cut off reads
        if read_cnt_column is not None:
            selected_df = selected_df.loc[selected_df[read_cnt_column] > self.cut_off_read_count]

        ####
        """
        targetDNA 칼럼은 일단 유지한다. on-target 예측시 같은 것을 2D로 올려서 계산하기 위해서 일단 남겨둔다. 
        """
        ####

        if targetDNA_column is None:
            sgRNA = np.array(selected_df[sgRNA_column])
            targetDNA = sgRNA
            indel = np.array(selected_df[indel_column])
        else:
            sgRNA = np.array(selected_df[sgRNA_column])
            targetDNA = np.array(selected_df[targetDNA_column])
            indel = np.array(selected_df[indel_column])
            
        if read_cnt_column is not None:
            read_cnt = np.array(selected_df[read_cnt_column])
        else:
            read_cnt = None
        

        # if indel is less than 0, make 0
        indel = self.indel_change(indel)

        return {'sgRNA' : sgRNA, 'targetDNA' : targetDNA, 'indel' : indel,
                'read_cnt' : read_cnt}


    def indel_change(self, indel_list):
        indel_list = np.array(indel_list)
        if indel_list.max() > 1.0:
            new_indel_list = list(map(lambda x: x / 100, indel_list))
        else:
            new_indel_list = indel_list
        new_indel_list = list(map(lambda x: 0 if x < 0 else x, new_indel_list))

        return np.array(new_indel_list)


    def Unify_length_processNGG(self,sgRNA, targetDNA,
                                sgRNA_have_NGG = True,
                                delete_NGG = True,
                                PAM_length = 3):
        PAM_length = PAM_length
        sgRNA = sgRNA.upper()
        targetDNA = targetDNA.upper()

        # Preprocessing NGG
        if sgRNA_have_NGG and delete_NGG:
            # delete PAM and fill Null(N) data
            sgRNA = sgRNA[:-PAM_length]
            sgRNA = sgRNA + 'N' * PAM_length
        elif sgRNA_have_NGG:
            # Remain NGG, but change 'N'
            if 'N' in sgRNA and not 'N' in targetDNA:
                N_position = sgRNA.find('N')
                sgRNA[N_position] = targetDNA[N_position]
            elif not 'N' in sgRNA and 'N' in targetDNA:
                N_position = sgRNA.find('N')
                targetDNA[N_position] = sgRNA[N_position]

        #Unify length
        sgRNA_length = len(sgRNA)
        targetDNA_length = len(targetDNA)

        #If sgRNA is short, append Null(N)
        if targetDNA_length > sgRNA_length:
            short = targetDNA_length - sgRNA_length
            sgRNA = 'N'*short + sgRNA

        return sgRNA, targetDNA #{'sgRNA' : sgRNA, 'targetDNA' : targetDNA}
    
    
    def mismatch_position_N_number(self, encoded_sgRNA, encoded_targetDNA):
        xor_seq = np.bitwise_xor(encoded_sgRNA, encoded_targetDNA)
        mismatch_position = list(map(lambda x: sum(x)//2 , xor_seq))
        mismatch_number = sum(mismatch_position)
        
        return mismatch_position, mismatch_number
    
    #### 이 부분 수정
    def data_information(self, data_list):
        data_list__ = [x for x in data_list if x is not None]
        data_info = np.stack(data_list__, axis=-1)
        return data_info
        

    def seq_one_hot_encoding(self,sequence):
        '''
        one_hot_encoder = {
            'A': np.array([1, 0, 0, 0]),
            'C': np.array([0, 1, 0, 0]),
            'G': np.array([0, 0, 1, 0]),
            'T': np.array([0, 0, 0, 1]),
            'N': np.array([0, 0, 0, 0])
        }
        '''
        one_hot_seq = []
        for bp in list(sequence):
            one_hot_seq += [import_data_preprocessing.seq_one_hot_encoder['{}'.format(bp)]]

        one_hot_seq = np.array(one_hot_seq)

        return one_hot_seq

    
    def mismatch_one_hot_encoding(self, mismatch_num):
        '''
        one_hot_encoder = {}
        for i in range(max_num):
            initial_list = [0] * max_num
            initial_list[i] = 1
            one_hot_encoder[i] = initial_list
        '''
        one_hot_mismatch = import_data_preprocessing.mismatch_one_hot_encoder[mismatch_num]
        return one_hot_mismatch


    def indel_class_one_hot_encoding(self, indel_rate):
        """
        indel_class_one_hot_encoder = {}
        for i in range(num_indel_class):
            rate = i / 10
            initial_list = [0] * num_indel_class
            initial_list[i] = 1
            indel_class_one_hot_encoder[rate] = initial_list
        """
        indel_rate_class = round(indel_rate,1)
        one_hot_indel_class = import_data_preprocessing.indel_class_one_hot_encoder[indel_rate_class]
        return one_hot_indel_class


    def data_preprocessing_ontarget(self, _file_path, sgRNA_column, indel_column,
                                          sgRNA_have_NGG = True, delete_NGG = False,
                                          PAM_length = 3):

        # {'sgRNA' : sgRNA, 'targetDNA' : targetDNA, 'indel' : indel}
        raw_data = self.Read_csv_file(file_path=_file_path,
                                     sgRNA_column=sgRNA_column, targetDNA_column=None,
                                     indel_column=indel_column
                                     )

        # One-hot encoding
        encoded_sgRNA = np.array(list(map(self.seq_one_hot_encoding,raw_data['sgRNA'])))
        encoded_targetDNA = np.array(list(map(self.seq_one_hot_encoding,raw_data['targetDNA'])))

        # stack operation, make 2D
        #stack_operation = np.stack((encoded_sgRNA, encoded_targetDNA), axis=-1)

        # mismatch and indel one-hot encoding
        # num_mismatch = np.array(list(map(self.mismatch_one_hot_encoding, raw_data[''])))
        indel_class = np.array(list(map(self.indel_class_one_hot_encoding, raw_data['indel'])))

        # train data
        #sequence_data = stack_operation
        sequence_data = encoded_sgRNA
        indel_rate = raw_data['indel']
        indel_class = indel_class
        result_dict = {'seq' : sequence_data, 'indel_rate' : indel_rate, 'indel_class' : indel_class}
        
        # Read_csv_file에서 새로 구현할 dictionary를 자동적으로 추가해주기 위해서 아래 과정을 추가한다
        # Read_cnt, num_mismatch 같은 요소들을 자동적으로 추가해주기 위해서 아래 과정이 필요하다.
        pass_list = ['sgRNA', 'targetDNA', 'indel']
        for key, value in zip(list(raw_data.keys()), list(raw_data.values())):
            if key in pass_list:
                continue
            result_dict[key] = value
            
            
        # Data information
        data_information_list = [raw_data['sgRNA'], indel_rate, raw_data['read_cnt']]
        data_information = self.data_information(data_information_list)
        result_dict['info'] = data_information
        
        
        return result_dict



    def data_preprocessing_offtarget(self,_file_path, sgRNA_column, targetDNA_column, indel_column,
                                          sgRNA_have_NGG = True, delete_NGG = False,
                                          PAM_length = 3,
                                          mismatch_calc=False
                                          ):

        # {'sgRNA' : sgRNA, 'targetDNA' : targetDNA, 'indel' : indel}
        raw_data = self.Read_csv_file(file_path=_file_path,
                                     sgRNA_column=sgRNA_column, targetDNA_column=targetDNA_column,
                                     indel_column=indel_column
                                     )

        # Unify length
        sgRNA, targetDNA = self.Unify_length_processNGG(sgRNA= raw_data['sgRNA'],
                                                        targetDNA= raw_data['targetDNA'],
                                                        sgRNA_have_NGG=sgRNA_have_NGG,
                                                        delete_NGG=delete_NGG,
                                                        PAM_length=PAM_length
                                                        )

        # One-hote encoding
        encoded_sgRNA = np.array(list(map(self.seq_one_hot_encoding,raw_data['sgRNA'])))
        encoded_targetDNA = np.array(list(map(self.seq_one_hot_encoding,raw_data['targetDNA'])))

        # stack operation, make 2D
        stack_operation = np.stack((encoded_sgRNA, encoded_targetDNA), axis=-1)

        # mismatch and indel one-hot encoding
        #num_mismatch = np.array(list(map(self.mismatch_one_hot_encoding, raw_data[''])))
        indel_class = np.array(list(map(self.indel_class_one_hot_encoding, raw_data['indel'])))

        
        # Calculate mismatch position and number
        if mismatch_calc:
            mismatch = list(map(self.mismatch_position_N_number, encoded_sgRNA, encoded_targetDNA))
            mismatch_position = list(map(lambda x: x[0], mismatch))
            mismatch_number = list(map(lambda x: x[1], mismatch))
        else:
            mismatch_position = None
            mismatch_number = None
        

        # train data
        sequence_data = stack_operation
        indel_rate = raw_data['indel']
        indel_class = indel_class
        result_dict = {'seq' : sequence_data, 'indel_rate' : indel_rate, 'indel_class' : indel_class,
                      'mis_position' : mismatch_position, 'mis_number' : mismatch_number}
        
        
        # Read_csv_file에서 새로 구현할 dictionary를 자동적으로 추가해주기 위해서 아래 과정을 추가한다
        # Read_cnt, num_mismatch 같은 요소들을 자동적으로 추가해주기 위해서 아래 과정이 필요하다.
        pass_list = ['sgRNA', 'targetDNA', 'indel']
        for key, value in zip(list(raw_data.keys()), list(raw_data.values())):
            if key in pass_list:
                continue
            result_dict[key] = value
            
            
        # Data information
        data_information_list = [raw_data['sgRNA'], raw_data['targetDNA'],indel_rate,
                                 raw_data['read_cnt'], mismatch_number]
        data_information = self.data_information(data_information_list)
        result_dict['info'] = data_information
        
        
        return result_dict


    
    def split_data(self, dictionary_data, test_size = 0.2):
        data_keys = list(dictionary_data.keys())
        data_values = list(dictionary_data.values())
        data_values.reverse()
        data_mass = [x for x in data_values if x is not None]
        
        splited_data = train_test_split(*data_mass,
                                        test_size = test_size,
                                        random_state = self.RD_seed)
        
        train_data = {}
        test_data = {}
        for key in data_keys:
            if data_values.pop() is not None:
                test_data[key] = splited_data.pop()
                train_data[key] = splited_data.pop()
            else:
                test_data[key] = None
                train_data[key] = None
                
        return train_data, test_data
    
    
    def __call__(self, sgRNA_column, indel_column, targetDNA_column=None,
                 offtarget=False, split_data = 0.0,
                 sgRNA_have_NGG=True, delete_NGG = False,
                 PAM_length=3, mismatch_calc=False):
        
        # read csv file & preprocessing
        if offtarget is False:
            dict_data = self.data_preprocessing_ontarget(_file_path=self.train_data_file_name,
                                                         sgRNA_column=sgRNA_column,
                                                         indel_column = indel_column,
                                                         sgRNA_have_NGG=sgRNA_have_NGG, 
                                                         delete_NGG=delete_NGG,
                                                         PAM_length=PAM_length)
            if self.test_data_file_name is not None:
                test_dict_data = self.data_preprocessing_ontarget(_file_path=self.test_data_file_name,
                                                                  sgRNA_column=sgRNA_column, 
                                                                  indel_column = indel_column,
                                                                  sgRNA_have_NGG=sgRNA_have_NGG, 
                                                                  delete_NGG=delete_NGG,
                                                                  PAM_length=PAM_length)
            
        else:
            dict_data = self.data_preprocessing_offtarget(_file_path=self.train_data_file_name,
                                                          gRNA_column=sgRNA_column, 
                                                          argetDNA_column=targetDNA_column,
                                                          indel_column=indel_column,
                                                          sgRNA_have_NGG=sgRNA_have_NGG, 
                                                          delete_NGG=delete_NGG,
                                                          PAM_length=PAM_length,
                                                          mismatch_calc=mismatch_calc)
            if self.test_data_file_name is not None:
                test_dict_data = self.data_preprocessing_offtarget(_file_path=self.test_data_file_name,
                                                                   gRNA_column=sgRNA_column,
                                                                   indel_column = indel_column,
                                                                   sgRNA_have_NGG=sgRNA_have_NGG, 
                                                                   delete_NGG=delete_NGG,
                                                                   PAM_length=PAM_length)
        
            
        
        # split data
        if split_data < 0.01:
            result_dict = {'train' : dict_data, 'val' : None, 'total' : dict_data, 'test' : test_dict_data}
        else:
            train_data, test_data = self.split_data(dictionary_data=dict_data, test_size=split_data)
            result_dict = {'train' : train_data, 'val' : test_data, 'total' : dict_data, 'test' : test_dict_data}
            
        return result_dict
    
    
    
    
if __name__ == '__main__':
    pass