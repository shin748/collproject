import pydicom
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from optflow import dicom_optflow
from intense_sum import dicom_intense
from scipy import interpolate
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score, cross_validate
from interpolation import interp
from data_analysis import analysis
from sklearn.metrics import mean_squared_error
from math import sqrt


#엑셀파일로 경로를 파악하기 위한 클래스 (+벡터수 저장/확인용 함수)
class excel:
    df = pd.read_excel('ESCARGOT/ESCARGOT.xlsx')

    def __init__(self):
        self.drop_df = self.df.drop([0,2,4,9,11,12,15,22,27,29,34,36,38,41,43,49], axis=0, inplace=False)
        #self.drop_df = self.df.drop([0,2,3,4,5,6,7,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49], axis=0, inplace=False)

        #self.drop_df = self.df.drop([0,2,4,9,11,12,22,34,38,41,43,49], axis=0, inplace=False)
        self.df.drop(['비고'], axis=1, inplace=True) #불필요한 열 제거
        self.df.dropna(axis=1); self.df.fillna(0, inplace=True) #혹시나 불필요한 열 더 제거
        self.idx = self.df.index
        self.didx = self.drop_df.index #드랍하고 남은 인덱스행 파악

    def load_excel(self, pat_num=-1, bas=True): #엑셀로부터 파일의 경로를 추출해온다.
        paths = []
        if pat_num == -1: #모든 경로를 추출하고 싶을 때
            for i in self.didx:
                if bas==True: num = self.drop_df['Img_bas_n'][i].astype(int)
                else: num = self.drop_df['Img_hyp_n'][i].astype(int)
                paths.append('ESCARGOT/' + f'{i+1}/{i+1}({num}).dcm')
        else: #특정 경로만 추출하고 싶을 때
            if bas==True: num = self.df['Img_bas_n'][self.idx[pat_num]].astype(int)
            else: num = self.df['Img_hyp_h'][self.idx[pat_num]].astype(int)
            paths = 'ESCARGOT/' + f'{pat_num+1}/{pat_num+1}({num}).dcm'

        return paths
    #####################################################

    def set_all_plot(self, save=False, bas=True, preprocess=False, interpolate=False, opt=False, test=False):
        paths = self.load_excel()
        plen = len(paths)
        results=[]

        deglen = 5
        
        for i in range(deglen): #원래는 5가 아니라 따로 변수로 지정해야함
            results.append([]) # results = [ [2차피팅], [3차피팅], [4차피팅], [5차피팅], ...]
            results[i].append([]); results[i].append([]); results[i].append([]); results[i].append([]) #[n차피팅] 안에 [ [TFCS들], [TFCE들], ...]

        rtfcs=[]; rtfce=[]; rccfcs=[]; rccfce=[]
        for i in range(plen): #모든 파일들에 대해
            dicom_data = pydicom.dcmread(paths[i]) #force=True 인자를 넣으면 인식안되는 dicom 읽을수 있음
            dicom_img = dicom_data.pixel_array #dicom이미지 호출 (dicom_img[n] = n번째 프레임)
            k = self.didx[i]
            print(f'[{k+1}]파일')

            if bas==True: dicom_img = np.load('DENOISED/'+str(k+1)+'(b).npy')
            else: dicom_img = np.load('DENOISED/'+str(k+1)+'(h).npy')

            all_frame = dicom_img.shape[0] #dicom_img.shape = [389, 512, 512] -> 프레임 389장, 512x512

            ##옵티컬플로우 or 밝기값
            if opt == True: mat_i, mat_x = dicom_optflow().do_optflow(dicom_img, all_frame, preprocess)
            else: mat_i, mat_x = dicom_intense().do_intense(dicom_img, all_frame, preprocess)
            #print('Intensity 게산완료')
            #dicom_img = dicom_intense().proc_data(dicom_img)

            ##보간사용/미사용
            plt.plot(mat_i, mat_x, color='black')

            if interpolate == False: plt.plot(mat_i, mat_x, color='lime')
            else:
                interp_i, mat_x1 = interp().peak_interpolation(mat_i, mat_x) #1차 보간
                #print('프레임 계산 시작')
                TFCS, TFCE, CCFCS, CCFCE = analysis().get_TFC_CCFC(np.array(interp_i), np.array(mat_x1), isgmm=False, test=test) #클러스터링(n=2)기반 극대극소를 발견하는 TFC
                #print('프레임 계산 종료')

            ### 파일개수만큼 실제 프레임값을 추가               
            series = ''
            if bas==True: series = 'b'
            else: series='h'
            ####실제 TFC, CCFC 저장
            tfcs_real, tfce_real = self.drop_df['TFC_start_'+series][k].astype(int), self.drop_df['TFC_end_'+series][k].astype(int)
            ccfcs_real, ccfce_real = self.drop_df['CCFC_start_'+series][k].astype(int), self.drop_df['CCFC_end_'+series][k].astype(int)
            rtfcs.append(tfcs_real); rtfce.append(tfce_real); rccfcs.append(ccfcs_real); rccfce.append(ccfce_real)
            
            print(f'TFC: [{tfcs_real}, {tfce_real}], predict: [{TFCS}, {TFCE}]')
            print(f'CCFC: [{ccfcs_real}, {ccfce_real}], predict: [{CCFCS}, {CCFCE}]')
            #1개의 파일에 대한 [[TFCS], [TFCE], [CCFCS], [CCFCE]]가 반환되므로.. n차항별로 분리해야한다.
            #[ [2차TFCS, 3차TFCS ...], [2차TFCE, 3차TFCE, ...], ...] 이기 때문에.
            
            for j in range(deglen): 
                results[j][0].append(TFCS[j]) # [ [j차 피팅], [j+1차 피팅], ..] -> [j차피팅]안에는 [ [j차 TFCS], [j차 TFCE], [j차 CCFCS], [j차 CCFCE] ] 처럼 생김
                results[j][1].append(TFCE[j])
                results[j][2].append(CCFCS[j])
                results[j][3].append(CCFCE[j])
            #####################################

            ##엑셀속 frame count 표시
            #plt.axvspan(tfcs_real, tfce_real, facecolor='red')
            #plt.axvspan(ccfcs_real, ccfce_real, facecolor='blue')
            if save == True: plt.savefig('all_plot/'+f'{k+1}.png')
            plt.clf()
            #print('파일작업 종료')

        ###오차 평가
        print('RMSE 평가 시작')
        min_tfcs, min_tfce, min_ccfcs, min_ccfce = [1000,0], [1000,0], [1000,0], [1000,0]

        rlen = len(results)
        for i in range(rlen):
            res = results[i] #res = [i차 피팅] = [ [i차 TFCS], [i차 TFCE], [i차 CCFCS], [i차 CCFCE] ]
            tfcs_rmse = sqrt(mean_squared_error(rtfcs, res[0])) #res[0] = n차 피팅 TFCS
            if min_tfcs[0] > tfcs_rmse:
                min_tfcs[0] = tfcs_rmse; min_tfcs[1] = i+2 #2차함수부터 시작하니까.

            tfce_rmse = sqrt(mean_squared_error(rtfce, res[1]))
            if min_tfce[0] > tfce_rmse:
                min_tfce[0] = tfce_rmse; min_tfce[1] = i+2

            ccfcs_rmse = sqrt(mean_squared_error(rccfcs, res[2]))
            if min_ccfcs[0] > ccfcs_rmse:
                min_ccfcs[0] = ccfcs_rmse; min_ccfcs[1] = i+2

            ccfce_rmse = sqrt(mean_squared_error(rccfce, res[3]))
            if min_ccfce[0] > ccfce_rmse:
                min_ccfce[0] = ccfce_rmse; min_ccfce[1] = i+2
            
        print(f'TFCS RMSE = {min_tfcs[0]} ({min_tfcs[1]}차함수)')
        print(f'TFCE RMSE = {min_tfce[0]} ({min_tfce[1]}차함수)')
        print(f'CCFCS RMSE = {min_ccfcs[0]} ({min_ccfcs[1]}차함수)')
        print(f'CCFCE RMSE = {min_ccfce[0]} ({min_ccfce[1]}차함수)')

        print('----------RMSE 평가 종료')

        #################
    ####################################################




    def show_plot(self, pat_num, bas=True, preprocess=False, interpolate=False, opt=False):  #특정파일에 대한 하강벡터수 확인
        pat_num-=1
        path = self.load_excel(pat_num)
        dicom_data = pydicom.dcmread(path) #force=True 인자를 넣으면 인식안되는 dicom 읽을수 있음
        dicom_img = dicom_data.pixel_array #dicom이미지 호출 (dicom_img[n] = n번째 프레임)
        
        if bas==True: dicom_img = np.load('DENOISED/'+str(pat_num+1)+'(b).npy')
        else: dicom_img = np.load('DENOISED/'+str(pat_num+1)+'(h).npy')
        all_frame = dicom_img.shape[0] #dicom_img.shape = [389, 512, 512] -> 프레임 389장, 512x512
        
        ##옵티컬플로우 or 밝기값
        if opt == True: mat_i, mat_x = dicom_optflow().do_optflow(dicom_img, all_frame, preprocess)
        else: mat_i, mat_x = dicom_intense().do_intense(dicom_img, all_frame, preprocess)

        
        ##보간사용/미사용
        k = pat_num; interp_i=[]
        plt.plot(mat_i, mat_x, color='black')
        if interpolate == False: plt.plot(mat_i, mat_x, color='black')
        else:
            interp_i, mat_x1 = interp().peak_interpolation(mat_i, mat_x) #1차 보간
            TFCS, TFCE, CCFCS, CCFCE = analysis().get_TFC_CCFC(np.array(interp_i), np.array(mat_x1), isgmm=False) #클러스터링(n=2)기반 극대극소를 발견하는 TFC

        series = ''
        if bas==True: series = 'b'
        else: series='h'
        rtfcs, rtfce = self.df['TFC_start_'+series][k].astype(int), self.df['TFC_end_'+series][k].astype(int)
        rccfcs, rccfce = self.df['CCFC_start_'+series][k].astype(int), self.df['CCFC_end_'+series][k].astype(int)

        print(f'TFC: [{rtfcs}, {rtfce}], predict: [{TFCS}, {TFCE}]')
        print(f'CCFC: [{rccfcs}, {rccfce}], predict: [{CCFCS}, {CCFCE}]')
        plt.title(f'[{int(k+1)}] TFC: [{rtfcs}, {rtfce}], predict: [{TFCS}, {TFCE}]\n CCFC: [{rccfcs}, {rccfce}], predict: [{CCFCS}, {CCFCE}]', fontsize=15)

        ##엑셀속 frame count 표시
        plt.axvspan(self.df['TFC_start_'+series][k].astype(int), self.df['TFC_end_'+series][k].astype(int), facecolor='grey', alpha=0.4, zorder=1)
        plt.axvspan(self.df['CCFC_start_'+series][k].astype(int), self.df['CCFC_end_'+series][k].astype(int), facecolor='grey', alpha=0.4, zorder=1)
        plt.show()
    #######################################################
