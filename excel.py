import pydicom
import matplotlib.pyplot as plt
import pandas as pd
from optflow import dicom_optflow
from intense_sum import dicom_intense
from scipy import interpolate

#엑셀파일로 경로를 파악하기 위한 클래스 (+벡터수 저장/확인용 함수)
class excel:
    df = pd.read_excel('C:/Users/USERH/Desktop/ESCARGOT/ESCARGOT.xlsx')

    def __init__(self):
        self.drop_df = self.df.drop([0,2,4,9,11,22,34,38,43,49], axis=0, inplace=False)
        self.df.drop(['비고'], axis=1, inplace=True) #불필요한 열 제거
        self.df.dropna(axis=1); self.df.fillna(0, inplace=True) #혹시나 불필요한 열 더 제거
        self.idx = self.df.index
        self.didx = self.drop_df.index #드랍한 인덱스행 파악

    def load_excel(self, pat_num=-1): #엑셀로부터 파일의 경로를 추출해온다.
        paths = []
        if pat_num == -1: #모든 경로를 추출하고 싶을 때
            for i in self.didx:
                num = self.drop_df['Img_bas_n'][i].astype(int)
                paths.append('C:/Users/USERH/Desktop/ESCARGOT/' + f'{i+1}/{i+1}({num}).dcm')
        else: #특정 경로만 추출하고 싶을 때
            num = self.df['Img_bas_n'][self.idx[pat_num]].astype(int)
            paths = 'C:/Users/USERH/Desktop/ESCARGOT/' + f'{pat_num+1}/{pat_num+1}({num}).dcm'

        return paths
    #####################################################
    def save_all_plot(self, preprocess=False,  interpolate=False, opt=False):
        paths = self.load_excel()
        len = len(paths)

        for i in range(len):
            dicom_data = pydicom.dcmread(paths[i]) #force=True 인자를 넣으면 인식안되는 dicom 읽을수 있음
            dicom_img = dicom_data.pixel_array #dicom이미지 호출 (dicom_img[n] = n번째 프레임)
            all_frame = dicom_img.shape[0] #dicom_img.shape = [389, 512, 512] -> 프레임 389장, 512x512

            ##옵티컬플로우 or 밝기값
            if opt == True: mat_i, mat_x = dicom_optflow().do_optflow(dicom_img, all_frame, preprocess)
            else: mat_i, mat_x = dicom_intense().do_intense(dicom_img, all_frame, preprocess)

            ##보간사용/미사용
            k = self.didx[i]
            plt.plot(mat_i, mat_x, color='black')
            if interpolate == False: plt.plot(mat_i, mat_x, color='black')
            else: self.peak_interpolation(mat_i, mat_x)

            ##엑셀속 frame count 표시
            plt.axvspan(self.drop_df['TFC_start_b'][k].astype(int), self.drop_df['TFC_end_b'][k].astype(int), facecolor='red')
            plt.axvspan(self.drop_df['CCFC_start_b'][k].astype(int), self.drop_df['CCFC_end_b'][k].astype(int), facecolor='blue')
            plt.savefig('C:/Users/USERH/Desktop/all_plot/'+f'{k+1}.png')
            plt.clf()
    ####################################################
    def show_plot(self, pat_num, preprocess=False, interpolate=False, opt=False):  #특정파일에 대한 하강벡터수 확인
        pat_num-=1
        path = self.load_excel(pat_num)
        dicom_data = pydicom.dcmread(path) #force=True 인자를 넣으면 인식안되는 dicom 읽을수 있음
        dicom_img = dicom_data.pixel_array #dicom이미지 호출 (dicom_img[n] = n번째 프레임)
        all_frame = dicom_img.shape[0] #dicom_img.shape = [389, 512, 512] -> 프레임 389장, 512x512

        ##옵티컬플로우 or 밝기값
        if opt == True: mat_i, mat_x = dicom_optflow().do_optflow(dicom_img, all_frame, preprocess)
        else: mat_i, mat_x = dicom_intense().do_intense(dicom_img, all_frame, preprocess)
        
        ##보간사용/미사용
        k = pat_num
        plt.plot(mat_i, mat_x, color='black')
        if interpolate == False: plt.plot(mat_i, mat_x, color='black')
        else: self.peak_interpolation(mat_i, mat_x)

        ##엑셀속 frame count 표시
        plt.axvspan(self.df['TFC_start_b'][k].astype(int), self.df['TFC_end_b'][k].astype(int), facecolor='red')
        plt.axvspan(self.df['CCFC_start_b'][k].astype(int), self.df['CCFC_end_b'][k].astype(int), facecolor='blue')
        plt.show()
    ########################################################
    def peak_interpolation(mat_i, mat_x):
        prev_x=mat_x[0]
        incr=1
        peak_i=[]; peak_x=[]
        ###########  피크탐색  ###################
        for i in mat_i:
            i-=1         
            if prev_x >= mat_x[i]: #감소중
                if incr==1:
                    peak_i.append(i+1); peak_x.append(prev_x) #증가하다 감소시작 = 피크위치만 저장
                    incr=0 #증가 종료를 알림
                prev_x = mat_x[i]

            elif prev_x < mat_x[i]:
                incr=1 #증가하면 증가시작을 알림
                prev_x = mat_x[i]
        
        ###########  보간진행  ###################
        f = interpolate.interp1d(peak_i, peak_x, kind='linear')
        if mat_i[-1] > peak_i[-1]: mat_i = mat_i[0:peak_i[-1]]
        f_y = f(mat_i)
        return plt.plot(mat_i, f_y, '-', color='lime')
