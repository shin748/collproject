import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from interpolation import interp
from multiprocessing import Pool, Process
import multiprocessing
from functools import partial
import parmap

class analysis:
    def __init__(self):
        self.colors = ['lime', 'yellow', 'magenta', 'cyan', 'red', 'blue']

    #일반 클러스터링
    def clustering(self, num, xdata, ydata, isgmm, draw=True):
        cluster = []
        if isgmm == True: cluster = GaussianMixture(n_components=2, random_state=0)   
        else: cluster = KMeans(n_clusters=2, random_state=5)
        cluster.fit(ydata.reshape(-1,1))
        c_list = cluster.predict(ydata.reshape(-1,1)) #클러스터 리스트

        gsize = c_list.shape[0]
        prej=0; gcnt=0; group=[c_list[0]]
        
        for j in range(gsize):
            prev=c_list[prej]; curr=c_list[j]
            if prev != curr:
                if draw==True: plt.plot(xdata[prej:j+1], ydata[prej:j+1])
                if gcnt < num-1: group.append(curr); gcnt+=1 #gcnt < n_components-1
                prej=j
        if draw==True: plt.plot(xdata[prej:j+1], ydata[prej:j+1])

        return c_list, group #클러스터링 결과와 군집 등장순서(group[0~x])를 알려줌



    def get_ex(self, ex, start, end): #맨처음, 맨끝 극값을 반환
        exs=[]
        for value in ex:
            if str(value)[-1] == 'I': value=abs(complex(value))
            if value>=start and value<=end: exs.append(value)
        if len(exs) == 0: exs.append(end); exs.append(start) #극값이 없으면 개형을 따른다.
        return int(exs[0]), int(exs[-1])
    


    def get_gsection(self, c_list, group, offset=0): #그룹1->그룹2->그룹1 형태로 클러스터링됐다 가정, (그룹1->그룹2)와 (그룹2->그룹1)으로 변환되는 지점을 반환
        gsize = c_list.shape[0]; chk=0
        section1_end=0; section2_end=0
        #print(c_list)
        for i in range(1,gsize): 
            if c_list[i]==group[1]:
                if section1_end == 0: section1_end=i #구간1이 더 이상 아닌 위치
                chk=1
            elif c_list[i]!=group[1] and chk==1: #구간2 이탈
                section2_end=i; break #구간2 끝부분
        if section2_end==0: section2_end=gsize-1

        section1_end+=offset
        section2_end+=offset
        
        return int(section1_end), int(section2_end)

    def cal_TFC_CCFC(self, xdata, ydata, c_list, group, test=False): #TFC, CCFC 계산
        gsize = c_list.shape[0]; section1_end=0; section2_end=0
        section1_end, section2_end = self.get_gsection(c_list, group)

        if test == True:
            TFCS, TFCE, CCFCS, CCFCE = [], [], [], []
            #########################################################
            p = Pool(processes = multiprocessing.cpu_count())
            p1 = p.starmap_async(interp().data_interpolation, [[i, xdata[0:section1_end+1], xdata[0:section1_end+1], ydata[0:section1_end+1], True] for i in range(2,7)])
            print('구간1 complete..')
            p2 = p.starmap_async(interp().data_interpolation, [[i, xdata[section1_end:section2_end+1], xdata[section1_end:section2_end+1], ydata[section1_end:section2_end+1], True] for i in range(2,7)])
            print('구간2 complete..')
            p3 = p.starmap_async(interp().data_interpolation, [[i, xdata[section2_end:gsize], xdata[section2_end:gsize], ydata[section2_end:gsize], True] for i in range(2,7)])
            print('구간3 complete..')

            print('getting return values...')
            res1 = p1.get(); res2 = p2.get(); res3 = p3.get()
            print('getting return values has completed')
            for n_deg in res1:
                mat_i1, ypred1, ex = n_deg
                left_ex1, right_ex1 = self.get_ex(ex, 0, section1_end) #구간1 극값 = 극솟값
                TFCS.append(right_ex1)

            for n_deg in res2:
                mat_i2, ypred2, ex = n_deg
                left_ex2, right_ex2 = self.get_ex(ex, section1_end, section2_end) #구간2 극값 = 극댓값
                TFCE.append(left_ex2); CCFCS.append(right_ex2)

            for n_deg in res3:
                mat_i3, ypred3, ex = n_deg
                left_ex3, right_ex3 = self.get_ex(ex, section2_end, gsize)
                CCFCE.append(left_ex3)

            p.close(); p.join()
            '''
            for a in range(9,11):
                ##구간1 피팅####################
                #mat_i1, ypred1, ex = interp().data_interpolation(a, xdata[0:section1_end+1], xdata[0:section1_end+1], ydata[0:section1_end+1], True)
                #left_ex1, right_ex1 = self.get_ex(ex, 0, section1_end) #구간1 극값 = 극솟값
                TFCS.append(right_ex1)
            print('구간1 complete..')

            for b in range(9, 11):
            ######구간2 피팅####################
                mat_i2, ypred2, ex = interp().data_interpolation(b, xdata[section1_end:section2_end+1], xdata[section1_end:section2_end+1], ydata[section1_end:section2_end+1], True)
                left_ex2, right_ex2 = self.get_ex(ex, section1_end, section2_end) #구간2 극값 = 극댓값
                TFCE.append(left_ex2); CCFCS.append(right_ex2)
            print('구간2 complete..')

            for c in range(9, 11):
                #######구간3 피팅####################
                mat_i3, ypred3, ex = interp().data_interpolation(c,xdata[section2_end:gsize], xdata[section2_end:gsize], ydata[section2_end:gsize], True)
                left_ex3, right_ex3 = self.get_ex(ex, section2_end, gsize)
                CCFCE.append(left_ex3)
            print('구간3 complete..')
            '''
        else:
            mat_i1, ypred1, ex = interp().data_interpolation(4, xdata[0:section1_end+1], xdata[0:section1_end+1], ydata[0:section1_end+1])
            left_ex1, right_ex1 = self.get_ex(ex, 0, section1_end) #구간1 극값 = 극솟값
            TFCS = right_ex1

            mat_i2, ypred2, ex = interp().data_interpolation(4, xdata[section1_end:section2_end+1], xdata[section1_end:section2_end+1], ydata[section1_end:section2_end+1])
            left_ex2, right_ex2 = self.get_ex(ex, section1_end, section2_end) #구간2 극값 = 극댓값
            TFCE, CCFCS = left_ex2, right_ex2

            mat_i3, ypred3, ex = interp().data_interpolation(4,xdata[section2_end:gsize], xdata[section2_end:gsize], ydata[section2_end:gsize])
            left_ex3, right_ex3 = self.get_ex(ex, section2_end, gsize)
            CCFCE = left_ex3

            plt.plot(mat_i1, ypred1, color='magenta')
            plt.plot(mat_i2, ypred2, color='cyan')
            plt.plot(mat_i3, ypred3, color='red') 

        return TFCS, TFCE, CCFCS, CCFCE

    ####################################################
    def get_TFC_CCFC(self, xdata, ydata, isgmm, test=False): #TFC, CCFC 출력부
        c_list, group = self.clustering(2, xdata, ydata, isgmm, draw=False)
        return self.cal_TFC_CCFC(xdata, ydata, c_list, group, test)
    ####################################################
