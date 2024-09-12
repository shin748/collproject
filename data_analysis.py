import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold, cross_val_score, cross_validate
from interpolation import interp

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


    def get_ex(self, ex, start, end, degree): #함수에 출력된 극값중 가장 작은 값을 반환
        #print(f'ex={ex}')
        a=0; b=0
 
        if degree==2:
            a = ex[0]
            if str(a)[-1] == 'I': a=abs(complex(ex[0]))

            if a<start or a<0 or a>end: 
                print('there is no extream point'); a=0
            return a

        if degree==3:
            a=ex[0]; b=ex[1]
            swap=1; find_ex=0
            if str(a)[-1] == 'I': a=abs(complex(ex[0]))
            if str(b)[-1] == 'I': b=abs(complex(ex[1]))

            if a<start or a<0 or a>end:
                find_ex=b; swap=0
            if b<start or b<0 or b>end:
                find_ex=a; swap=0

            if swap==1:
                if a<b: find_ex = b
                else: find_ex = a
            
            return find_ex
    


    def get_gsection(self, c_list, group, offset=0): #그룹1->그룹2->그룹1 형태로 클러스터링됐다 가정, (그룹1->그룹2)와 (그룹2->그룹1)으로 변환되는 지점을 반환
        gsize = c_list.shape[0]; chk=0
        section1_end=0; section2_end=0
        #print(c_list)
        for i in range(1,gsize): 
            if c_list[i]==group[1]:
                if section1_end == 0: section1_end=i #구간1 끝부분
                chk=1
            elif c_list[i]!=group[1] and chk==1: #구간2 이탈
                section2_end=i; break #구간2 끝부분
        if section2_end==0: section2_end=gsize-1

        section1_end+=offset
        section2_end+=offset
        
        return section1_end, section2_end

    def div3_drawing(self, xdata, ydata, g1, g2, cs=0): #3등분 전용 그리기
        plt.plot(xdata[:g1+1], ydata[:g1+1], color=self.colors[cs])
        plt.plot(xdata[g1+1:g2+1], ydata[g1+1:g2+1], color=self.colors[cs+1])
        plt.plot(xdata[g2+1:], ydata[g2+1:], color=self.colors[cs])
        
            

    def localminmax(self, xdata, ydata, c_list, group): #TFC, CCFC 계산
        gsize = c_list.shape[0]; chk=0; section1_end=0; section2_end=0

        ############## 전처리(?) 시작 ###############################
        ###구간1,2 파악(클러스터링 수정 전, 구간1만 파악하는 용도)######################
        section1_end, section2_end = self.get_gsection(c_list, group)

        #####구간2 to 3 통째로 피팅####
        mat_i2, ypred, ex = interp().data_interpolation(3, np.array(range(section1_end, gsize)), np.array(range(section1_end, gsize)), ydata[section1_end:gsize], True)
        plt.plot(mat_i2, ypred, color='orange') 

        
        ####구간[2,3]에 대해 피팅한 그래프를 클러스터링 (임시적임. -> 기존 peak_interpolation의 구간1,2,3을 좀 더 명확하게 분리하기 위함)
        c_list2, group2 = self.clustering(2, mat_i2, ypred, False, False)
        section1_end2, section2_end2 = self.get_gsection(c_list2, group2, section1_end)

        ######구간[2,3] 클러스터링 결과 나눠진 구간점을 이용하여
        ######구간 [1, 2 3]으로 좀 더 명확히 분리함
        for i in range(gsize):
            if i>=section1_end and i<=section1_end2:
                if c_list[i] == group[0]:
                    c_list[i] = group[1]; section2_end = i

            if i>section1_end2:
                if c_list[i] == group[1]: c_list[i] = group[0]
        
        self.div3_drawing(xdata, ydata, section1_end, section2_end) #그냥 지저분해서 함수로 만들어 그림
        ############# 전처리 종료 ################################


        ######### 이제 제대로 peak_interp 결과가 구간[1,2,3]으로 분리됐으므로 이를 기반으로 분석 시작 #######################


        ####구간1 피팅##############
        mat_i2, ypred, ex = interp().data_interpolation(2, np.array(range(0,section1_end+1)), np.array(range(0,section1_end+1)), ydata[0:section1_end+1], True)
        plt.plot(mat_i2, ypred, color='orange')
        
        ex1 = self.get_ex(ex, 0, section1_end, 2)
        #print(f'구간1 극소점 = {ex1}')
        
        ####구간2 피팅##############
        mat_i2, ypred, ex = interp().data_interpolation(2, np.array(range(section1_end, section2_end+1)), np.array(range(section1_end, section2_end+1)), ydata[section1_end:section2_end+1], True)
        plt.plot(mat_i2, ypred, color='orange')
        max_ex = int(self.get_ex(ex, section1_end, section2_end, 2))

        ###구간[2,3] 피팅#######################
        mat_i2, ypred, ex = interp().data_interpolation(3, np.array(range(section1_end, gsize)), np.array(range(section1_end, gsize)), ydata[section1_end:gsize], True)
        plt.plot(mat_i2, ypred, color='orange') 
        # 특이사항: 눈에 보이는 것과 다르게 극점이 제대로 안나온다.
        
        ###따라서 구간[2,3] 피팅함수의 극소점을 수작업으로 찾기위해 클러스터링########
        start = section1_end2-section1_end
        c_list_min, group_min = self.clustering(2, mat_i2[start:len(mat_i2)], ypred[start:len(ypred)], True)
        minsection1_end, minsection2_end = self.get_gsection(c_list_min, group_min, section1_end2)

        self.div3_drawing(mat_i2[start:len(mat_i2)], ypred[start:len(ypred)], minsection1_end, minsection2_end, cs=2) #왜 색칠안됨??
        min_ex = int((minsection1_end + minsection2_end)/2) #극소점은 대략 작은 데이터로 분리된 구간의 중앙값일 것이다

        #print(f'극대={max_ex}({ypred[max_ex-section1_end]}), 극소={min_ex}({ypred[min_ex-section1_end]})')
        plt.scatter(min_ex, ypred[min_ex-section1_end], s=100, color='crimson', zorder=5)

        ################################################################################

        #speed = (ypred[max_ex-section1_end] - ypred[min_ex-section1_end])/(min_ex - max_ex)
        #print(f'speed = {speed}')

        ##### 프레임 계산: 아직 임시값이므로 정확한 값 아님 #################################################
        
        #TFCS = int((section1_end/2) + (ex2-(section1_end/2))*0.2)
        #TFCE = int((section1_end/2) + (ex2-(section1_end/2))*0.6)

        #CCFCS = int(ex2 + ( ((gsize+section2_end)/2)-ex2)*0.05)
        #CCFCE = int(ex2 + ( ((gsize+section2_end)/2)-ex2)*0.8)
        #CCFCE = section2_end 
        
        mindex = max_ex-section1_end
        nindex = min_ex-section1_end
        upspeed = (ypred[mindex] - ydata[0]) / (max_ex-0) #상승방향으로의 프레임당 intensity 변화수
        downspeed = (ypred[mindex] - ypred[nindex]) / (min_ex - max_ex) #하강방향으로의 프레임당 intensity 변화수
        #speed가 높은 경우: 프레임 거리가 짧되, intensity차이는 클수록. (<-> 프레임 거리가 길되 intensity차이는 작으면 낮은값)
        print(f'upspeed = {upspeed}, downspeed = {downspeed}')
        
        TFCS = int((ex1+section1_end)/2)
        TFCE = int((section1_end+max_ex)/2)
        CCFCS = int((max_ex+section1_end2)/2)
        CCFCE = int((section1_end2+min_ex)/2)
        return TFCS, TFCE, CCFCS, CCFCE

    ####################################################
    def get_TFC_CCFC(self, xdata, ydata, isgmm): #TFC, CCFC 출력부
        c_list, group = self.clustering(2, xdata, ydata, isgmm)
        return self.localminmax(xdata, ydata, c_list, group)
    ####################################################
