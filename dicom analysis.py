from excel import excel
#py baseline

#15, 21 (h)
def main():
    excel().show_plot(15, bas=False, preprocess=True, interpolate=True, cycle=1, opt=False)
    #excel().show_plot(15, bas=True, preprocess=True, interpolate=True, cycle=1, opt=False)
    #excel().set_all_plot(save=False, bas=False, preprocess=True, interpolate=True, cycle=1, opt=False)
    #bas: bas=미투여, hyp=투여
    #preprocess: 전처리 사용/미사용
    #interpolate: 피크 보간법 사용/미사용 
    #cycle: 보간 횟수(적정은 2회일듯)
    #opt: 옵티컬플로우 사용 / 밝기값 사용
    
if __name__ == '__main__': main()
