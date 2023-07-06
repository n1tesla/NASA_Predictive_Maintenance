from preprocessing.prepare_data import read_data,drop_columns


feature_columns=[]
useful_columns=[]

fd_001_train,fd_001_test,RUL_1=read_data()
print(f"fd_001_train: {fd_001_train}")
constant_columns=['ReqFanSpeed','ReqFanConvSpeed','FanInP','FanInTemp','TRA','BurnerFuelAirRatio','EngPreRatio']
fd_001_train=drop_columns(fd_001_train,constant_columns)
