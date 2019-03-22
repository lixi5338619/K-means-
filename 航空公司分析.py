import pandas as pd                 #导入数据处理库pandas        安装方法pip install pandas
import numpy as np                  #导入科学计算库numpy         安装方法pip install numpy
from sklearn.cluster import KMeans  #导入KMeans非监督聚类算法    安装方法太麻烦百度吧
import matplotlib.pyplot as plt     #导入绘图库matplotlib        安装方法pip install matplotlib

# 步骤:数据清洗。 属性挑选。数据标准化。数据分析

'''
1.  RFM模型介绍
R( Recency)指的是最近一次消费时间与截止时间的间隔
F(Frequency)指某段时回内所消费的次数
M( Monetary)指顾客在某段时间内所消费的金额
'''

'''
2.  航空公司客户价值分析的 LRFMC模型 
以2014年3月3日为结束时间，选取宽度为两年的时间段作为分析观测窗口
L(LOAD TIME-FFP DATE) : 会员入会时间距某时间的月数
R(RELAST TO END) :      最近一次消费距当前的月数     
F(FLIGHT COUNT) :       某段时间消费次数    
M(SEG_KM_SUM) :         某段时间飞行里程数
C(AVG_DISCOUNT) :       所对应的折扣系数的平均值
'''

'''处理数据缺失值与异常值'''
data = pd.read_csv("air_data.csv", encoding="ansi")              #使用pandas的read_csv读取csv文件
print(data.shape)                                                  #查看当前数据的结构
##要求1、丢弃票价为空的记录。                                       #那就取了票价不为空的数据
data = data[data["SUM_YR_1"].notnull() & data["SUM_YR_2"].notnull()]  #两个价格分别为YR_1和YR_2
# print(data.shape)                                                #查看当前数据的结构
##要求2、丢弃票价为0、平均折扣率不为0、总飞行千米数大于0的记录
        ##那就把票价不为0，飞行公里==0，平均折扣==0的找出来。
doc1 = (data["SUM_YR_1"] !=0) |  (data["SUM_YR_2"] !=0)          #票价不为0的, '|'是或。 '&'是与。
doc2 = (data["SEG_KM_SUM"] == 0) & (data["avg_discount"] == 0)   #飞行公里==0,平均折扣==0的
data = data[doc1 | doc2]
# print(data.shape)


'''构建航空客户价值分析关键特征'''
## 使用LRFMC模型                        我们要拿到需要的属性。上面写着有属性要求。
## FFP_DATE、  LOAD_TIME、  FLIGHT_COUNT、   AVG_DISCOUNT、   SEG_KM_SUM、  LAST_TO_END。

##1、会员入会时间距观测窗口结束的月数L = 观测窗口的结束时间LOAD_TIME -  入会时间(单位月) FFP_DATE
data["LOAD_TIME"] = pd.to_datetime(data["LOAD_TIME"])               #要按pd.datetime格式转换时间格式
data["FFP_DATE"] = pd.to_datetime(data["FFP_DATE"])
data["会员入会时间"] = (((data["LOAD_TIME"] - data["FFP_DATE"])))   # L - F = 入会的总天数,然后转成月
mon = []                                                              #设置一个空列表
for i in data["会员入会时间"]:                                       #遍历下时间
    months = int(i.days/30)                                           #计算出对应的月份
    mon.append(months)                                                #加入空列表
data["会员入会月份"] = np.array(mon)                                  #在data里面重新插入一列月份
# print(data["会员入会月份"])

data["LAST_TO_END"] = data["LAST_TO_END"]/30                       #最后飞行时间。算的是月数，直接除以30了。

    ##FLIGHT_COUNT:消费次数  LAST_TO_END最后一次时间  SEG_KM_SUM:总飞行公里数  avg_discount平均折扣率
my_data = data[["会员入会月份", "LAST_TO_END", "FLIGHT_COUNT" , "SEG_KM_SUM", "avg_discount"]]
print(my_data)


'''数据标准化'''
# 数据标准化处理              # 将输入序列进行小数定标标准化
def decimal_clean(arr):                                        # :param arr:输入的待优化的序列
    k = np.ceil(np.log10(np.max(np.abs(arr))))                 # 通过移动数据的小数点位置来进行标准化
                             # np.ceil:向上取整   np.abs:绝对值   np.max:最大值   np.log10:求对数，若N=a**x,x=loga**N
    return arr / 10 ** k                                      # :return:标准化后的序列
                             # 利用pandas中的concat连接函数，iloc切片函数，先切片再标准化然后合并起来。
xyzqw = pd.concat([decimal_clean(my_data.iloc[:, 0]), decimal_clean(my_data.iloc[:, 1]),
                   decimal_clean(my_data.iloc[:, 2]),decimal_clean(my_data.iloc[:, 3]),
                   decimal_clean(my_data.iloc[:, 4])], axis=1, join="outer")
# print(xyzqw)
# print(type(xyzqw))

'''K-means聚类算法'''
x = xyzqw[['会员入会月份','LAST_TO_END','FLIGHT_COUNT','SEG_KM_SUM','avg_discount']]
kms = KMeans(n_clusters=5)                            #Kmeans方法 导入5个聚类中心
y = kms.fit_predict(x)                                #计算聚类中心并预测每个样本的聚类指数
# print(y)                                            # y是numpy数组


'''绘制雷达图'''
def drow():
    plt.figure(figsize=(10,6))
    plt.rcParams['font.sans-serif'] = ['KaiTi']
    # plt.style.use('ggplot')                          #ggplot样式
    tu  = plt.subplot(321,polar=True)
    tu1 = plt.subplot(322,polar=True)
    tu2 = plt.subplot(323,polar=True)
    tu3 = plt.subplot(324,polar=True)
    tu4 = plt.subplot(325,polar=True)
    labels = np.array(['会员入会月份','LAST_TO_END','FLIGHT_COUNT','SEG_KM_SUM','avg_discount'])    #设置标签
    theta = np.linspace(0,2*np.pi,5,endpoint=False)  #生成角度值，从0开始到2π,生成5份，终端点为False
    theta = np.concatenate((theta,[theta[0]]))       #闭合:加上之后首尾相接，保持第一个值和最后一个值一样
    data = [x["会员入会月份"][y==0],x["LAST_TO_END"][y==0],x["FLIGHT_COUNT"][y==0],x["SEG_KM_SUM"][y==0],x["avg_discount"][y==0]]               #
    data1 = [x["会员入会月份"][y==1],x["LAST_TO_END"][y==1],x["FLIGHT_COUNT"][y==1],x["SEG_KM_SUM"][y==1],x["avg_discount"][y==1]]               #
    data2 = [x["会员入会月份"][y==2],x["LAST_TO_END"][y==2],x["FLIGHT_COUNT"][y==2],x["SEG_KM_SUM"][y==2],x["avg_discount"][y==2]]               #
    data3 = [x["会员入会月份"][y==3],x["LAST_TO_END"][y==3],x["FLIGHT_COUNT"][y==3],x["SEG_KM_SUM"][y==3],x["avg_discount"][y==3]]               #
    data4 = [x["会员入会月份"][y==4],x["LAST_TO_END"][y==4],x["FLIGHT_COUNT"][y==4],x["SEG_KM_SUM"][y==4],x["avg_discount"][y==4]]               #
    data  = np.concatenate((data, [data[0]]))          #闭合:保持第一个值和最后一个值相等
    data1 = np.concatenate((data1,[data1[0]]))
    data2 = np.concatenate((data2,[data2[0]]))
    data3 = np.concatenate((data3,[data3[0]]))
    data4 = np.concatenate((data4,[data4[0]]))
    tu .plot(theta,data, marker =(5,1))
    tu1.plot(theta,data1,marker = 'x')
    tu2.plot(theta,data2,marker ="o")
    tu3.plot(theta,data3,marker ="o")
    tu4.plot(theta,data4,marker ="o")
    tu.set_xticklabels(labels)
    tu1.set_xticklabels(labels)
    tu2.set_xticklabels(labels)
    tu3.set_xticklabels(labels)
    tu4.set_xticklabels(labels)
    tu.set_xticklabels(labels)

    plt.xticks(theta,labels)
    plt.title("卒")
    plt.show()

drow()