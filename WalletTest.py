from __future__ import unicode_literals

import sys
import pandas as pd
import time
import numpy as np


global df1
from urllib.parse import urlencode
from urllib.request import urlopen


#This Python code requires Python 3 or higher.
#There is only one global variable df1, which contains y and its prediction.
#This code call an API that makes the prediction.  The API takes a string and return a string.
#The input string to the API is a list of observations seperated by ;.  The data points in the
#observations are seperated by commas.
#The model read a CSV file, and assumes the file to have all the 304 variables, and
#to have missing values represented as NA.
#The predicted value is saved in the following file walletPredictionOutput.csv
#To Run this code in your terminal type
# python3 WalletTest.py "fileNameWithPathAndExtension"



def wolfram_cloud_call(**args):
    result = urlopen('http://www.wolframcloud.com/objects/5c737bf7-7339-4641-860d-5b89e89f677b', urlencode(args).encode('utf-8'))
    return result.read()

def call(s):
    textresult = wolfram_cloud_call(s=s)
    return textresult
#Calculate RMSE: Sqrt of the mean of the square of the differences between the actual and predicted values
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

#Calculate the Mean Absolute Percentage Error
def mape(predictions, targets):
    return 100* np.absolute( (targets- predictions)/targets).mean()
#Build a string to pass to
def buildTheInput(lNumb):
    st = ''
    for i in lNumb:
        if pd.isnull(i):
            t = 'Missing'
        else:
            t = int(i)
        st = st + str(t) + ','
    st = st[:-1]
    return st
def prepareMultipleRowsForPrediction(df,startP,endP):
    stLong = ''
    for k in range(startP,endP):
        tp = buildTheInput(df.values[k])
        stLong = stLong+tp+';'
    return stLong[:-1]
def changeString2ListofNumbers(st,startP):
    t = st.split(',')
    k = int(startP)
    for i in t:
        df1.at[k,'PredictedY'] = float(i)
        k = k+1
    return k

def main(argv):
    global df1
    infile = argv[0]
    data = pd.read_csv(infile)
    Var2Keep = ['x005', 'x006', 'x007', 'x008', 'x009', 'x010', 'x011', 'x012',
            'x013', 'x014', 'x015', 'x016', 'x017', 'x018', 'x019', 'x020',
            'x021', 'x022', 'x023', 'x024', 'x025', 'x026', 'x027', 'x028',
            'x029', 'x030', 'x031', 'x032', 'x033', 'x034', 'x035', 'x036',
            'x037', 'x038', 'x039', 'x040', 'x042', 'x043', 'x044', 'x045',
            'x046', 'x047', 'x048', 'x049', 'x050', 'x051', 'x052', 'x053',
            'x054', 'x055', 'x056', 'x059', 'x061', 'x062', 'x063', 'x064',
            'x065', 'x066', 'x071', 'x072', 'x073', 'x074', 'x075', 'x076',
            'x080', 'x081', 'x082', 'x088', 'x089', 'x097', 'x099', 'x104',
            'x106', 'x107', 'x108', 'x110', 'x111', 'x112', 'x113', 'x114',
            'x115', 'x116', 'x119', 'x120', 'x121', 'x126', 'x147', 'x168',
            'x169', 'x170', 'x171', 'x172', 'x173', 'x174', 'x177', 'x178',
            'x179', 'x181', 'x182', 'x183', 'x184', 'x185', 'x186', 'x187',
            'x188', 'x189', 'x190', 'x191', 'x192', 'x193', 'x194', 'x195',
            'x196', 'x198', 'x199', 'x200', 'x201', 'x209', 'x210', 'x211',
            'x224', 'x225', 'x226', 'x227', 'x228', 'x229', 'x230', 'x231',
            'x232', 'x233', 'x234', 'x236', 'x240', 'x244', 'x245', 'x246',
            'x247', 'x248', 'x249', 'x250', 'x251', 'x254', 'x258', 'x260',
            'x261', 'x262', 'x263', 'x264', 'x269', 'x270', 'x271', 'x272',
            'x273', 'x274', 'x276', 'x277', 'x278', 'x279', 'x280', 'x281',
            'x282', 'x283', 'x284', 'x285', 'x291', 'x292', 'x294', 'x296',
            'x298', 'x299', 'x300', 'x301', 'x303']

    #Create a dataframe with only the variables needed for prediction
    #We assume that the file column names are the same as the one given
    #for this test.
    df = data[Var2Keep]
    #As we want the result to be exported to a csv file,
    #I have decided to create a dataframe that will have the actual and the predicted
    #This will help me not only
    df1 = pd.DataFrame({'y': data['y'], 'PredictedY': 0})
    #Remove data from memory
    #We want to call the API for every 100 observations
    numberOfObsercations = df1.shape[0]
    stepS = 1000
    startT = time.time()
    for j in range(0, numberOfObsercations, stepS):
        s = prepareMultipleRowsForPrediction(df, j, j+stepS)
        predictionVar = call(s)
        predictionVar = predictionVar.decode('utf8')
        predictionVar = predictionVar[2:len(predictionVar)-2]
        l = changeString2ListofNumbers(predictionVar, j)
    endT = time.time()
    print('Total Time To Get Predictions :', endT - startT)
    print('RMSE :',rmse(df1['PredictedY'],df1['y']))
    print('MAPE :',mape(df1['PredictedY'],df1['y']))


    #The prediction is in prediction.csv
    export_csv = df1.to_csv ('walletPredictionOutput.csv', index = None, header=True)

if __name__ == "__main__":
        main(sys.argv[1:])




