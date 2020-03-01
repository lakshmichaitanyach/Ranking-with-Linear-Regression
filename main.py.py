
# coding: utf-8

# In[128]:


from sklearn.cluster import KMeans
import numpy as np
import csv
import math
import matplotlib.pyplot
from matplotlib import pyplot as plt


# In[129]:


maxAcc = 0.0
maxIter = 0 #maximum number of iterations the test should have
C_Lambda = 0.9 #initializing the value of c_lambda
TrainingPercent = 80 #the size of training set
ValidationPercent = 10 #size of validation set
TestPercent = 10 #size of test set
M = 2 
PHI = []
IsSynthetic = False


# In[130]:


def GetTargetVector(filePath):
    t = []
    with open(filePath, 'rU') as f: #'rU' will open a file for reading in universal newline mode
        reader = csv.reader(f) #read the data froom csv
        for row in reader:  
            t.append(int(row[0])) #append each row into the array t[]
    #print("Raw Training Generated..")
    return t

def GenerateRawData(filePath, IsSynthetic): #generationg a data matrix if Issynthetic is 1 
    dataMatrix = [] #define an empty data matrix
    with open(filePath, 'rU') as fi: #open the csv file
        reader = csv.reader(fi) #read the csv file
        for row in reader: #generating rows
            dataRow = [] #store the row in dataRow[]
            for column in row:
                dataRow.append(float(column)) #update every column of  the row in float values
            dataMatrix.append(dataRow) #updating data matrix row by row by appending new rows  
    
    if IsSynthetic == False : #iif issynthetic is not true delet the sub array along axis 1
        dataMatrix = np.delete(dataMatrix, [5,6,7,8,9], axis=1)
    dataMatrix = np.transpose(dataMatrix)  #transpose the matrix   
    #print ("Data Matrix Generated..")
    return dataMatrix

def GenerateTrainingTarget(rawTraining,TrainingPercent = 80): #generating targeting vector matrix for training set
    TrainingLen = int(math.ceil(len(rawTraining)*(TrainingPercent*0.01)))
    t           = rawTraining[:TrainingLen]
    #print(str(TrainingPercent) + "% Training Target Generated..")
    return t

def GenerateTrainingDataMatrix(rawData, TrainingPercent = 80): #generate training data matrix
    T_len = int(math.ceil(len(rawData[0])*0.01*TrainingPercent)) #length of the target is equal to the 0.01 times the length of raw data times the percent we are using (in this case 80)
    d2 = rawData[:,0:T_len]
    #print(str(TrainingPercent) + "% Training Data Generated..")
    return d2

def GenerateValData(rawData, ValPercent, TrainingCount):#generating the validation set
    valSize = int(math.ceil(len(rawData[0])*ValPercent*0.01)) #calculationg the validation size matrix
    V_End = TrainingCount + valSize #updating the validation end which will be the axis value
    dataMatrix = rawData[:,TrainingCount+1:V_End] #generation validation data matrix from raw data
    #print (str(ValPercent) + "% Val Data Generated..")  
    return dataMatrix 

def GenerateValTargetVector(rawData, ValPercent, TrainingCount): #generating validation target vector 
    valSize = int(math.ceil(len(rawData)*ValPercent*0.01)) 
    V_End = TrainingCount + valSize
    t =rawData[TrainingCount+1:V_End]
    #print (str(ValPercent) + "% Val Target Data Generated..")
    return t

def GenerateBigSigma(Data, MuMatrix,TrainingPercent,IsSynthetic):
    BigSigma    = np.zeros((len(Data),len(Data))) #returns an array of size of data with zeroes
    DataT       = np.transpose(Data) #transpose the data
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01)) #takes training length        
    varVect     = [] #create a variance vector
    for i in range(0,len(DataT[0])): 
        vct = []
        for j in range(0,int(TrainingLen)):
            vct.append(Data[i][j])    
        varVect.append(np.var(vct))
    
    for j in range(len(Data)):
        BigSigma[j][j] = varVect[j]
    if IsSynthetic == True:
        BigSigma = np.dot(3,BigSigma)
    else:
        BigSigma = np.dot(200,BigSigma)
    ##print ("BigSigma Generated..")
    return BigSigma

def GetScalar(DataRow,MuRow, BigSigInv): #generating scalar  
    R = np.subtract(DataRow,MuRow)
    T = np.dot(BigSigInv,np.transpose(R))  
    L = np.dot(R,T)
    return L

def GetRadialBasisOut(DataRow,MuRow, BigSigInv): #radial basis function    
    phi_x = math.exp(-0.5*GetScalar(DataRow,MuRow,BigSigInv))
    return phi_x

def GetPhiMatrix(Data, MuMatrix, BigSigma, TrainingPercent = 80): #generating phi_matrix
    DataT = np.transpose(Data) #transpose target vector of training set
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))         
    PHI = np.zeros((int(TrainingLen),len(MuMatrix))) #create a zero matrix
    BigSigInv = np.linalg.inv(BigSigma)#get the inverse of bigsig
    for  C in range(0,len(MuMatrix)):
        for R in range(0,int(TrainingLen)):
            PHI[R][C] = GetRadialBasisOut(DataT[R], MuMatrix[C], BigSigInv)
    #print ("PHI Generated..")
    return PHI

def GetWeightsClosedForm(PHI, T, Lambda):
    Lambda_I = np.identity(len(PHI[0])) #form an identity matrix of length phi
    for i in range(0,len(PHI[0])):
        Lambda_I[i][i] = Lambda
    PHI_T       = np.transpose(PHI) #transpose of phi
    PHI_SQR     = np.dot(PHI_T,PHI) #dot product between phi_t and phi
    PHI_SQR_LI  = np.add(Lambda_I,PHI_SQR) #finding phi_sqr_li
    PHI_SQR_INV = np.linalg.inv(PHI_SQR_LI) #finds multiplicative inverse of phi_sqr_inv
    INTER       = np.dot(PHI_SQR_INV, PHI_T) 
    W           = np.dot(INTER, T) #finding the weights
    ##print ("Training Weights Generated..")
    return W

def GetValTest(VAL_PHI,W):
    Y = np.dot(W,np.transpose(VAL_PHI))
    ##print ("Test Out Generated..")
    return Y

def GetErms(VAL_TEST_OUT,ValDataAct):
    sum = 0.0
    t=0
    accuracy = 0.0
    counter = 0
    val = 0.0
    for i in range (0,len(VAL_TEST_OUT)):
        sum = sum + math.pow((ValDataAct[i] - VAL_TEST_OUT[i]),2)
        if(int(np.around(VAL_TEST_OUT[i], 0)) == ValDataAct[i]):
            counter+=1
    accuracy = (float((counter*100))/float(len(VAL_TEST_OUT)))
    ##print ("Accuracy Generated..")
    ##print ("Validation E_RMS : " + str(math.sqrt(sum/len(VAL_TEST_OUT))))
    return (str(accuracy) + ',' +  str(math.sqrt(sum/len(VAL_TEST_OUT))))


# ## Fetch and Prepare Dataset

# In[131]:


RawTarget = GetTargetVector('Querylevelnorm_t.csv') #reading the target vector
RawData   = GenerateRawData('Querylevelnorm_X.csv',IsSynthetic) #reading the raw data


# ## Prepare Training Data

# In[132]:


TrainingTarget = np.array(GenerateTrainingTarget(RawTarget,TrainingPercent)) #prepare training data
TrainingData   = GenerateTrainingDataMatrix(RawData,TrainingPercent)
print(TrainingTarget.shape)
print(TrainingData.shape)


# ## Prepare Validation Data

# In[133]:


ValDataAct = np.array(GenerateValTargetVector(RawTarget,ValidationPercent, (len(TrainingTarget)))) #prepare validation data
ValData    = GenerateValData(RawData,ValidationPercent, (len(TrainingTarget)))
print(ValDataAct.shape)
print(ValData.shape)


# ## Prepare Test Data

# In[134]:


TestDataAct = np.array(GenerateValTargetVector(RawTarget,TestPercent, (len(TrainingTarget)+len(ValDataAct))))#prepare test data
TestData = GenerateValData(RawData,TestPercent, (len(TrainingTarget)+len(ValDataAct)))
print(ValDataAct.shape)
print(ValData.shape)


# ## Closed Form Solution [Finding Weights using Moore- Penrose pseudo- Inverse Matrix]

# In[135]:


ErmsArr = []
AccuracyArr = []

kmeans = KMeans(n_clusters=M, random_state=0).fit(np.transpose(TrainingData))
Mu = kmeans.cluster_centers_

BigSigma     = GenerateBigSigma(RawData, Mu, TrainingPercent,IsSynthetic) #getting the value of big_sigma,training_phi,w,test_phi,val_phi
TRAINING_PHI = GetPhiMatrix(RawData, Mu, BigSigma, TrainingPercent)
W            = GetWeightsClosedForm(TRAINING_PHI,TrainingTarget,(C_Lambda)) 
TEST_PHI     = GetPhiMatrix(TestData, Mu, BigSigma, 100) 
VAL_PHI      = GetPhiMatrix(ValData, Mu, BigSigma, 100)


# In[136]:


print(Mu.shape) #print the shapes of mu,bigsigma,training_phi,w,val_phi,test_phi
print(BigSigma.shape)
print(TRAINING_PHI.shape)
print(W.shape)
print(VAL_PHI.shape)
print(TEST_PHI.shape)


# ## Finding Erms on training, validation and test set 

# In[137]:


TR_TEST_OUT  = GetValTest(TRAINING_PHI,W) #get the val_test values for all training test and validation sets
VAL_TEST_OUT = GetValTest(VAL_PHI,W)
TEST_OUT     = GetValTest(TEST_PHI,W)

TrainingAccuracy   = str(GetErms(TR_TEST_OUT,TrainingTarget))
ValidationAccuracy = str(GetErms(VAL_TEST_OUT,ValDataAct))
TestAccuracy       = str(GetErms(TEST_OUT,TestDataAct))


# In[138]:


print ('UBITname      = lakshmic')
print ('Person Number = 50290974')
print ('----------------------------------------------------')
print ("------------------LeToR Data------------------------")
print ('----------------------------------------------------')
print ("-------Closed Form with Radial Basis Function-------")
print ('----------------------------------------------------')
print ("M = 10 \nLambda = 0.003")
print ("E_rms Training   = " + str(float(TrainingAccuracy.split(',')[1])))
print ("E_rms Validation = " + str(float(ValidationAccuracy.split(',')[1])))
print ("E_rms Testing    = " + str(float(TestAccuracy.split(',')[1])))


# ## Gradient Descent solution for Linear Regression

# In[139]:


print ('----------------------------------------------------')
print ('--------------Please Wait for 2 mins!----------------')
print ('----------------------------------------------------')


# In[140]:


W_Now        = np.dot(220, W)
La           = 2
learningRate = 0.01
L_Erms_Val   = []
L_Erms_TR    = []
L_Erms_Test  = []
W_Mat        = []

for i in range(0,400):
    
    #print ('---------Iteration: ' + str(i) + '--------------')
    Delta_E_D     = -np.dot((TrainingTarget[i] - np.dot(np.transpose(W_Now),TRAINING_PHI[i])),TRAINING_PHI[i])
    La_Delta_E_W  = np.dot(La,W_Now)
    Delta_E       = np.add(Delta_E_D,La_Delta_E_W)    
    Delta_W       = -np.dot(learningRate,Delta_E)
    W_T_Next      = W_Now + Delta_W
    W_Now         = W_T_Next
    
    #-----------------TrainingData Accuracy---------------------#
    TR_TEST_OUT   = GetValTest(TRAINING_PHI,W_T_Next) 
    Erms_TR       = GetErms(TR_TEST_OUT,TrainingTarget)
    L_Erms_TR.append(float(Erms_TR.split(',')[1]))
    
    #-----------------ValidationData Accuracy---------------------#
    VAL_TEST_OUT  = GetValTest(VAL_PHI,W_T_Next) 
    Erms_Val      = GetErms(VAL_TEST_OUT,ValDataAct)
    L_Erms_Val.append(float(Erms_Val.split(',')[1]))
    
    #-----------------TestingData Accuracy---------------------#
    TEST_OUT      = GetValTest(TEST_PHI,W_T_Next) 
    Erms_Test = GetErms(TEST_OUT,TestDataAct)
    L_Erms_Test.append(float(Erms_Test.split(',')[1]))


# In[141]:


print ('----------Gradient Descent Solution--------------------')
print ("M = 15 \nLambda  = 0.0001\neta=0.01")
print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),5)))
print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),5)))
print ("E_rms Testing    = " + str(np.around(min(L_Erms_Test),5)))

