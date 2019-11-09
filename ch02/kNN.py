from numpy import *
import operator
from os import listdir

def createDataSet():
    '''
    params:
        goup: 打标好的四组数据
        labels: 即标记
    '''
    group = array([[1.0, 1.1],
                   [1.0, 1.0],
                   [0, 0],
                   [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    '''
    params:
        inX: 需要判断的输入数据
        dataSet: 已经打标好的数据
        labels: 标记
        k: 需要选取的最近邻居的个数
    '''
    dataSetSize = dataSet.shape[0] # shape会输出行和列, shape[0]即为行
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet # tile()可以理解为复制的意思，向右向下复制， 得到差矩阵(输入矩阵与原矩阵之差)
    sqDiffMat = diffMat ** 2 # 将差矩阵的每个元素的平方
    sqDistances = sqDiffMat.sum(axis=1) # 平方之后的每行加起来
    distances = sqDistances ** 0.5 # 和再开方得到距离
    sortedDistIndicies = distances.argsort() # 对距离进行排序，得到已排序的序列在distances里的索引
    classCount = {} # 新建分类统计变量
    for i in range(k): # 对离inX距离最近的k个数据进行标记的统计，从而得到inX最有可能的标记
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1 # 默认为0，如果get不到就是0，否则在其基础上加一
    sortedClassCount = sorted(classCount.items(), # 对classCount进行排序
                              key=operator.itemgetter(1),
                              reverse=True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines() # 得到所有的行
    numberOfLines = len(arrayOLines) # 得到行数
    returnMat = zeros((numberOfLines, 3)) # 初始化行数为文件行数，列数为3的0矩阵
    classLabelVector = [] # 标记向量
    index = 0
    for line in arrayOLines:
        line = line.strip() # 去除前后空格
        listFromLine = line.split('\t') # 以制表符为间隔分割
        returnMat[index,:] = listFromLine[0:3] # 将前三个数据给returnMat返回向量
        classLabelVector.append(listFromLine[-1]) # 将每行最后一个元素压入标记向量
        index += 1
    return returnMat, classLabelVector

def autoNorm(dataSet):
    '''
    dataSet: 需要归一化的数据
    '''
    minVals = dataSet.min(0) # min(0)得到每列的最小值，min(1)得到每行的最小值
    maxVals = dataSet.max(0) # max(0)得到每行的最大值，max(1)得到每行的最小值
    ranges = maxVals - minVals # 差
    m = dataSet.shape[0] # 行数
    normDataSet = dataSet - tile(minVals, (m, 1)) # (oldValue - minVal)
    normDataSet = normDataSet / tile(ranges, (m, 1)) # (oldValue - minValue) / (maxValue - minValue)
    return normDataSet, ranges, minVals

def datingClassTest():
    hoRation = 0.10 # 拿出10%的数据做测试
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt') # 从文件里加载数据
    normMat, ranges, minVals = autoNorm(datingDataMat) # 归一化数据
    m = normMat.shape[0] # 得到行数
    numTestVecs = int(m * hoRation) # 需要测试的数量
    errorCount = 0.0 # 错误数
    for i in range(numTestVecs):
        # 分类
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:], datingLabels[numTestVecs:m], 3)
        # 分类器得出的结果和原来的结果比较
        print("The classifier came back with: %d, the real answer is: %d" % (int(classifierResult), int(datingLabels[i])))
        # 不同的记录到errorCount
        if(classifierResult != datingLabels[i]):
            errorCount += 1.0
    # 得出错误率
    print("The total error rate is: %f" % (errorCount/float(numTestVecs)))

def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses'] # 结果的三种可能
    percentTats = float(input("percentage of time spent playing video games?")) # 打游戏的时间
    ffMiles = float(input("frequent flier miles earned per year?")) # 飞行距离
    iceCream = float(input("liters of ice cream cunsumed per year?")) # 消耗的冰淇淋数量
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt') # 从文件里加载数据
    normMat, ranges, minVals = autoNorm(datingDataMat) # 归一化数据
    inArr = array([ffMiles, percentTats, iceCream]) # 输入向量
    classifierResult = classify0((inArr - minVals)/ranges, normMat, datingLabels, 3) # 分类，但在分类之前要处理一下输入矩阵
    print("You will probably like this person: ", resultList[int(classifierResult) - 1]) # 得到结果

def img2vector(filename):
    returnVect = zeros((1, 1024)) # 初始化1行，1024列的图像向量
    fr = open(filename) # 打开文件
    for i in range(32): # 文件里有32行
        lineStr = fr.readline() # 读取每行
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j]) # 将每行读取到的数据一个一个给返回向量
    return returnVect

def handwritingClassTest():
    hwLabels = [] # 标记向量
    trainingFileList = listdir('trainingDigits') # 训练数字
    m = len(trainingFileList) # 训练文件的长度
    trainingMat = zeros((m, 1024)) # 初始化m * 1024的矩阵
    for i in range(m):
        fileNameStr = trainingFileList[i] # 0_0.txt
        fileStr = fileNameStr.split('.')[0] # 0_0
        classNumStr = int(fileStr.split('_')[0]) # 0
        hwLabels.append(classNumStr) # 将截取的数字压入标记向量
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr) # 读取文件里的数据并压入到训练向量里
    testFileList = listdir('testDigits') # 测试文件夹
    errorCount = 0.0 # 错误数
    mTest = len(testFileList) # 测试文件数
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0]) # 得到文件名中的数字
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr) # 得到测试文件中的向量
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3) # 分类结果
        print("The classifier came back with: %d, the real answer is: %d" % (int(classifierResult), classNumStr)) # 打印预测结果和原有结果
        if(int(classifierResult) != classNumStr):
            errorCount += 1.0
    print("The total error count is: %d" % errorCount) # 总的错误数
    print("The total error rate is: %f" % (errorCount/float(mTest))) # 总的错误率

if __name__ == '__main__':
    '''
    # case6 手写测试
    handwritingClassTest()
    # case5 图像转为向量
    testVector = img2vector('testDigits/0_1.txt')
    print(testVector[0, 32:63])
    # case4 分类测试和最终的使用
    datingClassTest()
    classifyPerson()
    
    # case3 归一化数据
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    print(minVals)
    # case2 从文件读取并处理成可用的数据
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    print(datingLabels[0:20])
    # case1 k近邻的简单表示
    '''
    dataSet, labels = createDataSet()
    print(classify0([0, 0], dataSet, labels, 3))