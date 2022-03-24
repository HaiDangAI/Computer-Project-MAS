import math
from scipy.stats import norm, t
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt

def Question1(data_sample,st):
    
    st.write(f'''Question 1: The mean Male height of 30 randomly selected countries is normally distributed with the mean and standard deviation calculated
    for the 30 countries. Find the probability that the average height of a randomly chosen male is greater than 170cm''')
    col1, col2= st.columns(2)
    count = len(data_sample.Male.loc[data_sample.Male>170])
    result = pd.Series({'Number of country':30, 'Male greater than 170Cm':count, 'Probability':count/30})
    col1, col2= st.columns(2)
    with col1:
        st.write(result)
    with col2:
        st.code('''
data = data_sample.Male.loc[data_sample.Male>170]
count = len(data)
P = count/len(data_sample.Male)''',
                language='python')

def Question2(data_sample,st):
    st.write(f'''Question 2:The mean Female height of 30 randomly selected countries is normally distributed with the mean and standard deviation calculated for
    the 30 countries. Find the probability that the average height of a randomly chosen Female is greater than 160cm''')
    count = len(data_sample.Female.loc[data_sample.Female>160])
    result = pd.Series({'Number of country': 30,'Female greater than 160Cm':count ,'Probability': count / 30})
    col1, col2= st.columns(2)
    with col1:
        st.write(result)
    with col2:
        st.code('''
data = data_sample.Female.loc[data_sample.Female>160]
count = len(data)
P = count/len(data_sample.Female)
''',
                language='python')

def Question3(data_sample, st):
    st.write('''Question 3: The mean Male height of 30 randomly selected countries is normally distributed with the mean and standard deviation calculated
    for the 30 countries. Find mean, Standard Deviation, Varience, probability that the average height of a randomly chosen Female is greater than 160cm and 
less than 165cm.''')
    col1, col2= st.columns(2)
    with col1:
        meanMale, meanFemale = data_sample[['Male','Female']].mean()
        stdMale, stdFemale = data_sample[['Male','Female']].std()
        varMale, varFemale = stdMale**2, stdFemale**2
        st.dataframe(pd.DataFrame({'Gender':['Male','Female'],
                               'Mean':[meanMale, meanFemale],
                               'Standard':[stdMale,stdFemale],
                               'Variance':[varMale,varFemale]}))
        result=len([i for i in data_sample.Male if i>160 and i<165])/30
        st.write('Probability that the average height of a randomly chosen Female is greater than 160cm and less than 165cm.')
        st.write(result)
    with col2:
        st.code('''
meanMale, meanFemale = data_sample[['Male','Female']].mean()
stdMale, stdFemale = data_sample[['Male','Female']].std()
varMale, varFemale = stdMale**2, stdFemale**2
result=len([i for i in data_sample.Male if i>160 and i<165])/30
st.write('Probability that the average height of a randomly chosen Female is greater than 160cm and less than 165cm.')
st.write(result)
''', language='python')
        

def Question4(data_sample,st):
    st.write(f'''Question 4: Suppose that Height Male is a normally distributed random variable with mean μ = 173,09 and standard deviation σ = 4.95. Find the probability P(170 < Height Male < 180).
''')
    u = 173.09 # mean
    o = 4.95 # standard
    x1 = 170
    x2 = 180
    Z1 = (x1-u)/o
    Z2 = (x2-u)/o
    col1, col2= st.columns(2)
    with col1:
        st.write(f'Mean = {u}')
        st.write(f'Standard = {o}')
        st.write('The probability P(170 < Male < 180) is') 
        st.write(norm.cdf(Z2)-norm.cdf(Z1))
    with col2:
        st.code('''
u = 173.09 # mean
o = 4.95 # standard
x1 = 170
x2 = 180
Z1 = (x1-u)/o
Z2 = (x2-u)/o
st.write('The probability P(170 < Male < 180) is') 
st.write(norm.cdf(Z2)-norm.cdf(Z1))''',
                language='python')


def Question5(dfsample, st):
    st.write('''Question 5:\na.For your random sample of n = 30 observations. Find the probability that the sample mean of Height Female lies in [160, 170].''')
    x1 = 160
    x2 = 170
    u = dfsample['Female'].mean()
    o = dfsample['Female'].std()
    Z1 = (x1-u)/o
    Z2 = (x2-u)/o
    n = pow(o,2)
    col1, col2= st.columns(2)
    with col1:
        st.write(f'Mean = {u}')
        st.write(f'Standard = {o}')
        st.write('The probability P(160 < Female < 170) is') 
        st.write(norm.cdf(Z2)-norm.cdf(Z1))
        st.write('''b.How large must the random sample be if we want the standard error of the sample mean to be 1?''')
        st.write('The sample must greater than',math.floor(n+1))
    with col2:
        st.code('''
x1 = 160
x2 = 170
u = dfsample['Female'].mean()
o = dfsample['Female'].std()
Z1 = (x1-u)/o
Z2 = (x2-u)/o
st.write('The probability P(160 < Female < 170) is') 
st.write(norm.cdf(Z2)-norm.cdf(Z1))
n = pow(o,2)
st.write('The sample must greater than',math.floor(n+1))
''',
                language='python')


def Question6(st):
    st.write('''Question 6: Create Histogram for Height of Female.''')
    from PIL import Image
    image = Image.open('output.png')
    st.image(image, 'Histogram Height of Female')
    

def Question7(data_sample, st):
    st.write('Question 7: Find the point estimation of mean and variance of Height (Male and Female).')
    col1, col2= st.columns(2)
    with col1:
        st.dataframe(pd.DataFrame({'Gender':['Male','Female'],
                               'Mean':[data_sample.Male.mean(), data_sample.Female.mean()],
                               'Variance':np.power([data_sample.Male.std(), data_sample.Female.std()],2)}))
    with col2:
        st.code('''
meanMale = data_sample.Male.mean()
meanFemale = data_sample.Female.mean()
VarianceMale = math.pow(data_sample.Male.std(),2)
VarianceFemale = math.pow(data_sample.Female.std(),2)''',
                language='python')

def Question8(data_sample, st):
    st.write('''Question 8: Suppose that Height Male is a normally distributed random variable with standard deviation σ = 4.95.
             Construct a 95% confidence interval on the true mean Height Male using data in your subsample.''')
    col1, col2= st.columns(2)
    with col1:
        result=t.interval(alpha=0.95, df=len(data_sample['Male']-1), 
        loc=np.mean(data_sample['Male']),
        scale=4.95/math.sqrt(data_sample['Male'].dropna().count()))
        st.write(result[0],'<= mean <=',result[1])
    with col2:
        st.code('''
result=t.interval(alpha=0.95, df=len(data_sample['Male']-1), 
                loc=np.mean(data_sample['Male']),
                scale=4.95/math.sqrt(data_sample['Male'].dropna().count()))
st.write(result[0],'<= mean <=',result[1])''',
                language='python')
    

def Question9(data_sample, st):
    st.write('''Question 9: Suppose that Ft is a normally distributed random variable with standard deviation σ = 0.25
             Construct a 99% confidence interval on the true mean Ft using data in your subsample.''')
    col1, col2= st.columns(2)
    with col1:
        d=list(data_sample['Maleft'].append(data_sample['Femaleft']))
        df1=pd.DataFrame({'Ft':d})
        result=t.interval(alpha=0.99, df=len(df1['Ft']-1), 
                loc=np.mean(df1['Ft']),
                scale=0.25/math.sqrt(df1['Ft'].count()))
        st.write(result[0],'<= mean <=',result[1])
    with col2:
        st.code('''
d=list(data_sample['Maleft'].append(data_sample['Femaleft']))
df1=pd.DataFrame({'Ft':d})
result=t.interval(alpha=0.99, df=len(df1['Ft']-1), 
            loc=np.mean(df1['Ft']),
            scale=0.25/math.sqrt(df1['Ft'].count()))
st.write(result[0],'<= mean <=',result[1])''',
                language='python')

def Question10(data_sample, st):
    st.write('''Question 10: If we want the error in estimating the mean height Male from the two-size 
             confidence interval to be 3.69 at 95% confidence. What sample size should be used? 
             Assume that heght Male is a normally distributed random variable with standard deviation σ = 4.95.''')
    col1, col2= st.columns(2)
    with col1:
        z = norm.ppf(1-(0.05/2))
        n=math.pow((2*z*4.95)/3.69,2)
        result=math.ceil(n)
        st.write('The sample size should be used',result)
    with col2:
        st.code('''
z = norm.ppf(1-(0.05/2))
n=math.pow((2*z*4.95)/3.69,2)
result=math.ceil(n)
st.write('The sample size should be used',result)''',
                language='python')


def Question11(data_sample, st):
    st.write(f'''Question 11: Use your subsample to test the hypothesis H0: mean Male height = 170 against H1: 
             mean Male height ≠ 170 at α = 1%. Assume that Height of male is a normally distributed random 
             variable with standard deviation σ = 4.''')
    col1, col2= st.columns(2)
    with col1:
        mean_sample_IQ = data_sample.Male.dropna().mean()
        Z0_IQ = (mean_sample_IQ-170)/(4/math.sqrt(len(data_sample)))
        Z0005_IQ = norm.ppf(1-0.01/2)
        result = pd.Series({'Mean Sample Male Height':mean_sample_IQ, 'H0: Z0':Z0_IQ, 'H1: Z0.005':Z0005_IQ})
        if Z0_IQ>Z0005_IQ or Z0_IQ<-Z0005_IQ:
            result = result, 'Reject H0'
        else:
            result = result, 'Fail to reject H0'
        st.dataframe(result[0])
        st.write(result[1])
    with col2:
        st.code('''
mean_sample_IQ = data_sample.Male.dropna().mean()
Z0_IQ = (mean_sample_IQ-170)/(4/math.sqrt(len(data_sample)))
Z0005_IQ = norm.ppf(1-0.01/2)
if Z0_IQ>Z0005_IQ or Z0_IQ<-Z0005_IQ:
    result = result, 'Reject H0'
else:
    result = result, 'Fail to reject H0'
st.dataframe(result[0])
st.write(result[1])''',
                language='python')

def Question12(data_sample, st):
    st.write('''Question 12: Use your subsample to test the hypothesis H0: mean of Female = 160 against 
             H1: mean of Female > 160 at 𝛼 = 10%. Assume that height of female is a normally distributed random variable. ''')
    col1, col2= st.columns(2)
    with col1:
        mean_sample_Female = data_sample.Female.dropna().mean()
        standard_sample_Female = data_sample.Female.dropna().std()
        len_sample_Female = len(data_sample.Female.dropna())
        T0_Female = (mean_sample_Female-162)/(standard_sample_Female/math.sqrt(len_sample_Female))
        T01_Female = t.ppf(1-0.1,len_sample_Female-1)
        
        result = pd.Series({'Mean Sample Female Height':mean_sample_Female, 
                            'Standard Sample Female Height':standard_sample_Female ,
                            'H0: T0':T0_Female, 
                            f'H1: T0.1,{len_sample_Female-1}':T01_Female})
        if T0_Female>T01_Female:
            result = result, 'Reject H0'
        else:
            result = result, 'Fail to reject H0'
        st.dataframe(result[0])
        st.write(result[1])
    with col2:
        st.code('''
mean_sample_Female = data_sample.Female.dropna().mean()
standard_sample_Female = data_sample.Female.dropna().std()
len_sample_Female = len(data_sample.Female.dropna())
T0_Female = (mean_sample_Female-162)/(standard_sample_Female/math.sqrt(len_sample_Female))
T01_Female = t.ppf(1-0.1,len_sample_Female-1)

result = pd.Series({'Mean Sample Female Height':mean_sample_Female, 
                    'Standard Sample Female Height':standard_sample_Female ,
                    'H0: T0':T0_Female, 
                    f'H1: T0.1,{len_sample_Female-1}':T01_Female})
if T0_Female>T01_Female:
    result = result, 'Reject H0'
else:
    result = result, 'Fail to reject H0'
st.dataframe(result[0])
st.write(result[1])''',
        language='python')

def Question13(dataframe, data_sample, st):
    st.write('''Question 13: Assume that the variables are normally distributed and the variance 
             height within each country be equal for all countries. ''')
    col1, col2= st.columns(2)
    with col1:
        data_Q18_1 = dataframe.sample(random.randrange(20,50))
        n1 = len(data_Q18_1)
        data_Q18_2 = data_sample
        n2 = len(data_Q18_2)
        st.write('''a. Test the difference between the average height of male
                at 1% significance level based on the subsample.''')
        mean1 = data_Q18_1['Male'].mean()
        mean2 = data_Q18_2['Male'].mean()
        standard1= data_Q18_1.Male.std()
        standard2 = data_Q18_2.Male.std()
        st.dataframe(pd.DataFrame({'n':[n1,n2],
                        'Mean':[mean1, mean2],
                        'Standard':[standard1, standard2]}))
        Z0_Q18a = (mean1-mean2)/math.sqrt(math.pow(standard1,2)/n1+math.pow(standard2,2)/n2)
        Z0005_Q18a = norm.ppf(1-0.01/2)
        if Z0_Q18a>Z0005_Q18a or Z0_Q18a<-Z0005_Q18a:
            st.write(f'Reject H0')
        else:
            st.write(f'Fail to reject H0')
        st.write('''b. Test the difference between two independent Male proportions that country with 
                low Male height (Male < 170) by variable black at 5% significance level based on the data.''')
        numbers_lowHeight1 = len(data_Q18_1.loc[data_Q18_1.Male<170])
        numbers_lowHeight2 = len(data_Q18_2.loc[data_Q18_2.Male<170])
        P_lowH_1 = numbers_lowHeight1/n1
        P_lowH_2 = numbers_lowHeight2/n2
        st.dataframe(pd.DataFrame({'n':[n1,n2],
                                'Numbers <170 Cm':[numbers_lowHeight1,numbers_lowHeight2],
                                'P <170 Cm':[P_lowH_1,P_lowH_2]}))
        P = (numbers_lowHeight1+numbers_lowHeight2)/(n1+n2)
        Z0_Q18b = (P_lowH_1-P_lowH_2)/math.sqrt(P*(1-P)*(1/n1+1/n2))
        Z0025_Q18b = norm.ppf(1-0.05/2)
        if Z0_Q18b>Z0025_Q18b or Z0_Q18b<-Z0025_Q18b:
            st.write('Reject H0')
        else:
            st.write('Fail to reject H0')
        with col2:
            from PIL import Image
            image = Image.open('Q13.jpg')
            st.image(image)

def Question14(data_sample, st):
    st.write('''Question 14. Suppose you are interested in studying the effect of Female Height. Use your subsample to calculate the regression equation of Male Height.''')
    st.write('''a. Estimated standard error of the slope''')
    col1, col2= st.columns(2)
    with col1:
        X_train = data_sample['Female']
        Y_train= data_sample['Male']
        xi=sum(X_train)
        xi2=sum(np.power(X_train,2))
        Sxx=xi2-math.pow(xi,2)/30
        std=data_sample['Male'].std()
        SeB1=math.sqrt(math.pow(std,2)/Sxx)
        st.write(SeB1)
        
        st.write('''b. Estimated standard error of the intercept''')
        SeB0=math.sqrt(math.pow(std,2)*(1/30+math.pow(xi,2)/Sxx))
        st.write(SeB0)
    
        st.write('''c. Find the coefficient and intercept of determination''')
        x=np.array([list(X_train)]).T
        y=np.array([list(Y_train)]).T
        ones=np.ones((x.shape[0],1),dtype=np.int8)
        fig, ax = plt.subplots()
        ax.plot(x,y,'ro')
        st.pyplot(fig)
        A=np.concatenate((x,ones),axis=1)
        m=np.linalg.inv(A.T.dot(A)).dot(A.T.dot(y))
        coefficient=m[0][0]
        intercept=m[1][0]
        st.write(f'''Coefficient:{coefficient} and Intercept:{intercept}''')
        st.write('''d.Use your subsample to find the regression line height''')
        x0=np.array([[150,175]]).T
        y0=x0*m[0][0]+m[1][0]
        fig, ax = plt.subplots()
        ax.plot(x0,y0)
        ax.plot(x,y,'ro')
        st.pyplot(fig)      
    with col2:
        from PIL import Image
        image = Image.open('Q14.jpg')
        st.image(image)
    

