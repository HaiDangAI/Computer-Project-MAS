import math
from scipy.stats import norm, t
import pandas as pd
import random

def Question1(file_name,st):
    def random_sample(count, start, stop, step=1):
        def gen_random():
            while True:
                yield random.randrange(start, stop, step)
 
        def gen_n_unique(source, n):
            seen = set()
            seenadd = seen.add
            for i in (i for i in source() if i not in seen and not seenadd(i)):
                yield i
                if len(seen) == n:
                    break
 
        return [i for i in gen_n_unique(gen_random,
                                        min(count, int(abs(stop - start) / abs(step))))]
    st.write(f'''Question 1: The mean Male height of 30 randomly selected countries is normally distributed with the mean and standard deviation calculated
    for the 30 countries. Find the probability that the average height of a randomly chosen male is greater than 170cm''')
    ar = random_sample(30, 1, 198)
    file = pd.read_csv(file_name)
    Rank = file['Rank']
    MaleH = file['Male Height in Cm']
    data = []
    count = 0
    for i in range(len(Rank)):
        for j in range(len(ar)):
            if (Rank[i] == ar[j]):
                data.append(MaleH[i])
    for i in data:
        if i>170:
            count+=1
    result = pd.Series({'Number of country':30, 'Male greater than 170Cm':count, 'Probability':count/30})
    return result

def Question2(file_name,st):
   def random_sample(count, start, stop, step=1):
       def gen_random():
           while True:
               yield random.randrange(start, stop, step)
       def gen_n_unique(source, n):
           seen = set()
           seenadd = seen.add
           for i in (i for i in source() if i not in seen and not seenadd(i)):
               yield i
               if len(seen) == n:
                   break

       return [i for i in gen_n_unique(gen_random,
                                       min(count, int(abs(stop - start) / abs(step))))]
   st.write(f'''Question 2:The mean Female height of 30 randomly selected countries is normally distributed with the mean and standard deviation calculated for
    the 30 countries. Find the probability that the average height of a randomly chosen Female is greater than 160cm''')
   ar = random_sample(30, 1, 198)
   file = pd.read_csv(file_name)
   Rank = file['Rank']
   MaleH = file['Female Height in Cm']
   data = []
   count=0
   for i in range(len(Rank)):
       for j in range(len(ar)):
           if (Rank[i] == ar[j]):
               data.append(MaleH[i])
   for i in data:
       if i>160:
           count+=1
   result = pd.Series({'Number of country': 30,'Female greater than 160Cm':count ,'Probability': count / 30})
   return result

def Question4(st):
    st.write('''Question 4: Create Histogram for Height of Female.''')
    from PIL import Image
    image = Image.open('output.png')
    st.image(image, 'Histogram Height of Female')
    

def Question16(data_sample, st):
    st.write(f'''Question 16: Use your subsample to test the hypothesis H0: mean Male height = 170 against H1: 
             mean Male height ≠ 170 at α = 1%. Assume that Height of male is a normally distributed random 
             variable with standard deviation α = 4.''')
    mean_sample_IQ = data_sample.Male.dropna().mean()
    Z0_IQ = (mean_sample_IQ-170)/(4/math.sqrt(len(data_sample)))
    Z0005_IQ = norm.ppf(1-0.01/2)
    result = pd.Series({'Mean of sample IQ':mean_sample_IQ, 'H0: Z0':Z0_IQ, 'Z0.005':Z0005_IQ})
    if Z0_IQ>Z0005_IQ or Z0_IQ<-Z0005_IQ:
        result = result, 'Reject H0'
    else:
        result = result, 'Fail to reject H0'
    return result

def Question17(data_sample, st):
    st.write('''Question 17: Use your subsample to test the hypothesis H0: mean of Female = 160 against 
             H1: mean of Female > 160 at 𝛼 = 10%. Assume that height of female is a normally distributed random variable. ''')
    mean_sample_lwage = data_sample.Female.dropna().mean()
    standard_sample_lwage = data_sample.Female.dropna().std()
    len_sample_lwage = len(data_sample.Female.dropna())
    T0_lwage = (mean_sample_lwage-162)/(standard_sample_lwage/math.sqrt(len_sample_lwage))
    T01_lwage = t.ppf(1-0.1,len_sample_lwage-1)
    
    result = pd.Series({'Mean of sample lwage':mean_sample_lwage, 
                        'Standard of sample lwage':standard_sample_lwage ,
                        'H0: T0':T0_lwage, 
                        f'H1: T0.1,{len_sample_lwage-1}':T01_lwage})
    if T0_lwage>T01_lwage:
        result = result, 'Reject H0'
    else:
        result = result, 'Fail to reject H0'
    return result

def Question18(dataframe, data_sample, st):
    st.write('''Question 18: Assume that the variables are normally distributed and the variance 
             height within each country be equal for all countries. ''')
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
    st.dataframe(pd.DataFrame({'len':[n1,n2],
                    'Mean':[mean1, mean2],
                    'Standard wage':[standard1, standard2]}))
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
    st.dataframe(pd.DataFrame({'len':[n1,n2],
                               'Numbers <170 Cm':[numbers_lowHeight1,numbers_lowHeight2],
                               'P <170 Cm':[P_lowH_1,P_lowH_2]}))
    P = (numbers_lowHeight1+numbers_lowHeight2)/(n1+n2)
    Z0_Q18b = (P_lowH_1-P_lowH_2)/math.sqrt(P*(1-P)*(1/n1+1/n2))
    Z0025_Q18b = norm.ppf(1-0.05/2)
    if Z0_Q18b>Z0025_Q18b or Z0_Q18b<-Z0025_Q18b:
        st.write('Reject H0')
    else:
        st.write('Fail to reject H0')
