import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Read and clean data
dataframe = pd.read_csv('Height of Male and Female by Country 2022.csv', usecols=['Country Name', 'Male Height in Cm', 'Female Height in Cm'])
dataframe = dataframe.rename(columns={'Male Height in Cm':'Male', 'Female Height in Cm':'Female'})
dataframe['Average']= (dataframe['Male']+dataframe['Female'])/2


# Create title and introduction
st.title('Computer Project MAS')
st.text('Wellcome everyone to come team n project')

st.sidebar.title('Questions')

st.header('Height of Male and Female by Country 2022')
st.dataframe(dataframe.style.highlight_max(axis=0, color='yellow'))


SubSample = dataframe.loc[dataframe['Country Name'].isin(dataframe['Country Name'].sample(30))]
st.header('Random sample')
st.dataframe(SubSample)#Table highlight the largest of each columns


## Table select by Average Age
# select_Average_Age = st.sidebar.slider(
#      label='Select a range of Average Height',
#      min_value=min(dataframe['Average']), 
#      max_value=max(dataframe['Average']), 
#      value=(min(dataframe['Average']), max(dataframe['Average'])))
# st.sidebar.write('Values:', (round(select_Average_Age[0],2), round(select_Average_Age[1],2)))
# Dataframe_selected_By_AverageAge = dataframe.loc[dataframe['Average'].between(select_Average_Age[0],select_Average_Age[1])]
# st.subheader('Table select by Average Age')
# st.dataframe(Dataframe_selected_By_AverageAge.style.highlight_max(axis=0, color='yellow'))


st.sidebar.header('\nSelect Question')
import Question as Q
# # Question 1
CheckBox_Question1 = st.sidebar.checkbox('Question 1', False)
if CheckBox_Question1:
    result = Q.Question1('Height of Male and Female by Country 2022.csv', st)
    st.write(result)
    

# Question 2
CheckBox_Question2 = st.sidebar.checkbox('Question 2', False)
if CheckBox_Question2:
    result = Q.Question2('Height of Male and Female by Country 2022.csv', st)
    st.write(result)


# Question 3
# CheckBox_Question3 = st.sidebar.checkbox('Question 3', False)
# if CheckBox_Question3:
#     st.write('Question 3: Visual display that describes important features of data by Boxplot')
#     fig, ax = plt.subplots()
#     ax.boxplot(dataframe[['Male','Female','Average']])
#     ax.set(ylabel='Height')
#     ax.set_title('Three quartiles, min/max values and unusual observations of height', pad=20)
#     plt.xticks([1, 2, 3], ['Male','Female','Average'])
#     st.pyplot(fig)
    
    
    

# Question 4
CheckBox_Question4 = st.sidebar.checkbox('Question 4', False)
if CheckBox_Question4:
    Q.Question4(st)





CheckBox_Question16 = st.sidebar.checkbox('Question 16', False)
if CheckBox_Question16:
    result = Q.Question16(SubSample,st)
    st.dataframe(result[0])
    st.write(result[1])

CheckBox_Question17 = st.sidebar.checkbox('Question 17', False)
if CheckBox_Question17:
    result = Q.Question17(SubSample,st)
    st.dataframe(result[0])
    st.write(result[1])

CheckBox_Question18 = st.sidebar.checkbox('Question 18', False)
if CheckBox_Question18:
    Q.Question18(dataframe, SubSample, st)

CheckBox_Map = st.sidebar.checkbox('Map', False)
if CheckBox_Map:
    st.markdown("""
        <iframe width="900" height="900" src="https://datastudio.google.com/embed/reporting/c5bb0dc2-3e9b-424c-9f72-84416e38869d/page/p_22itp37nsc" frameborder="0" style="border:0" allowfullscreen></iframe>
        """, unsafe_allow_html=True)
