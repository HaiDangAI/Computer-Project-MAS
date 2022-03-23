import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


# Read and clean data
dataframe = pd.read_csv('Height of Male and Female by Country 2022.csv', usecols=['Country Name', 'Male Height in Cm', 'Female Height in Cm', 'Male Height in Ft','Female Height in Ft'])
dataframe = dataframe.rename(columns={'Male Height in Cm':'Male', 'Female Height in Cm':'Female', 'Male Height in Ft':'Maleft',	'Female Height in Ft':'Femaleft'})

# dataframe['Average']= (dataframe['Male']+dataframe['Female'])/2


# Create title and introduction
st.title('Computer Project MAS')
# st.text('Wellcome everyone to come team n project')

videofile = open('video-1647835218.mp4', 'rb')
st.video(videofile.read(), start_time=0)

genre = st.sidebar.radio("Select file:",
    ('Question', 'QuestionVersion2'))
if genre == 'Question':
    import Question as Q
elif genre == 'QuestionVersion2':
    import QuestionVersion2 as Q

st.sidebar.title('Questions')

st.header('Height of Male and Female by Country 2022')
st.dataframe(dataframe.style.highlight_max(axis=0, color='yellow'))


SubSample = dataframe.loc[dataframe['Country Name'].isin(dataframe['Country Name'].sample(30))]
st.header('Random sample')
st.dataframe(SubSample)#Table highlight the largest of each columns


st.sidebar.header('\nSelect Question')

# # Question 1
CheckBox_Question1 = st.sidebar.checkbox('Question 1', True)
if CheckBox_Question1:
    Q.Question1(SubSample, st)
    

CheckBox_Question2 = st.sidebar.checkbox('Question 2', True)
if CheckBox_Question2:
    Q.Question2(SubSample, st)

CheckBox_Question3 = st.sidebar.checkbox('Question 3', True)
if CheckBox_Question3:
    Q.Question3(SubSample, st)

CheckBox_Question4 = st.sidebar.checkbox('Question 4', True)
if CheckBox_Question4:
    Q.Question4(SubSample,st)

CheckBox_Question5 = st.sidebar.checkbox('Question 5', True)
if CheckBox_Question5:
    Q.Question5(SubSample,st)

CheckBox_Question6 = st.sidebar.checkbox('Question 6', True)
if CheckBox_Question6:
    Q.Question6(st)

CheckBox_Question7 = st.sidebar.checkbox('Question 7', True)
if CheckBox_Question7:
    result = Q.Question7(SubSample,st)

CheckBox_Question8 = st.sidebar.checkbox('Question 8', True)
if CheckBox_Question8:
    result = Q.Question8(SubSample,st)

CheckBox_Question9 = st.sidebar.checkbox('Question 9', True)
if CheckBox_Question9:
    result = Q.Question9(SubSample,st)

CheckBox_Question10 = st.sidebar.checkbox('Question 10', True)
if CheckBox_Question10:
    Q.Question10(SubSample,st)

CheckBox_Question11 = st.sidebar.checkbox('Question 11', True)
if CheckBox_Question11:
    Q.Question11(SubSample,st)
    

CheckBox_Question12 = st.sidebar.checkbox('Question 12', True)
if CheckBox_Question12:
    Q.Question12(SubSample,st)
    

CheckBox_Question13 = st.sidebar.checkbox('Question 13', True)
if CheckBox_Question13:
    Q.Question13(dataframe, SubSample, st)

CheckBox_Question14 = st.sidebar.checkbox('Question 14', True)
if CheckBox_Question14:
    Q.Question14(SubSample, st)



CheckBox_Map = st.sidebar.checkbox('Map', False)
if CheckBox_Map:
    st.markdown("""
        <iframe width="1200" height="900" src="https://datastudio.google.com/embed/reporting/c5bb0dc2-3e9b-424c-9f72-84416e38869d/page/p_22itp37nsc" frameborder="0" style="border:0" allowfullscreen></iframe>
        """, unsafe_allow_html=True)
