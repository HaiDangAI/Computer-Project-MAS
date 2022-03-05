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

st.sidebar.title('More Option')
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 500px;
        margin-left: -500px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.sidebar.header('Select Country')
## Table select by country name
selectCountry = st.sidebar.multiselect(
    label='Countries want to show',
    options=dataframe['Country Name'],
    default=dataframe['Country Name'][:20])
Dataframe_selected_By_CountryName = dataframe.loc[dataframe['Country Name'].isin(selectCountry)]
st.header('Height of Male and Female by Country 2022')
st.subheader('Table selected by country name')
st.dataframe(Dataframe_selected_By_CountryName.style.highlight_max(axis=0, color='yellow'))#Table highlight the largest of each columns


## Table select by Average Age
select_Average_Age = st.sidebar.slider(
     label='Select a range of Average Height',
     min_value=min(dataframe['Average']), 
     max_value=max(dataframe['Average']), 
     value=(min(dataframe['Average']), max(dataframe['Average'])))
st.sidebar.write('Values:', (round(select_Average_Age[0],2), round(select_Average_Age[1],2)))
Dataframe_selected_By_AverageAge = dataframe.loc[dataframe['Average'].between(select_Average_Age[0],select_Average_Age[1])]
st.subheader('Table select by Average Age')
st.dataframe(Dataframe_selected_By_AverageAge.style.highlight_max(axis=0, color='yellow'))


st.sidebar.header('\nSelect Question')
# Question 1
CheckBox_Question1 = st.sidebar.checkbox('Question 1', True)
if CheckBox_Question1:
    st.write('Question 1: Determine the mean and standard deviation for height of Male and Female in the world?')
    data = {None:['Male', 'Female', 'Average'], 'Mean':[i for i in dataframe.mean()], 'Standard deviation':[i for i in dataframe.std()]}
    Dataframe_Question1 = pd.DataFrame(data)
    st.dataframe(Dataframe_Question1)




# Question 2
CheckBox_Question2 = st.sidebar.checkbox('Question 2', True)
if CheckBox_Question2:
    st.write('Question 2: Determine median, first quartile, third quartile, IQR')
    st.dataframe(dataframe.describe())


# Question 3
CheckBox_Question3 = st.sidebar.checkbox('Question 3', True)
if CheckBox_Question3:
    st.write('Question 3: Visual display that describes important features of data by Boxplot')
    fig, ax = plt.subplots()
    ax.boxplot(dataframe[['Male','Female','Average']])
    ax.set(ylabel='Height')
    ax.set_title('Three quartiles, min/max values and unusual observations of height', pad=20)
    plt.xticks([1, 2, 3], ['Male','Female','Average'])
    st.pyplot(fig)
    
    
    

# Question 4
CheckBox_Question4 = st.sidebar.checkbox('Question 4', True)
if CheckBox_Question4:
    st.write('Question 4: Create frequency distribution and visualize histogram for height')
    fig,ax=plt.subplots(nrows=1,
                        ncols=2,
                        figsize=(10,5))
    ax[0].hist(dataframe["Male"],bins=10,color="palegreen",alpha=0.8)
    ax[0].set(title="Height of male",ylabel="Number of peoples",xlabel="Height")
    ax[1].hist(dataframe["Female"],bins=10,color="palegreen",alpha=0.8)
    ax[1].set(title="Height of female",ylabel="Number of students",xlabel="Height")
    st.pyplot(fig)

st.markdown("""
    <iframe width="900" height="675" src="https://datastudio.google.com/embed/reporting/c5bb0dc2-3e9b-424c-9f72-84416e38869d/page/p_22itp37nsc" frameborder="0" style="border:0" allowfullscreen></iframe>
    """, unsafe_allow_html=True)









