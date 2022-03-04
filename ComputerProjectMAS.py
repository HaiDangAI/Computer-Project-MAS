from audioop import avg
import streamlit as st
import pandas as pd

# Read and clean data
dataframe = pd.read_csv('Height of Male and Female by Country 2022.csv', usecols=['Country Name', 'Male Height in Cm', 'Female Height in Cm'])
dataframe = dataframe.rename(columns={'Male Height in Cm':'Male', 'Female Height in Cm':'Female'})
dataframe['Average']= (dataframe['Male']+dataframe['Female'])/2


# Create title and introduction
st.title('Computer Project MAS')
st.text('Wellcome everyone to come team n project')

st.sidebar.header('More Option')
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 500px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 500px;
        margin-left: -500px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

selectCountry = st.sidebar.multiselect(
                                    label='Countries want to show',
                                    options=dataframe['Country Name'],
                                    default=dataframe['Country Name'][:20])
dataframe_selected_By_CountryName = dataframe.loc[dataframe['Country Name'].isin(selectCountry)]

print(dataframe_selected_By_CountryName)


# Show the table
st.header('Height of Male and Female by Country 2022')
st.dataframe(dataframe_selected_By_CountryName.style.highlight_max(axis=0))#Table highlight the largest of each columns


select_Average_Age = st.sidebar.slider(
     label='Select a range of Average Height',
     min_value=min(dataframe['Average']), 
     max_value=max(dataframe['Average']), 
     value=(min(dataframe['Average']), max(dataframe['Average'])))
st.sidebar.write('Values:', select_Average_Age)






