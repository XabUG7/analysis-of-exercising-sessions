# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 11:37:04 2024

@author: hp
"""

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium


st.set_page_config(layout="wide", page_title="Exercising data analysis", page_icon=":bicyclist:")


st.write("#")
st.markdown("""<h5 style='text-align: left;'><strong>Author: </strong><a href="https://cv-xabier-urruchua.streamlit.app/" target="_blank">Xabier Urruchua</a></h5>""", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>Graphical analysis of exercising sessions from Strava step by step</h1>", unsafe_allow_html=True)
st.markdown("""#""")

# OVERVIEW
st.subheader("Overview")
st.markdown("""<h5 style='text-align: left;'>The purpose of this notebook is analysing the the exercising data of a client that wants to have several insights of his training. The data was taken by a variety of devices and was shared through the social sharing site named "Strava".</h5>""", unsafe_allow_html=True)




# IMPORTING LIBRARIES
st.subheader("Imports")

st.code("""
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import folium
        from folium.plugins import HeatMapWithTime
        import requests
        import io
        import geopy.distance
        import seaborn as sns
        """)
        
 
    
# Analysis
st.subheader("Analysis")
st.markdown("""The main focus of this analysis will be looking for patterns of the exercising routes and analysing the influence of the weather in our client's training routines. Besides that, the analysis of the data, will also get general information about the training zone of the client and patterns of training days and hours.""", unsafe_allow_html=True)


# Secondary header
st.markdown(r"""$\textsf{\large Reading data and making modifications for further analysis}$""")


st.markdown("""Having all the relevant libraries imported, all the tools that we will use for the present analysis are ready so first step is getting the data. In this case the data is stored in a csv file and will be imported using pandas library to create a pandas data frame for the data manipulation and processing. Once imported, we will look at the first rows of the data frame to have a first approcah to understanding the information that we have.""", unsafe_allow_html=True)
st.code("""
        df_raw = pd.read_csv("strava.csv")
        df_raw.head()
        """)

#Showing the raw data head
raw_df_head = pd.read_csv("assets/strava_raw_head.csv")
st.dataframe(raw_df_head)

st.markdown("""As time is one of the significant parameters for the analysis, we will take the timestamp of every measurement that is initially imported as a string and transform it into a more practical format using "to_datetime()" method from pandas. Apart from saving the timestamp, we will create a column that will contain only the date of the measurement, given that for part of the analysis we are only going to use the information of the exercising day and not the hour.""", unsafe_allow_html=True)     


st.code("""
        df = df_raw.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.sort_values("timestamp", inplace=True)
        df["date"] = df["timestamp"].dt.date
        """)

st.markdown("""After having a first look at the latitude and longitude data, we can see that they are stored in a format that is used for precision purposes but is not the widespread format of coordinates in degrees. Therefore, in order to make things easy for later analysis we will transform the data to latitude and longitude in degrees.""", unsafe_allow_html=True)
    


st.code("""
        df["position_lat_deg"] = df["position_lat"] * ( 180 / 2**31)
        df["position_long_deg"] = df["position_long"] * ( 180 / 2**31)
        df["coords"] = list(zip(df["position_lat_deg"],df["position_long_deg"]))
        """)    



# LINK WEATHER TO EXERCISE
st.markdown(r"""$\textsf{\large Linking the weather information to exercising data}$""")

st.markdown("""In order to relate the weather with the location of the client, we need to see where he trained and get weather data from that areas. For this, we will take the average of his locations per day. Assuming that the client will not train 2 times in a day in locations where climate is significantly different looks reasonable, so a simple mean of the latitudes and longitudes will be enough to see all training locations.""", unsafe_allow_html=True)

st.markdown("""Below code lines simply calculate the average values mentioned before and store the coordinates in a tuple that is linked to the day of training.""", unsafe_allow_html=True)

st.code("""
        df_daily_loc = df.groupby(["date"])[["position_lat_deg", "position_long_deg"]].mean()
        df_daily_loc["day_coords"] = list(zip(df_daily_loc["position_lat_deg"], df_daily_loc["position_long_deg"]))
        df_daily_loc.head(3)
        """)  
        
df_locations_head = pd.read_csv("assets/head_loc.csv")
st.dataframe(df_locations_head)



st.markdown("""A quick and interactive way of putting all the locations created above in a map is using folium. With this library we can provide only the coordinates and a map will be created.""", unsafe_allow_html=True)
st.markdown("""Representing the locations with folium:""", unsafe_allow_html=True)




# SHOWING THE TRAINING LOCATIONS IN FOLIUM
st.code("""
        m=folium.Map(location=[42.296,-83.768], zoom_start=13)
        
        for i in df_daily_loc["day_coords"]:
            folium.Marker(i).add_to(m)
        
        title = 'Exercising locations'
        title_html = '''
                     <h3 align="center" style="font-size:16px"><b>{}</b></h3>
                     '''.format(title) 
        
        # Adding title
        m.get_root().html.add_child(folium.Element(title_html))
        m.save('exercising_locations.html')
        
        m
        """) 
        
# Reading locations for folium
df_locations = pd.read_csv("assets/all_locations.csv")

df_locations["day_coords"] = list(zip(df_locations["position_lat_deg"], df_locations["position_long_deg"]))
        

# CREATING FOLIUM MAP
m=folium.Map(location=[42.296,-83.768], zoom_start=13)

for i in df_locations["day_coords"]:
    folium.Marker(i).add_to(m)

title = 'Exercising locations'
title_html = '''
             <h3 align="center" style="font-size:16px"><b>{}</b></h3>
             '''.format(title) 

# Adding title
m.get_root().html.add_child(folium.Element(title_html))
m.save('exercising_locations.html')

# DISPLAY FOLIUM MAPS WITH LOCATIONS
# Deploy folium map
st_folium(m, width=1525)


st.markdown("""Surfing through the world map we can see that all the training sessions that were registered were near Ann Arbor with the exception of 1 day training near Jackson. In order to make the analysis simpler and considering that Ann Arbor and Jackson are near each other, all the weather information for this analysis will be from Ann Arbor.""", unsafe_allow_html=True)


# DOWNLOAD WEATHER DATA FROM API
st.markdown(r"""$\textsf{\large Downloading weather data from a weather API}$""")

st.markdown("""Now that we have concluded that we will use the weather in Ann Arbor, we need to get the data and the API that we will use for this purpose will be the following one:

            https://rapidapi.com/visual-crossing-corporation-visual-crossing-corporation-default/api/visual-crossing-weather/""", unsafe_allow_html=True)



st.markdown("""The weather API needs some parameters that are included in the dictionary named "querystring". The meaning of the parameters are available in the API website. I will mention here the location (Ann Arbor), starting date and end date. Additionally, the API requires a key that is given to subscriptiors, however the information needed for this analysis is accessible with a free membership. The weather data will be donwloaded per day and the average weather conditions of the day will be used for the trainig days analysis.""", unsafe_allow_html=True)



st.markdown("""The cell below saves the code in the directory so we only need to download the data one time, for the rest of the executions, we will read it directly with pandas.""", unsafe_allow_html=True)


st.code("""
        url = "https://visual-crossing-weather.p.rapidapi.com/history"
        
        querystring = {"startDateTime":"2019-07-07T00:00:00","aggregateHours":"24","location":"Ann Arbor","endDateTime":"2019-10-04T00:00:00","unitGroup":"metric","contentType":"csv","shortColumnNames":"0"}
        
        headers = {
            "X-RapidAPI-Key": "**** GET YOUR KEY BY SUBSCRIBING ****",
            "X-RapidAPI-Host": "visual-crossing-weather.p.rapidapi.com"
        }
        
        response = requests.request("GET", url, headers=headers, params=querystring)
        
        
        df_weather = pd.read_csv(io.StringIO(response.text))
        df_weather.to_csv("ann_arbor_weather.csv", index=False)
        """)



# COMBINE WEATHER WITH EXERCISE DATA
st.markdown("""Combining running data with weather data will required to have date information in the same format as the strava data frame. Below cells will first convert it into the format we need and then the Strava data frame and the weather data frame will be merged using the date as the key. The new dataset will be named df_comb and will store all the information required for the current analysis. Below the columns of the dataset are shown allowing a better exploration of our data.""", unsafe_allow_html=True)



st.code("""
        df_weather["Date time"] = pd.to_datetime(df_weather["Date time"])
        df_weather["Date time"] = df_weather["Date time"].dt.date
        df_comb = pd.merge(df, df_weather, how="left", left_on=["date"], right_on=["Date time"])
        """)


# Getting GENERAL INFORMATION ABOUT TRAINING
st.markdown(r"""$\textsf{\large General information about training type}$""")

st.markdown("""First thing that we need to understand about our client is the type of training he is doing. There are several ways to calculate the training type but the one that we will use here will be using the percentage of maximum heart rate. The heart rate data will be plotted in a violin plot together with the HRmax value.""", unsafe_allow_html=True)

st.code('''
        distribution = (df_comb["heart_rate"].dropna());
        plt.figure(figsize=(15,10));
        plt.violinplot(distribution, showmeans=True);
        plt.title("Heart rate", fontsize=20);
        plt.ylabel("Beats per minute",fontsize=14);
        
        plt.annotate(f"""Max heart rate {np.max(df_comb["heart_rate"])}
                     Average heart rate {round(np.mean(df_comb["heart_rate"]),2)}
                     Percentage of HRmax {round(round(np.mean(df_comb["heart_rate"]),2) / np.max(df_comb["heart_rate"]) * 100,2)}%""", 
                     xy=(1, 120), xycoords='data', xytext=(100, 120), textcoords='axes points', 
                     fontsize=16, zorder=3);
        ''')

# SHOW EXERCISE TYPE PICTURE
st.image("assets/heart_violin.png")


st.markdown("""We can see in the above plot that the average training zone is 73.6% of HRmax. It means that the average most of the training is done in zone 3. Working out in heart rate zone 3 is especially effective for improving the efficiency of blood circulation in the heart and skeletal muscles. This is the zone in which that pesky lactic acid starts building up in your bloodstream. We can also see from the violin plot that although most of the training is done in zone 3, the client also has periods of training in zone 2 and zone 4.""", unsafe_allow_html=True)

# UNDESRTANGING WEATHER INFLUENCE
st.markdown(r"""$\textsf{\large Understanding the influence of the weather}$""")


# UNDERSTANDING STRAVA DATA
st.markdown("""First step to analyse the influence of the weather, will be understanding the data in Strava. Firstly we want to understand the distance parameter, so we will use geopy and compare the distance between the coordinates with the distance in the "distance" column, which will give us an idea about units used for measurements.""", unsafe_allow_html=True)

st.markdown("""Below code compares consecutive coordinates distance with the distance column. One comparisson is done with a random test and another one is done comparing row 33 with row 34 (33 is just an example but can be modified in the code cell if wanted).""", unsafe_allow_html=True)



st.code('''
        # This assertion may fail if the random number is at the end of the training session
        # but probability is very low
        rand_1 = np.random.randint(1,df_comb.shape[0])
        rand_2 = 33
        
        # testing the distance column against the geological distance
        coords_1 = df_comb["coords"][rand_1]
        coords_2 = df_comb["coords"][rand_1+1]
        dist_geo_1 = geopy.distance.geodesic(coords_1, coords_2).m
        dist_df_1 = df_comb["distance"][rand_1+1] - df_comb["distance"][rand_1]
        
        assert round(dist_geo_1) == round(dist_geo_1)
        
        coords_3 = df_comb["coords"][rand_2]
        coords_4 = df_comb["coords"][rand_2+1]
        dist_geo_2 = geopy.distance.geodesic(coords_3, coords_4).m
        dist_df_2 = df_comb["distance"][rand_2+1] - df_comb["distance"][rand_2]
        
        assert round(dist_geo_2) == round(dist_geo_2)
        ''')

st.markdown("""The above assertions have small chances of failing because there is a chance to assert the end of a training session vs the beging of a new one. In case of error it is recommended to run the cell again and see if assertions are passed.""", unsafe_allow_html=True)

st.markdown("""Above cell uses the library geopy that was installed at the beginning of the notebook. We can see that the values are very similar so we conclude that the distances in the Strava data are in meters. Next step, is separating running sessions from cycling sessions. Leg Spring Stiffness is measured by a device named stryd and is used for running. As a first step, we will consider running whenever we have Leg Spring Stiffness measurement, and we will consider cycling for the rest of the cases. For this we will create 2 new columns named "run" and "bike" that will have as values 1 and nan.""", unsafe_allow_html=True)


st.code('''
        df_comb["run"] = df_comb["Leg Spring Stiffness"].apply(lambda x: np.nan if np.isnan(x) else 1)
        df_comb["bike"] = df_comb["Leg Spring Stiffness"].apply(lambda x: 1 if np.isnan(x) else np.nan)
        ''')
        
# DIVIDE AND NAME EXERCISING SESSIONS
st.markdown("""Next step to understan the data, will be seeing if we can use the distances to separate the exercising sessions. For this we can plot the distances with the data frame index and see if the diatances go to Zero after every session or the measuring device accumulates the total distance. In the same plot we will add the distinction between "run" and "bike" and the enhanced speed to see if all assumptions make sense. The Plot will have normalized values in order to compare all the parameters in the same scale.""", unsafe_allow_html=True)


st.markdown("""The 2 cells below plot the distances, the speed, stryd usage and the type of exercise done thorughout time.""", unsafe_allow_html=True)

st.code('''
        last_run_date = df_comb["timestamp"][df_comb["run"] == 1].values[-1].astype('datetime64[D]')
        last_run_index = df_comb["timestamp"][df_comb["run"] == 1].idxmax()
        stryd_use = df_comb["timestamp"][df_comb["run"] == 1].values[0].astype('datetime64[D]')
        ''')
        
st.code('''
        plt.figure(figsize=(25,10),dpi=300)
        plt.plot(df_comb.index, df_comb["distance"]/np.max(df_comb["distance"]),label="Distance")
        plt.plot(df_comb.index, df_comb["run"]/np.max(df_comb["run"]),color="red",label="Running")
        plt.plot(df_comb.index, df_comb["bike"]/np.max(df_comb["bike"]),label="Cycling",color="black")
        plt.plot(df_comb.index, df_comb["enhanced_speed"]/np.max(df_comb["enhanced_speed"]), alpha=0.25, label="Speed")
        
        plt.axvspan(last_run_index-200, last_run_index, color='green', alpha=0.5)
        plt.axvspan(0, 200, color='green', alpha=0.5)
        plt.axvspan(df_comb.index[-1]-200, df_comb.index[-1], color='green', alpha=0.5)
        
        plt.annotate('Running', 
                     xy=(200, 0.6), xycoords='data', xytext=(500, 315), textcoords='axes points', 
                     arrowprops=dict(arrowstyle='->, head_width=0.5', linewidth=2),fontsize=16, zorder=3)
        plt.annotate('Running', 
                     xy=(32600, 0.6), xycoords='data', xytext=(500, 315), textcoords='axes points', 
                     arrowprops=dict(arrowstyle='->, head_width=0.5', linewidth=2),fontsize=16, zorder=3)
        
        
        plt.annotate('Cycling', 
                     xy=(32870, 0.6), xycoords='data', xytext=(1200, 315), textcoords='axes points', 
                     arrowprops=dict(arrowstyle='->, head_width=0.5', linewidth=2),fontsize=16, zorder=3)
        plt.annotate('Cycling', 
                     xy=(40488, 0.6), xycoords='data', xytext=(1200, 315), textcoords='axes points', 
                     arrowprops=dict(arrowstyle='->, head_width=0.5', linewidth=2),fontsize=16, zorder=3)
        
        plt.annotate(f'On {stryd_use} started to use Stryd\nto monitor running parameters', 
                     xy=(15000, 1), xycoords='data', xytext=(388, 445), textcoords='axes points', 
                     arrowprops=dict(arrowstyle='->, head_width=0.5', linewidth=2),fontsize=16, zorder=3)
        
        
        
        plt.xlabel("Measurement number",fontsize=14);
        plt.ylabel("Normalized values",fontsize=14);
        plt.title("Distance, speed and run/bike sessions normalized",fontsize=17);
        plt.legend(loc=0);
        ''')
        
st.image("assets/dist_speed.png")      

  
        
        
        

st.markdown("""Looking at the plot above, we can see that we are able to separate exercising sessions. The strategy for this will be counting sessions and when the distance decreases from one row to another we will count it as another session. We will take the last row and multiply it by 1000 to mark the latest measurement of each session. We can observe in the graph that all sessions go back to zero or almost zero meters distance so in case there is some error and a sessions starts from a very low value we will not change our analysis. Additionally, we know that red line has stryd measurements so we know that under the red line we have running sessions. Visually, the average speed for running is the same from the beginning and also the distances are similar. So we can assume that the client started running, bought a Stryd to monitor running parameters and after some time started cycling. Now we will store the real running and cycling sessions in a column named exrc_type to continue with the analysis.""", unsafe_allow_html=True)

# LABELLING RUN AND BIKE SESSIONS IN THE DATA
st.code('''
        # adding the type of exercise (Run or Cycle) to the dataframe
        exerc_type = []
        for i in range(df_comb.shape[0]):
            if i <= last_run_index:
                exerc_type.append("Running")
            else:
                exerc_type.append("Cycling")
                
        df_comb["exerc_type"] = exerc_type
        ''')
        
        
        
        


st.markdown("""The cell below will implement the strategy to count sessions explained above. for illustration of the strategy, the column "training_session" will have values like: 1,1,1,1,1,1,1000,2,2,2,2000. From 1 to 1000 is the first training session, for 2 to 2000 the second one and so on. Values above 1000 will mark the last reading of each session.""", unsafe_allow_html=True)



st.code('''
        training_session = []
        training_session.append(1)
        
        count = 1
        for i in range(1, len(df_comb["distance"])):
            
            if df_comb["distance"][i] >= df_comb["distance"][i-1]:
                training_session.append(count)
                
            elif df_comb["distance"][i] < df_comb["distance"][i-1]:
                training_session[-1] = count*1000
                count += 1
                training_session.append(count)
        
        training_session[-1] *= 1000
                
        df_comb["training_session"] = training_session
        ''')
        


# HEART RATE DATA
st.markdown("""Now that we have the information about each training session, we will gate a dictionary with the average heart rate per session.""", unsafe_allow_html=True)

st.code('''
        heart_rate_dic = {}
        
        for i in range(len(df_comb["training_session"])):
            
            if df_comb["training_session"][i] < 1000:
                try:
                    heart_rate_dic[df_comb["training_session"][i]*1000].append(df_comb["heart_rate"][i])
                except:
                    heart_rate_dic[df_comb["training_session"][i]*1000] = [df_comb["heart_rate"][i]]
                    
            if df_comb["training_session"][i] >= 1000:
                heart_rate_dic[df_comb["training_session"][i]].append(df_comb["heart_rate"][i])
                
        for dic_key in heart_rate_dic.keys():
            
            temp_np_array = np.array(heart_rate_dic[dic_key])
            
            # remove nan values
            temp_np_array= temp_np_array[~np.isnan(temp_np_array)]
            
            # Save the mean in the dictionary
            heart_rate_dic[dic_key] = round(temp_np_array.mean(),2)
        ''')


st.markdown("""Now we will do the same with speed and get a dictionary with the average speed per session.""", unsafe_allow_html=True)

st.code('''
        speed_dic = {}

        for i in range(len(df_comb["training_session"])):
            
            if df_comb["training_session"][i] < 1000:
                try:
                    speed_dic[df_comb["training_session"][i]*1000].append(df_comb["enhanced_speed"][i])
                except:
                    speed_dic[df_comb["training_session"][i]*1000] = [df_comb["enhanced_speed"][i]]
                    
            if df_comb["training_session"][i] >= 1000:
                speed_dic[df_comb["training_session"][i]].append(df_comb["enhanced_speed"][i])
                
        for dic_key in speed_dic.keys():
            
            temp_np_array = np.array(speed_dic[dic_key])
            # remove nan values
            temp_np_array= temp_np_array[~np.isnan(temp_np_array)]
            speed_dic[dic_key] = round(temp_np_array.mean(),2)
        ''')
        
        
        
        
# COMPARING EXERCISE WITH WEATHER DATA IN A GRAPH

st.markdown("""In order to compare the effect of the weather in our client, we will get the information that we are interested in and create scatterplot matrices, also known as SPLOMs. We will use here the dictionaries with the average heart rate and average speed to fill the data frame columns. For this purpose we will use the apply method from pandas. We want the information to be as clear as possible so the SPLOMs for running and cycling will be different.""", unsafe_allow_html=True)

st.code('''
        df_pairplot = df_comb[["distance","Precipitation","Temperature","exerc_type", "training_session"]][df_comb["training_session"] >=1000]
        df_pairplot["Avg_heart_rate"] = df_pairplot["training_session"].apply(lambda x: heart_rate_dic[x])
        df_pairplot["Avg_speed"] = df_pairplot["training_session"].apply(lambda x: speed_dic[x])
        df_pairplot.head(3)
        ''')

pairplot_head = pd.read_csv("assets/pairplot_head.csv")
st.dataframe(pairplot_head)        
        
st.markdown("""Transforming distance to km units and creating the names of the axes for the sploms plot.""", unsafe_allow_html=True)

st.code('''
        df_pairplot["distance"] = df_pairplot["distance"]/1000
        splom_columns = {"Precipitation":"Precipitation (mm)","Temperature":"Temperature (celsius)","Avg_speed":"Avg_speed m/s","distance":"Distance (km)"}
        ''')
        
st.markdown("""Creating SPLOMs for running.""", unsafe_allow_html=True)

       
st.code('''
        g1 =sns.pairplot(df_pairplot[["Precipitation","Temperature","exerc_type", "Avg_heart_rate","Avg_speed",
                                      "distance"]][df_pairplot["exerc_type"]=="Running"].rename(columns=splom_columns),
                         kind='reg', diag_kind='kde', plot_kws={'line_kws':{'color':'red'}});
        
        g1.fig.suptitle("Running exercise", y=1.04, fontsize=20);
        g1.fig.set_size_inches(16,16);
        ''')
# COMPARING EXERCISE WITH WEATHER DATA GRAPH 1
st.image("assets/SPLOM1.png") 






st.markdown("""Analysing the resulting plots, we can highlight the following information.""", unsafe_allow_html=True)
st.markdown("""- Speed and heart reate have the strongest correlation, which is normal because in order to keep a high speed training a greater effort is required.""", unsafe_allow_html=True)
st.markdown("""- Although the measurements for precipitations are usually around zero because big part of the data was taken in summer, we can see that the regression line has negative slope, which means that the higher the precipitations, the distance runned is smaller. However, the slope of speed vs precipitations graph suggests that when having precipitations the client runs slightly faster.""", unsafe_allow_html=True)
st.markdown("""- The regression line of temperature and distance is flat so it looks like contrarily to the precipitations, temperature does not affect much our clients running distance. However, temperature and heart rate have negative slope which means that the level of effort lowers with temperature.""", unsafe_allow_html=True)
st.markdown("""- Distance and heart rate have positive but low slope so looks like for longer distance trainings, the client relies more on mental strength to run for longer time, rather than pushing to finish faster. This is also inline with the distance vs speed plot which shows that for longer distances the speed is lower (negative slope).""", unsafe_allow_html=True)





st.markdown("""Creating SPLOMs for cycling.""", unsafe_allow_html=True)

st.code('''
        g2 = sns.pairplot(df_pairplot[["Precipitation","Temperature","exerc_type", "Avg_heart_rate","Avg_speed",
                                       "distance"]][df_pairplot["exerc_type"]=="Cycling"].rename(columns=splom_columns),
                          kind='reg', diag_kind='kde', plot_kws={'line_kws':{'color':'red'}});
        
        g2.fig.suptitle("Bicycle exercise", y=1.04, fontsize=20);
        g2.fig.set_size_inches(16,16);
        ''')
# COMPARING EXERCISE WITH WEATHER DATA GRAPH 2
st.image("assets/SPLOM2.png") 



# CONCLUSIONS WEATHER AND TRAINING DATA
st.markdown("""The data frame for cycling is smaller so the conclusions from SPLOMs are more relative but we can say the following:""", unsafe_allow_html=True)
st.markdown("""- Heart rate and speed have lower slope than running which means that small change in heart rate causes bigger change in speed.""", unsafe_allow_html=True)
st.markdown("""- Temperature vs distance regression line is close to flat so similar conclusions than previous graph can apply.""", unsafe_allow_html=True)
st.markdown("""- The error areas for precipitations are quite big which suggests that the conclusions taken for cycling regarding precipitations may not be accurate.""", unsafe_allow_html=True)
st.markdown("""- Heart rate and distance have close to zero slope which means that for longer cycling distances the client mantains the same speed for longer, which is aligned with the information taken from running data.""", unsafe_allow_html=True)

# ANALYSING TRAINING TIME AND LOCATION
st.markdown(r"""$\textsf{\large Analysing training times and locations}$""")




st.markdown("""The next part of the analysis will focus on the training patterns of the client when it comes to training times, days and locations.""", unsafe_allow_html=True)

st.markdown("""First we will plot the hour at the end of each training session and the day of each session. This way we will understand how the clients trains and the habits.""", unsafe_allow_html=True)
st.markdown("""Taking the relevant information from the main data frame named df_comb""", unsafe_allow_html=True)


st.code('''
        df_training_times = df_comb["timestamp"][df_comb["training_session"]>=1000].to_frame()
        df_training_times["day"] = df_training_times["timestamp"].apply(lambda x: pd.to_datetime(x).day_name())
        df_training_times["hour"] = df_training_times["timestamp"].apply(lambda x: pd.to_datetime(x).hour)
        df_training_times.head(3)
        ''')
        
train_times_df = pd.read_csv("assets/train_times.csv")   
st.dataframe(train_times_df)


     
        
st.markdown("""Creating subplots with training hours and days.""", unsafe_allow_html=True)


# CREATING THE BAR CHARTS WITH HOURS AND FREQUENCY AND DAY OF THE WEEK
st.code('''
        figure, ax = plt.subplots(1,2, figsize=(24, 10))
        
        training_hours = df_training_times.groupby("hour").count()["day"]
        
        ax[0].bar(training_hours.index.astype("str"), training_hours);
        
        ax[0].set_title("The hours of the end of trainings",fontsize=15);
        ax[0].set_xlabel("Hour of the day")
        ax[0].set_ylabel("Number of trainings")
        
        training_days= df_training_times.groupby("day").count()["hour"].sort_values(ascending=False)
        
        ax[1].bar(training_days.index,training_days);
        
        ax[1].set_title("Trainings distributed in days of the week",fontsize=15);
        ax[1].set_xlabel("Day of the week")
        ax[1].set_ylabel("Number of trainings");
        ''')
        
st.image("assets/hours.png")        
        
# TRAINIG TIME AND DAY CONCLUSION   
st.markdown("""We can see in the above plots that our client prefers to train during the night having even finished several training sessions at 1 a.m. However, the favourite training times are between 9 and 11 p.m. It is also significant that the earliest end of session is at noon, meaning that the client prefers to stay late at night for traininig rather than waking up early to exercise.""", unsafe_allow_html=True)

st.markdown("""Regarding the training days, the favourite day for training is Wednesday followed by Saturday and Sunday. It means that is the type of person that exercises during the weekend. We also observe that the least training sessions are done on Friday which can be considered normal because it is the last day of the week and people tend to be tired.""", unsafe_allow_html=True)


# CREATING A HEATMAP FOR EXERCISING DAYS AND ROUTES
st.markdown("""Finally, heat maps with the training routes and heart heart rate values for color will be created to see if there are any significant patterns that are visible. First step is getting all the necessary data from the main dataset.""", unsafe_allow_html=True)

st.code('''
        df_heatmap = df_comb[["date", "position_lat_deg", "position_long_deg", "heart_rate", "timestamp"]].dropna()
        df_heatmap["hour"] = df_heatmap["timestamp"].apply(lambda x: pd.Timestamp(x).hour)
        time_index = [i.strftime('%m-%d-%Y') for i in df_heatmap["date"].unique()]
        df_heatmap.head(3)
        ''')

heatmap_head_df = pd.read_csv("assets/heatmap_head.csv")
st.dataframe(heatmap_head_df)

st.markdown("""The heat maps will reflect the training routes and heart rates per day, and also per training hour. The folium heat map needs 2 inputs, a list of lists containing coordinates and heart rate values and a list of indexes for every values in the list of lists.""", unsafe_allow_html=True)

st.markdown("""Firstly, a dictionary will be created that will have dates as keys and will store lists of coordinates and heart rate.""", unsafe_allow_html=True)


# PREPARING HEART RATE AND LOCATION DATA
st.code('''
        location_heart_dic = {}
        
        
        for i in range(len(df_heatmap["position_lat_deg"])):
            
            coords_heart_list = [df_heatmap["position_lat_deg"].values[i],
                                 df_heatmap["position_long_deg"].values[i], df_heatmap["heart_rate"].values[i]]
            date_temp = df_heatmap["date"].values[i].strftime('%m-%d-%Y')
            
            try:
                location_heart_dic[date_temp].append(coords_heart_list)
            except:
                location_heart_dic[date_temp] = [coords_heart_list]
        ''')

st.markdown("""Secondly, we will take the keys of the dictionary to create the index list and store all the values of the dictionary in a single list.""", unsafe_allow_html=True)

st.code('''
        data_heat = []
        
        for day in location_heart_dic.keys():
            data_heat.append(location_heart_dic[day])
            
        day_index = list(location_heart_dic.keys())
        day_index = [f"Training on {d}" for d in day_index]
        ''')


st.markdown("""Now that we have the list of dates as index and list of locations with heart rate, we will follow the same steps to do the same with the trainig hours as the index.""", unsafe_allow_html=True)

st.code('''
        location_heart_dic_hour = {}
        
        
        for i in range(len(df_heatmap["position_lat_deg"])):
            
            coords_heart_list = [df_heatmap["position_lat_deg"].values[i],
                                 df_heatmap["position_long_deg"].values[i], df_heatmap["heart_rate"].values[i]]
            hour_temp = df_heatmap["hour"].values[i]
            
            try:
                location_heart_dic_hour[hour_temp].append(coords_heart_list)
            except:
                location_heart_dic_hour[hour_temp] = [coords_heart_list]
        ''')
        
st.code('''
        data_heat_hour = []
        
        for day in sorted(list(location_heart_dic_hour.keys())):
            data_heat_hour.append(location_heart_dic_hour[day])
            
        hour_index = sorted(list(location_heart_dic_hour.keys()))
        hour_index = [f"Trainings at {h}H"for h in hour_index]
        ''')


st.markdown("""The four lists created above, will be combined into a list of indexes and a list of locations. The result will be that the folium map will play all the routes linked to training days and once finished all the routes linked to the training hour. It means that training hours will have several routes in the same frame.""", unsafe_allow_html=True)



# CREATING FOLIUM MAP
st.code("""
        hmt = folium.Map(location=[42.296,-83.768], zoom_start=13,
                       tiles='cartodbpositron',
                       control_scale=True)
        
        HeatMapWithTime(data_heat + data_heat_hour,
                        index=day_index + hour_index,
                        auto_play=True,
                        use_local_extrema=True
                       ).add_to(hmt)
        
        title = 'Exercising routes and heart rate values heat map'
        title_html = '''
                     <h3 align="center" style="font-size:16px"><b>{}</b></h3>
                     '''.format(title) 
        
        # Adding title
        hmt.get_root().html.add_child(folium.Element(title_html))
        hmt.save('exercising_routes_days_hours.html')
        
        hmt
        """)



# DISPLAY THE HEATMAP
path_to_html = "assets/heat_map.html" 

# Read file and keep in variable
with open(path_to_html,'r') as f: 
    html_data = f.read()

## Show in webpage
st.components.v1.html(html_data,height=600)


st.markdown("""Looking at the routes, we can see that the client does not exercise in the north side of the river. He has mainly 2 preferences for exercising areas, either in the areas that look more urbanised (because there are many roads visible in the map) in the south side of the river or routes along the river. In most cases the riverside routes are longer so it looks like he stays closer to more populated areas for shorter distance sessions. It can also be seen that he has trained only 3 times in Ferry Field track that is in E Hoover Ave. so it does not look like this is one his favourite places for training. Regarding the heart rate, in general looks quite consistent but when zooming we can see that there are more differences than it initially would look like. When analysisng the routes against the training time, does not seem to be a clear pattern linking the sunlight and the location. This suggests that the client is comfortable exercising in the same areas during the day and during the night. From this observation we may assume that Ann Arbor is relatively safe because otherwise we would expect to see more clear differentiation between day sessions and night sessions.""", unsafe_allow_html=True)



        
        
        
        
        
        
        
        
