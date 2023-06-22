import branca
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import folium
import json
import plotly

from plotly import __version__
import plotly.graph_objs as go 
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from plotly.subplots import make_subplots

from flask import Flask, request, render_template

import plotly.express as px

from folium.plugins import MarkerCluster
import folium.plugins as plugins
import branca

app = Flask(__name__)
import pickle
import numpy as np
import pandas as pd

data = np.load('dataset.npy', allow_pickle='true')
cr = pd.DataFrame(data)
cr.columns = ['INCIDENT_NUMBER', 'OFFENSE_CODE', 'OFFENSE_CODE_GROUP',
       'OFFENSE_DESCRIPTION', 'DISTRICT', 'REPORTING_AREA', 'SHOOTING',
       'OCCURRED_ON_DATE', 'YEAR', 'MONTH', 'DAY_OF_WEEK', 'HOUR', 'UCR_PART',
       'STREET', 'Lat', 'Long', 'Location']

cr1 = cr.copy()
tmp = cr1.groupby('INCIDENT_NUMBER')['YEAR'].count().sort_values(ascending = False)
tmp = pd.DataFrame({'INCIDENT_NUMBER': tmp.index, 'NUM_RECORDS': tmp.values})
seriousCrimes = cr.merge(tmp[tmp['NUM_RECORDS'] > 2 ], on = 'INCIDENT_NUMBER', how = 'inner')
seriousCrimes = seriousCrimes[['Lat','Long','OFFENSE_CODE_GROUP']].dropna()

cr["SHOOTING"].fillna("N", inplace = True)

cr.Lat.replace(-1, None, inplace=True)
cr.Long.replace(-1, None, inplace=True)

cr.apply(pd.Series.nunique)

cr.drop_duplicates(subset="INCIDENT_NUMBER", inplace=True)

#For sorting weekdays in correct order a key is created
m = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

#The data of total crime values
crimes_per_year = pd.DataFrame(data=cr['YEAR'].value_counts().reset_index().values, columns=["YEAR","CRIME COUNT"]).sort_values('YEAR').reset_index(drop=True)
crimes_per_month = pd.DataFrame(data=cr['MONTH'].value_counts().reset_index().values, columns=["MONTH","CRIME COUNT"]).sort_values('MONTH').reset_index(drop=True)
crimes_per_day = pd.DataFrame(data=cr['DAY_OF_WEEK'].value_counts().reset_index().values, columns=["DAY","CRIME COUNT"])
crimes_per_day["DAY"] = pd.Categorical(crimes_per_day['DAY'], categories=m, ordered=True)
crimes_per_day = crimes_per_day.sort_values('DAY').reset_index(drop=True)
crimes_per_hour = pd.DataFrame(data=cr['HOUR'].value_counts().reset_index().values, columns=["HOUR","CRIME COUNT"]).sort_values('HOUR').reset_index(drop=True)
crimes_per_district = pd.DataFrame(data=cr['DISTRICT'].value_counts().reset_index().values, columns=["DISTRICT","CRIME COUNT"])
crimes_per_street = pd.DataFrame(data=cr['STREET'].value_counts().reset_index().values, columns=["STREET","CRIME COUNT"]).sort_values('CRIME COUNT',ascending=False).reset_index(drop=True).head(50)

#Data for total crime counts for each UCR Parts
ucr_year = pd.DataFrame(data =(cr.groupby(["YEAR","UCR_PART"]).count()[['INCIDENT_NUMBER']]).reset_index().values, columns=["YEAR","UCR_PART","CRIME COUNT"]).sort_values('YEAR').reset_index(drop=True)
ucr_month = pd.DataFrame(data =(cr.groupby(["MONTH","UCR_PART"]).count()[['INCIDENT_NUMBER']]).reset_index().values, columns=["MONTH","UCR_PART","CRIME COUNT"]).sort_values('MONTH').reset_index(drop=True)
ucr_day = pd.DataFrame(data =(cr.groupby(["DAY_OF_WEEK","UCR_PART"]).count()[['INCIDENT_NUMBER']]).reset_index().values, columns=["DAY","UCR_PART","CRIME COUNT"]).sort_values('DAY').reset_index(drop=True)
ucr_day["DAY"] = pd.Categorical(ucr_day['DAY'], categories=m, ordered=True)
ucr_day = ucr_day.sort_values('DAY').reset_index(drop=True)
ucr_hour = pd.DataFrame(data =(cr.groupby(["HOUR","UCR_PART"]).count()[['INCIDENT_NUMBER']]).reset_index().values, columns=["HOUR","UCR_PART","CRIME COUNT"]).sort_values('HOUR').reset_index(drop=True)
ucr_district = pd.DataFrame(data =(cr.groupby(["DISTRICT","UCR_PART"]).count()[['INCIDENT_NUMBER']]).reset_index().values, columns=["DISTRICT","UCR_PART","CRIME COUNT"])
ucr_street = pd.DataFrame(data =(cr.groupby(["STREET","UCR_PART"]).count()[['INCIDENT_NUMBER']]).reset_index().values, columns=["STREET","UCR_PART","CRIME COUNT"]).sort_values('CRIME COUNT',ascending=False).reset_index(drop=True).head(50)

offense_code_count = pd.DataFrame(cr['OFFENSE_CODE_GROUP'].value_counts().reset_index().values, columns=["OFFENSE_CODE_GROUP'","CRIME COUNT"])

model = pickle.load(open('ethos.pkl', 'rb'))

@app.route('/', methods =  ['GET'])
def Index():
    return render_template('index.html') 

@app.route('/index.html', methods =  ['GET'])
def About(): 
    return render_template('index.html') 
    
@app.route('/Predictor.html', methods = ['POST', 'GET'])
def predict():
    if request.method == 'POST':
        lat = (request.form.get("latitude"))
        long = (request.form.get("longitude"))
        print(lat,long)
        inp = np.array((lat, long)).reshape(1, -1)
        output = model.predict(inp)[0]
        if(output == 0):
            output = "Accident"
        elif(output == 1):
            output = "Drugs Violation"
        elif(output == 2):
            output = "Violations"
        elif(output == 3):
            output = "Assault"
        elif(output == 4):
            output = "Larceny"
        return render_template('Predictor.html', prediction = output)
    else:
        return render_template('Predictor.html', prediction = "")


@app.route('/plot1', methods =  ['GET'])
def fig1():
    fig = make_subplots(
    rows=2, cols=3,specs=[[{"type": "scatter"}, {"type": "scatter"},{"type": "scatter"}],[{"type": "scatter"},{"type": "bar"}, {"type": "bar"}]],
    subplot_titles=("Number of crimes per year", "Number of crimes per month", "Number of crimes per day", 
            "Number of crimes per hour","Number of crimes per district","Number of crimes per streets (Top 50)"))
    
    # Add traces
    fig.add_trace(go.Scatter(x=crimes_per_year["YEAR"], y=crimes_per_year["CRIME COUNT"]), row=1, col=1)
    fig.add_trace(go.Scatter(x=crimes_per_month["MONTH"], y=crimes_per_month["CRIME COUNT"]), row=1, col=2)
    fig.add_trace(go.Scatter(x=crimes_per_day["DAY"], y=crimes_per_day["CRIME COUNT"]), row=1, col=3)
    fig.add_trace(go.Scatter(x=crimes_per_hour["HOUR"], y=crimes_per_hour["CRIME COUNT"]), row=2, col=1)
    fig.add_trace(go.Bar(x=crimes_per_district["DISTRICT"], y=crimes_per_district["CRIME COUNT"]), row=2, col=2)
    fig.add_trace(go.Bar(x=crimes_per_street["STREET"], y=crimes_per_street["CRIME COUNT"]), row=2, col=3)
    
    # Update xaxis properties
    fig.update_xaxes(title_text="Year", row=1, col=1)
    fig.update_xaxes(title_text="Month", range=[0, 13], row=1, col=2)
    fig.update_xaxes(title_text="Day", row=1, col=3)
    fig.update_xaxes(title_text="Hour",row=2, col=1)
    fig.update_xaxes(title_text="District", row=2, col=2)
    fig.update_xaxes(title_text="Street", row=2, col=3)
    
    # Update yaxis properties
    fig.update_yaxes(title_text="Crime Count", row=1, col=1)
    fig.update_yaxes(title_text="Crime Count",row=1, col=2)
    fig.update_yaxes(title_text="Crime Count", row=1, col=3)
    fig.update_yaxes(title_text="Crime Count", row=2, col=1)
    fig.update_yaxes(title_text="Crime Count", row=2, col=2)
    fig.update_yaxes(title_text="Crime Count", row=2, col=3)
    
    # Update title
    fig.update_layout(showlegend=False,title_text="Distributions of Total Number of Crimes Between 2015-2018")

    return fig._repr_html_()

@app.route('/plot2', methods =  ['GET'])
def fig2():
    df = pd.DataFrame(data =(cr.groupby(["YEAR","UCR_PART",'OFFENSE_CODE_GROUP']).count()[['INCIDENT_NUMBER']]).reset_index().values, columns=["YEAR","UCR_PART",'OFFENSE_CODE_GROUP',"CRIME COUNT"])
    fig = px.sunburst(df, path=['YEAR', 'UCR_PART', 'OFFENSE_CODE_GROUP'], values='CRIME COUNT')

    return fig._repr_html_()

@app.route('/plot3', methods =  ['GET'])
def fig3():
    ucr_offense = pd.DataFrame(data=(cr.groupby(["UCR_PART", "OFFENSE_CODE_GROUP"]).count()[['INCIDENT_NUMBER']]).reset_index(
    ).values, columns=["UCR_PART", "OFFENSE_CODE_GROUP", "CRIME COUNT"]).sort_values('UCR_PART').reset_index(drop=True)


    ucr_offense['all'] = 'all'
    fig = px.treemap(ucr_offense, path=['all', 'UCR_PART', 'OFFENSE_CODE_GROUP'],
                 values='CRIME COUNT', color='CRIME COUNT', color_continuous_scale='RdBu')
# fig.show()
    return fig._repr_html_()

@app.route('/plot4', methods =  ['GET'])
def fig4():
    crime_loc = pd.DataFrame(data =(cr.groupby(['DISTRICT',"Lat","Long","OFFENSE_CODE_GROUP"]).count()[['INCIDENT_NUMBER']]).reset_index().values,columns=['DISTRICT',"Lat","Long","OFFENSE_CODE_GROUP","CRIME COUNT"])
    fig = px.scatter(crime_loc[crime_loc['Lat']!=-1], x="Lat", y="Long", animation_frame="OFFENSE_CODE_GROUP", animation_group="CRIME COUNT",
    color="DISTRICT")
    fig["layout"].pop("updatemenus") 
    return fig._repr_html_()

@app.route('/map1', methods =  ['GET'])
def fig5():
    location = pd.DataFrame(data =(cr.groupby(["Lat","Long"]).count()[['INCIDENT_NUMBER']]).reset_index().values, columns=["Lat","Long","CRIME COUNT"])
    x,y = location['Long'], location['Lat']
    fig = px.density_mapbox(location,lat="Lat",lon="Long",z="CRIME COUNT",radius=10,center=dict(lat=42.357791, lon=-71),zoom = 10,mapbox_style="stamen-terrain")
    return fig._repr_html_()

@app.route('/map2', methods =  ['GET'])
def fig6():
    f = folium.Figure()
    colorscale = branca.colormap.linear.YlGnBu_09.scale(0, 61)
    boston_map = folium.Map(location = [seriousCrimes['Lat'].mean(), 
                                        seriousCrimes['Long'].mean()], 
                      zoom_start = 11).add_to(f)

    incidents2=plugins.MarkerCluster().add_to(boston_map)
    for lat,lon,label in zip(seriousCrimes.Lat,seriousCrimes.Long,seriousCrimes.OFFENSE_CODE_GROUP):
        folium.Marker(location=[lat,lon],icon=None,popup=label).add_to(incidents2)

    boston_map.add_child(incidents2)
    return boston_map._repr_html_()

@app.route('/Analytics.html', methods =  ['GET'])
def Analytics():
    return render_template('Analytics.html')

@app.route('/References.html', methods =  ['GET'])
def Reference():
    return render_template('References.html')   

@app.route('/Apphome.html', methods =  ['GET'])
def Apphome():
    return render_template('Apphome.html')   

@app.route('/GeoAnalytics.html', methods =  ['GET'])
def lifeExp():
    return render_template('GeoAnalytics.html')   

@app.route('/demo.html', methods =  ['GET'])
def demo():
    return render_template('demo.html')

@app.route('/Crime1.html', methods =  ['GET'])
def Crime1():
    return render_template('Crime1.html')

@app.route('/Crime2.html', methods =  ['GET'])
def Crime2():
    return render_template('Crime2.html')

if __name__=="__main__":
    app.run(debug=True)

