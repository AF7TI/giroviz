"""
This example demonstrates adding tick labels to maps on rectangular
projections using special tick formatters.

"""
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.path import Path
from pandas.plotting import table
import datetime as dt
import numpy as np
import datetime as dt
import cartopy.feature
#from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
matplotlib.style.use('ggplot')
from scipy import interpolate
#import scipy
import sys
import psycopg2
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=[
        #logging.FileHandler("{0}/{1}.log".format(logPath, fileName)),
        logging.StreamHandler(sys.stdout)
    ])

logger = logging.getLogger()

try:
    print("connecting to db . . . ")
    con = psycopg2.connect("dbname='postgres' user='lilpurpp' host='girodb.af7ti.com' password='esskeetit'")
except:
    logger.error("Unable to connect to the database")


def main():
    plt.clf()
    df = pd.read_sql('SELECT * from station', con)
    data = pd.read_sql('SELECT tt.* FROM measurement tt INNER JOIN (SELECT station_id, MAX(time) AS MaxDateTime FROM measurement GROUP BY station_id) groupedtt ON tt.station_id = groupedtt.station_id AND tt.time = groupedtt.MaxDateTime', con, parse_dates=True)
    df = df.merge(data, left_on='id', right_on='station_id', how='inner')
    print(df)
    
    #delete low confidence measurements
    df = df.drop(df[df.cs == 0].index)
    df = df.drop(df[df.mufd == 0].index)
    df = df.dropna(subset=['mufd'])


    df['time'] = pd.to_datetime(df.time)
    df['time'] = df['time'].dt.strftime('%Y-%m-%d %H:%M')
    
    df[["mufd"]] = df[["mufd"]].apply(pd.to_numeric)

    df = df.dropna(subset=['mufd'])

    df.ix[df.longitude > 180, 'longitude'] = df.longitude - 360
    df.longitude = df.longitude.round(2)
    df.sort_values(by=['longitude'], inplace=True)
    plt.figure(figsize=(8, 6))

    # Label axes of a Plate Carree projection with a central longitude of 180:
    ax1 = plt.subplot(211,projection=ccrs.PlateCarree(central_longitude=0), frame_on=False)
    #ax1.set_extent([-147.07, 167.96, -51.6, 69.6], crs=ccrs.PlateCarree())
    #ax1.set_extent([-20, 60, -40, 45], crs=ccrs.PlateCarree())
    #ax1.set_global()
    ax1.add_feature(cartopy.feature.LAND, facecolor = [0.78, 0.78, 0.78])
    #ax1.gridlines()
    #ax1.coastlines()
    
    #ax1.xformatter = LONGITUDE_FORMATTER
    #ax1.yformatter = LATITUDE_FORMATTER

    ax1.grid(linewidth=.5, color='black', alpha=0.25, linestyle='--')
    ax1.set_xticks([0, 60, 120, 180, 240, 300, 360], crs=ccrs.PlateCarree())
    ax1.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    ax1.set_extent([-158.15, 169.93, -51.6, 69.6], crs=ccrs.PlateCarree())
    # prep increasing values of v covering values of Z (matrixTec)
    v = np.arange(-0.15, 0.15, 0.025)
 
    numcols, numrows = 100, 100
    xi = np.linspace(df['longitude'].min(), df['longitude'].max(), numcols)
    yi = np.linspace(df['latitude'].min(), df['latitude'].max(),numrows)
    xi, yi = np.meshgrid(xi, yi)


    rbf = interpolate.Rbf(df["longitude"].values, df["latitude"].values, df["mufd"].values, function='linear')
    zi = rbf(xi, yi)

    contour = plt.contourf(xi, yi, zi, 32,
             zorder=3, transform=ccrs.PlateCarree(),
             alpha=0.4,
             vmin=df['mufd'].min(), vmax=df['mufd'].max(),
   )
    
    cbar = plt.colorbar(contour, orientation='horizontal', fraction=.05, pad=.1, format='%1i')
    cbar.ax.tick_params(labelsize=6) 

    for index, row in df.iterrows():
      lon = float(row['longitude'])
      lat = float(row['latitude'])
      ax1.text(lon, lat, row['mufd'], fontsize=6,ha='left') 

    df = df[['name', 'time', 'mufd', 'cs', 'altitude', 'longitude', 'latitude', 'fof2', 'tec']]
    df.rename(index=str, columns={'mufd_float' : 'mufd'}, inplace=True) 
    # build a rectangle in axes coords
    left, width =0, 2
    bottom, height = 0, 2
    right = left + width
    top = bottom + height 

    plt.title('MUF(D) 3000km ' + dt.datetime.now().strftime('%Y-%m-%d %H:%M') + ' @RealAF7TI' , fontsize=8)
    ax1.text(left, bottom, 'Data Global Ionospheric Radio Observatory (GIRO) Principal Investigator Prof. B. W. Reinisch UMass Lowell', horizontalalignment='left',verticalalignment='bottom',transform=ax1.transAxes, size='5')

    df['mufd'].astype(int)
    the_table = table(ax1, df,
          bbox=[0,-1.35,1,1],
          cellLoc = 'left',)

    for key, cell in the_table.get_celld().items():
    	cell.set_linewidth(.25)

    plt.setp(ax1.get_xticklabels(), fontsize=6)
    plt.setp(ax1.get_yticklabels(), fontsize=6)
    the_table.set_fontsize(8)
    

    plt.savefig('contour.png', dpi=300,bbox_inches='tight')


if __name__ == '__main__':
    main()
