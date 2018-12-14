import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.path import Path
from pandas.plotting import table
import datetime as dt
from datetime import timezone
import numpy as np
import datetime as dt
import cartopy.feature
matplotlib.style.use('ggplot')
from scipy import interpolate
import sys
import logging
import urllib.request, json
from pandas.io.json import json_normalize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=[
        #logging.FileHandler("{0}/{1}.log".format(logPath, fileName)),
        logging.StreamHandler(sys.stdout)
    ])

logger = logging.getLogger()
now = dt.datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')


def main():
    plt.clf()

    with urllib.request.urlopen("http://metrics.af7ti.com:8080/stations.json") as url:
        data = json.loads(url.read().decode())
        print(data)

    df = json_normalize(data)
  
    print(df)

    #delete low confidence measurements
    df = df.drop(df[df.cs == 0].index)
    df = df.drop(df[df.mufd == 0].index)
    df = df.dropna(subset=['mufd'])

    #filter out data older than 1hr
    age = (dt.datetime.now() - dt.timedelta(minutes=60)).strftime('%Y-%m-%d %H:%M')
    df = df.loc[df['time'] > age]

    df['time'] = pd.to_datetime(df.time)
    df['time'] = df['time'].dt.strftime('%Y-%m-%d %H:%M')

    df[["mufd"]] = df[["mufd"]].apply(pd.to_numeric)
    df[['station.longitude']] = df[['station.longitude']].apply(pd.to_numeric)
    df[['station.latitude']] = df[['station.latitude']].apply(pd.to_numeric)
    df[['fof2']] = df[['fof2']].apply(pd.to_numeric)

    df = df.dropna(subset=['mufd'])

    df.ix[df['station.longitude'] > 180, 'station.longitude'] = df['station.longitude'] - 360
    df.sort_values(by=['station.longitude'], inplace=True)
   
    # grid data
    numcols, numrows = 100, 100
    xi = np.linspace(df['station.longitude'].min(), df['station.longitude'].max(), numcols)
    yi = np.linspace(df['station.latitude'].min(), df['station.latitude'].max(),numrows)
    xi, yi = np.meshgrid(xi, yi)
    # interpolate safe way with no extrapolation
    x, y, z = df['station.longitude'].values, df['station.latitude'].values, df.mufd.values
    rbf = interpolate.Rbf(x, y, z, function='linear')
    zi = rbf(xi, yi)
    
    
    #plot data
    
    fig = plt.figure(figsize=(16, 24))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    mycontour = plt.contourf(xi, yi, zi, 16,
                cmap = plt.cm.get_cmap("viridis"),
                transform=ccrs.PlateCarree(),
                alpha=0.33)
    
    #ax.coastlines()
    ax.add_feature(cartopy.feature.LAND)
    ax.set_global()
    
    ax.grid(linewidth=.5, color='black', alpha=0.25, linestyle='--')
    ax.set_xticks([-180, -160, -140, -120,-100, -80, -60,-40,-20, 0, 20, 40, 60,80,100, 120,140, 160,180], crs=ccrs.PlateCarree())
    ax.set_yticks([-80, -60,-40,-20, 0, 20, 40, 60,80], crs=ccrs.PlateCarree())
    
    #ax.set_extent([-147.07, 167.96, -51.6, 69.6], ccrs.PlateCarree())
    ax.set_extent([-180, 180, -51.6, 69.6], ccrs.PlateCarree())
    
    for index, row in df.iterrows():
      lon = float(row['station.longitude'])
      lat = float(row['station.latitude'])
      ax.text(lon, lat, int(row['mufd']), fontsize=10,ha='left', transform=ccrs.PlateCarree()) 
    
    plt.clabel(mycontour, inline=0, fontsize=10, fmt='%.0f')
    
    CS2 = plt.contour(mycontour, linewidths=.5, levels=mycontour.levels[1::1])
    
    # Make a colorbar for the ContourSet returned by the contourf call.
    cbar = plt.colorbar(mycontour, fraction=0.03, orientation='horizontal', pad=0.02)
    cbar.set_label('MHz')
    cbar.add_lines(CS2)
    
    plt.title("MUF(D) 3000km " + str(now) + ' @RealAF7TI')
   
    df = df[['station.name', 'time', 'mufd', 'cs', 'altitude', 'station.longitude', 'station.latitude', 'fof2', 'tec']]

    df = df.round(2)

    the_table = table(ax, df,
          bbox=[0,-1.35,1,1],
          cellLoc = 'left',)

    for key, cell in the_table.get_celld().items():
        cell.set_linewidth(.25)
 
    plt.tight_layout()
    plt.savefig('contour.png', dpi=300,bbox_inches='tight')


if __name__ == '__main__':
    main()
