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
from cartopy.feature.nightshade import Nightshade
#from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
matplotlib.style.use('ggplot')
import scipy
import os
import sys
import logging
import urllib.request, json
from pandas.io.json import json_normalize
import geojsoncontour
import statsmodels
import statsmodels.api as sm
import rbf


metric = sys.argv[1]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=[
        #logging.FileHandler("{0}/{1}.log".format(logPath, fileName)),
        logging.StreamHandler(sys.stdout)
    ])

logger = logging.getLogger()
now = dt.datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')
date = dt.datetime.now(timezone.utc) #.strftime('%Y, %m, %d, %H, %M')

def sph_to_xyz(lon, lat):
    lon = lon * np.pi / 180.
    lat = lat * np.pi / 180.
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)

    return x, y, z

def real_sph(m, n, theta, phi):
    if m == 0:
        return np.real(scipy.special.sph_harm(m, n, theta, phi))
    else:
        harm = scipy.special.sph_harm(abs(m), n, theta, phi)
        if m > 0:
            harm = np.real(harm)
        else:
            harm = np.imag(harm)

        odd_even = -1 if m % 2 else 1
        return np.sqrt(2) * odd_even * harm

def main():
    SPH_ORDER = 3
    SPH_WEIGHT = 0.8
    RESIDUAL_WEIGHT = 0.9

    plt.clf()

    with urllib.request.urlopen(os.getenv("METRICS_URI")) as url:
        data = json.loads(url.read().decode())

    df = json_normalize(data)
  
    #delete low confidence measurements
    df = df.drop(df[pd.to_numeric(df.cs) == 0].index)
    df = df.drop(df[df[metric] == 0].index)
    df = df.dropna(subset=[metric])

    #filter out data older than 1hr
    age = (dt.datetime.now() - dt.timedelta(minutes=60)).strftime('%Y-%m-%d %H:%M')
    df = df.loc[df['time'] > age]

    df['time'] = pd.to_datetime(df.time)
    df['time'] = df['time'].dt.strftime('%Y-%m-%d %H:%M')

    df[[metric]] = df[[metric]].apply(pd.to_numeric)
    df[['station.longitude']] = df[['station.longitude']].apply(pd.to_numeric)
    df[['station.latitude']] = df[['station.latitude']].apply(pd.to_numeric)
    df['longitude_radians'] = df['station.longitude'] * np.pi / 180.
    df['latitude_radians'] = (df['station.latitude'] + 90) * np.pi / 180.
    df[['cs']] = df[['cs']].apply(pd.to_numeric)
    df.loc[df['cs'] == -1, 'cs'] = 80
    df[['cs']] = df[['cs']] / 100.
    df['transformed'] = np.log(df[metric])

    df = df.dropna(subset=[metric])
    df.loc[df['station.longitude'] > 180, 'station.longitude'] = df['station.longitude'] - 360

    df.sort_values(by=['station.longitude'], inplace=True)

    sph = []
    alpha = []
    for n in range(SPH_ORDER):
        for m in range(0-n,n+1):
            sph.append(real_sph(m, n, df['longitude_radians'].values, df['latitude_radians'].values).reshape((-1,1)))
            alpha.append(0 if n == 0 else 0.005)
    sph = np.hstack(sph)

    wls_model = sm.WLS(df['transformed'].values, sph, df['cs'].values)
    wls_result = wls_model.fit_regularized(alpha=np.array(alpha), L1_wt = 0.6)
    coeff = wls_result.params

    numcols, numrows = 360, 180
    loni = np.linspace(-180, 180, numcols)
    lati = np.linspace(-90, 90, numrows)

    theta = loni * np.pi / 180.
    phi = (lati + 90) * np.pi / 180.

    zi = np.zeros((len(phi),len(theta)))
    theta, phi = np.meshgrid(theta, phi)

    df['pred'] = np.zeros(len(df))

    coeff_idx = 0
    for n in range(SPH_ORDER):
        for m in range(0-n,n+1):
            sh = real_sph(m, n, theta, phi)
            weight = 1 if n == 0 else SPH_WEIGHT
            zi = zi + weight * coeff[coeff_idx] * sh
            df['pred'] = df['pred'] + weight * np.real(coeff[coeff_idx] * real_sph(m, n, df['longitude_radians'].values, df['latitude_radians'].values))
            coeff_idx = coeff_idx + 1

    df['residual'] = df['transformed'] - df['pred']
    #plot data
    
    loni, lati = np.meshgrid(loni, lati)
    x, y, z = sph_to_xyz(df['station.longitude'].values, df['station.latitude'].values)
    t = df['residual'].values

    stdev = 0.7 - 0.5 * df['cs']

    gp = rbf.gauss.gpiso(rbf.basis.se, (0.0, 0.7, 0.8))
    gp_cond = gp.condition(np.vstack((x,y,z)).T, t, sigma=stdev)

    xxi, yyi, zzi = sph_to_xyz(loni, lati)
    xyz = np.array([xxi.flatten(), yyi.flatten(), zzi.flatten()]).T
    resi, sd = gp_cond.meansd(xyz)
    resi = resi.reshape(xxi.shape)
    sd = sd.reshape(xxi.shape)

    zi = zi + RESIDUAL_WEIGHT * resi
    zi = np.exp(zi)
    
    fig = plt.figure(figsize=(16, 24))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    levels = 16
    cmap = plt.cm.get_cmap('viridis')
    cmap.set_under(cmap(1e-5))
    cmap.set_over(cmap(1 - 1e-5))
    norm = matplotlib.colors.Normalize(clip=False)

    if metric == 'mufd':
        levels = [3, 3.5, 4, 4.6, 5.3, 6.1, 7, 8.2, 9.5, 11, 12.6, 14.6, 16.9, 19.5, 22.6, 26, 30]
        norm = matplotlib.colors.LogNorm(3.5,30, clip=False)

    mycontour = plt.contourf(loni, lati, zi, levels,
                cmap=cmap,
                extend='both',
                transform=ccrs.PlateCarree(),
                alpha=0.3,
                norm=norm
                )
    
    ax.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '110m',
        edgecolor='face',
        facecolor=np.array((0xdd,0xdd,0xcc))/256.,
        zorder=-1
        )
        )
    ax.set_global()
    ax.add_feature(Nightshade(date, alpha=0.08))

    ax.grid(linewidth=.5, color='black', alpha=0.25, linestyle='--')
    ax.set_xticks([-180, -160, -140, -120,-100, -80, -60,-40,-20, 0, 20, 40, 60,80,100, 120,140, 160,180], crs=ccrs.PlateCarree())
    ax.set_yticks([-80, -60,-40,-20, 0, 20, 40, 60,80], crs=ccrs.PlateCarree())
    
    for index, row in df.iterrows():
      lon = float(row['station.longitude'])
      lat = float(row['station.latitude'])
      alpha = 0.2 + 0.6 * row['cs']
      ax.text(lon, lat, int(row[metric] + 0.5),
              fontsize=9,
              ha='left',
              transform=ccrs.PlateCarree(),
              alpha=alpha,
              bbox={
                  'boxstyle': 'circle',
                  'alpha': alpha - 0.1,
                  'color': cmap(norm(row[metric])),
                  'mutation_scale': 0.5,
                  }
              )
    
    CS2 = plt.contour(mycontour, linewidths=.5, alpha=0.66, levels=mycontour.levels[1::1])

    prev = None
    levels = []
    for lev in CS2.levels:
        if prev is None or '%.0f'%(lev) != '%.0f'%(prev):
            levels.append(lev)
            prev = lev

    plt.clabel(CS2, levels, inline=True, fontsize=10, fmt='%.0f', use_clabeltext=True )
    
    # Make a colorbar for the ContourSet returned by the contourf call.
    cbar = plt.colorbar(mycontour, fraction=0.03, orientation='horizontal', pad=0.02, format=matplotlib.ticker.ScalarFormatter())
    #cbar.set_label('MHz') #TODO add unit
    cbar.add_lines(CS2)
    
    plt.title(metric + ' ' + str(now))
   
#    df = df[['station.name', 'time', metric, 'cs', 'altitude', 'station.longitude', 'station.latitude']]

#    df = df.round(2)

#    the_table = table(ax, df,
#          bbox=[0,-1.25,1,1],
#          cellLoc = 'left',)

#    for key, cell in the_table.get_celld().items():
#        cell.set_linewidth(.25)
 
    plt.tight_layout()
    plt.savefig('/output/{}.png'.format(metric), dpi=180,bbox_inches='tight')
    plt.savefig('/output/{}.svg'.format(metric), dpi=180,bbox_inches='tight')
    
    # Convert matplotlib contour to geojson
    """
    geojsoncontour.contourf_to_geojson(
        contourf=mycontour,
        geojson_filepath='/output/{}.geojson'.format(metric),
        min_angle_deg=3.0,
        ndigits=2,
        stroke_width=2,
        fill_opacity=0.5,
        )
    """


if __name__ == '__main__':
    main()
