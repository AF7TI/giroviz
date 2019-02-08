# giroviz
Cartopy visualization environment for [giroapp](https://github.com/AF7TI/giroapp)

## Installation
Build image from Dockerfile, tag with giroviz
    `docker build -t giroviz .`

## Usage
Provide contour.py a data source and metric to plot. A png and geojson file is saved in the current directory:
    `docker run -e "METRICS_URI=http://metrics.af7ti.com/stations.json" -v $(pwd):/output giroviz python contour.py mufd`
    
See available metrics in the [measurement table schema](https://github.com/AF7TI/girotick/blob/master/dbsetup.sql)

## Running Code
- KC2G aka @arodland has MUFD and foF2 maps at https://prop.kc2g.com/
- Data sources online at https://prop.kc2g.com/stations.json and http://metrics.af7ti.com/stations.json

## Contributing
Contributions welcome! Please fork and open a pull request.

## Thank you
Huge thanks to [@arodland](https://github.com/arodland/giroviz) for incorporating spherical geometry and many other improvements.

Data made available through [UMass Lowell Global Ionospheric Radio observatory (GIRO)](http://umlcar.uml.edu/DIDBase/RulesOfTheRoadForDIDBase.htm)
