# giroviz
cartopy visualization environment for giro data at metrics.af7ti.com

build image from Dockerfile, tag with giroviz
    `docker build -t giroviz .`
    
provide contour.py a metric to plot. a png file is saved in the current directory:
    `docker run -v $(pwd):/output xxxxx python contour.py mufd`
    
see available metrics in the [measurement table schema] https://github.com/AF7TI/girotick/blob/master/dbsetup.sql
    
