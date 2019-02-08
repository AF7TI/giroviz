FROM continuumio/miniconda3

RUN apt-get update && apt-get -y install unzip build-essential

WORKDIR /
RUN wget -O /tmp/ne_110m_land.zip http://naciscdn.org/naturalearth/110m/physical/ne_110m_land.zip
WORKDIR /root/.local/share/cartopy/shapefiles/natural_earth/physical
RUN unzip /tmp/ne_110m_land.zip && rm /tmp/ne_110m_land.zip

RUN conda create -y -q -n my_cartopy_env -c conda-forge python=3.7 cartopy statsmodels pandas cython xarray sympy networkx

ENV PATH /opt/conda/envs/my_cartopy_env/bin:$PATH

RUN echo "conda activate my_cartopy_env" >> ~/.bashrc

RUN pip install geojsoncontour

WORKDIR /src
RUN git clone http://github.com/treverhines/RBF.git
WORKDIR /src/RBF
RUN python setup.py install

WORKDIR /
COPY contour*.py /
