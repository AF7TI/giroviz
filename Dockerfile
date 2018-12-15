FROM continuumio/miniconda3

RUN conda create -y -q -n my_cartopy_env -c conda-forge cartopy

ENV PATH /opt/conda/envs/my_cartopy_env/bin:$PATH

RUN echo "conda activate my_cartopy_env" >> ~/.bashrc

RUN conda install -y -q pandas -n my_cartopy_env

RUN pip install geojsoncontour

COPY contour.py /
