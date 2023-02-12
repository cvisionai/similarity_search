FROM cvisionai/openem_lite:latest
RUN apt update
RUN pip3 install redis configargparse
COPY ../similarity_search /grafit
WORKDIR /grafit
RUN pip3 install .
COPY grafit_single_inference.py /scripts
WORKDIR /scripts
