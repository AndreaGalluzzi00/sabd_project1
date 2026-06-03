FROM apache/spark:3.5.0

USER root

RUN pip install --no-cache-dir matplotlib==3.7.5 redis==5.0.8 scikit-learn==1.3.2 tdigest==0.5.2.2 datasketches==5.0.1 \
 && mkdir -p /opt/spark/matplotlib-cache \
 && chown spark:spark /opt/spark/matplotlib-cache

ENV SPARK_HOME=/opt/spark
ENV PATH="/opt/spark/bin:${PATH}"
ENV MPLCONFIGDIR=/opt/spark/matplotlib-cache

COPY conf/log4j2.properties /opt/spark/conf/log4j2.properties

USER spark