FROM python:3.11-slim

RUN apt-get update -y && apt-get install -y \
    libgeos-dev \
    wget \
    build-essential

WORKDIR /code

# Download and install TA-Lib
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install

COPY ./requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# install git
RUN apt-get install -y git

COPY . /code

EXPOSE 8080

CMD ["uvicorn", "deployments/deploy_model:app", "--host", "0.0.0.0", "--port", "8080"]