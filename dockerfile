FROM python:3.11-slim

RUN apt-get update -y && apt-get install -y \
    libgeos-dev \
    wget \
    build-essential

WORKDIR /code

# Download and install TA-Lib
RUN apt-get update -y && apt-get install -y \
    libgeos-dev \
    wget \
    build-essential \
    git \
    && wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib/ \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /ta-lib-0.4.0-src.tar.gz /ta-lib

# Copy only the requirements file, to cache the pip install step
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# install git
RUN apt-get install -y git

COPY . /code

EXPOSE 8080

CMD ["uvicorn", "deployments/deploy_model:app", "--host", "0.0.0.0", "--port", "8080"]