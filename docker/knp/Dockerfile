FROM yusanish/jumanpp_knp:latest

RUN pip install six && \
    wget http://nlp.ist.i.kyoto-u.ac.jp/nl-resource/knp/pyknp-0.3.tar.gz && \
    tar xvf pyknp-0.3.tar.gz && \
    cd pyknp-0.3 && \
    python setup.py install

WORKDIR /workspace