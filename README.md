# Stavia

An python client on Vietnamese Address Standardization problem
This tutorial was writen for ```MacOSX``` or ```Linux```. To run on ```Window```, please find corresponding commands.

### How it works

![alt tag](https://github.com/ngocjr7/stavia/blob/llm/dos/howitworks.png)

### Requirements

##### Python
It requires ```python``` (or ```python3```), ```pip```.
Install required packages:
```sh
pip install -r requirements.txt
```

##### Elasticsearch
Download Elasticsearch [here](https://www.elastic.co/downloads/elasticsearch). and extract it.
On Elasticsearch home folder, run this command (This command open a server which is used by our service, hold it on, more information [here](https://www.elastic.co/guide/en/elasticsearch/reference/current/starting-elasticsearch.html)):

```sh
./bin/elasticsearch
```

On ```Linux``` you can use ```systemctl``` instead
```sh
systemctl start elasticsearch.service
```

### Testing

```sh
python test.py
```

```test.py```

```python
# -*- encoding: utf-8 -*-
import stavia
print(stavia.standardize('vinh lai phu tho'))
```