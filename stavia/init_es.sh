#!/usr/bin/env bash
curl -X DELETE "localhost:9200/smart_address_tf"
curl -X PUT "localhost:9200/smart_address_tf" -H 'Content-Type: application/json' -d'
{   "settings":{
    "number_of_shards": 3,
    "index" : {
            "similarity" : {
                "scripted_tfidf": {
                    "type": "scripted",
                        "script": {
                          "source": "double tf = Math.sqrt(doc.freq);  double norm = 1/Math.sqrt(doc.length); return query.boost * tf * norm;"
                        }
                 }
            }
        }
    },
    "mappings" : {
        "doc" : {
            "properties" : {
                "project" : {
                    "type" : "text",
                    "similarity" : "scripted_tfidf"
                },
                "street" : {
                    "type" : "text",
                    "similarity" : "scripted_tfidf",
                    "boost" : 3
                 },
                 "street_no" : {
                    "type" : "text",
                    "similarity" : "scripted_tfidf",
                    "boost" : 3
                 },
                "ward" : {
                    "type" : "text",
                    "similarity" : "scripted_tfidf",
                     "boost" : 2
                 },
                 "ward_no" : {
                    "type" : "text",
                    "similarity" : "scripted_tfidf",
                     "boost" : 2
                 },
                "district" : {
                    "type" : "text",
                    "similarity" : "scripted_tfidf",
                      "boost" : 2
                      },
                 "district_no" : {
                    "type" : "text",
                    "similarity" : "scripted_tfidf",
                      "boost" : 2
                      },
                 "city" : {
                    "type" : "text",
                    "similarity" : "scripted_tfidf",
                     "boost" : 2
                      },
                 "city_no" : {
                    "type" : "text",
                    "similarity" : "scripted_tfidf",
                     "boost" : 2
                      },
                "country" : {
                    "type" : "text"
                     },
                "code" : { "type" : "text" },
                "lat" : { "type" : "double" },
                "lng" : { "type" : "double" }
                }
            }
        }
}
'
curl -H 'Content-Type: application/x-ndjson' -XPOST 'localhost:9200/smart_address_tf/doc/_bulk?pretty' --data-binary @$1
#!/usr/bin/env bash