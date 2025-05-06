from elasticsearch import Elasticsearch

es = Elasticsearch(
    ["https://localhost:9200"],
    basic_auth=("elastic", "O7h34vCodiAuGhFMcKuf"),
    verify_certs=False  
)





doc = {
    "title": "Introduction à Elasticsearch",
    "content": "Elasticsearch est un moteur de recherche distribué.",
    "date": "2025-05-06"
}



res = es.index(index='articles', document=doc)
print(res)


query_body = {
    "query": {
        "match": {
            "content": "recherche"
        }
    }
}

res = es.search(index="articles", body=query_body)

for hit in res["hits"]["hits"]:
    print(hit["_source"])
