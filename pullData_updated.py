from pytz import timezone
from elasticsearch7 import Elasticsearch
import json
import dateutil.parser
import datetime

OutputFileName = "news_article.json"

# initialise ES connection
esConn = Elasticsearch(['http://10.90.20.41:9200'], http_auth=('elastic', 'ElasticsearchDSA@2021'))

# define the query parameter
query = {
    # match all data in the database
    "size": 10000, "query": {"match_all": {}}
}

# sample for query to match time bound and filter
# currentDate = dateutil.parser.parse("%sT00:00:00+8" % datetime.datetime.now().strftime("%Y-%m-%d"))
# lastDate = "2022-01-01T00:00:00+8"
# query = {
#     "size": 10000, 
#     "query": {
#         "bool": {
#             "must": [
#                 {
#                     "range": {
#                         "published_date": {
#                             "lt": currentDate, # less than the date
#                             "gte": lastDate, # greater or equal than the date
#                         }
#                     }
#                 },
#                 {
#                     # filter by the news source
#                     "terms": {
#                         "source": ["The Star", "Malay Mail"]
#                     }
#                 },
#                 {
#                     # filter by the section
#                     "terms": {
#                         "section": ["business"]
#                     }
#                 }
#             ]
#         }
#     }
# }

# search the elasticsearch using "scroll" features to download all data by multiple chunk
res = esConn.search(index="panoptes_articles", scroll='10s', body=query)

# open flat file to output as json data
outputBase = open(OutputFileName, "w+", encoding="utf-8")

# getting the current scroll id
sid = res['_scroll_id']

count = 0
# loop until there's no more data return from ES
while len(res['hits']['hits']):
    print ("Current data chunk size = %d" % (len(res['hits']['hits'])))

    count += len(res['hits']['hits'])
    for each in res['hits']['hits']:
        # write each result as json string
        outputBase.write(json.dumps(each['_source']))
        outputBase.write(("\n"))
        outputBase.flush()

    # request the next segment/chunk
    res = esConn.scroll(scroll_id = sid, scroll = '10s')
    if sid != res['_scroll_id']:
        # Update the scroll ID
        sid = res['_scroll_id']

print ("Total data: %d" % (count))
# closed the output file
outputBase.close()

######################################################################
# Consuming the output data
######################################################################
# inputFile = open(OutputFileName, "r+")

# curLine = inputFile.readline()

# while curLine:
#     # convert to json object
#     curJson = json.loads(curLine)

#     # access the json object
#     curUuid = curJson['uuid']
#     title = curJson['title']
#     content = curJson['content']

#     # TODO: process data here

#     # read the next line
#     curLine = inputFile.readline()

# inputFile.close()