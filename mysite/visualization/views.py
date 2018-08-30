from django.shortcuts import render

# Create your views here.

import json
from django.shortcuts import render
from visualization.extraction.extraction import EnPreprocess


def index(request):
    enpre = EnPreprocess()
    results = enpre.enpre_main('visualization/extraction/movie_star_imdb')

    # 数据：word-probability
    topics_json = []
    for key, value in results[0].items():
        topic_json = {}
        topic_json["name"] = key
        topic_json["value"] = value
        topics_json.append(topic_json)

    # 数据：word-doc-count
    topics_info_json = []
    for topic_dict in results[1]:
        topic_info = {}
        topic_info["word"] = topic_dict['w']
        topic_info["name"] = topic_dict['doc']
        topic_info["value"] = topic_dict['count']
        topics_info_json.append(topic_info)

    # 比例：word-doc-percent
    topics_percent_json = []
    for topic_dict in results[1]:
        topic_percent = {}
        count_sum = sum(topic_dict['count'])
        topic_percent["word"] = topic_dict['w']
        topic_percent["name"] = topic_dict['doc']
        topic_percent["value"] = [float('%.2f' % (n/count_sum)) for n in topic_dict['count']]
        topics_percent_json.append(topic_percent)

    # 文件名
    files_name_list = results[2]

    return render(request, 'index.html', {'topics_json': json.dumps(topics_json),
                                          'topics_info_json': json.dumps(topics_info_json),
                                          'files_name_list': files_name_list,
                                          'topics_percent_json': json.dumps(topics_percent_json)})

