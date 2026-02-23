<span id="bdae7a9d"></span>
# 流程简介
大模型录音文件识别服务的处理流程分为提交任务和查询结果两个阶段
任务提交：提交音频链接，并获取服务端分配的任务 ID
结果查询：通过任务 ID 查询转写结果

<div style="text-align: center"><img src="https://p9-arcosite.byteimg.com/tos-cn-i-goo7wpa0wc/b2db50e28e304214bcfd2c9e7f5993ae~tplv-goo7wpa0wc-image.image" width="368px" /></div>


<span id="8a0814b2"></span>
# 提交任务
<span id="45358956"></span>
## 接口地址
火山地址：https://openspeech.bytedance.com/api/v3/auc/bigmodel/submit
<span id="979caf11"></span>
## 请求
请求方式：HTTP POST。
请求和应答，均采用在 HTTP BODY 里面传输 JSON 格式字串的方式。
Header 需要加入内容类型标识：

| | | | \
|Key |说明 |Value 示例 |
|---|---|---|
| | | | \
|X-Api-App-Key |使用火山引擎控制台获取的APP ID，可参考 [控制台使用FAQ-Q1](/docs/6561/196768#q1%EF%BC%9A%E5%93%AA%E9%87%8C%E5%8F%AF%E4%BB%A5%E8%8E%B7%E5%8F%96%E5%88%B0%E4%BB%A5%E4%B8%8B%E5%8F%82%E6%95%B0appid%EF%BC%8Ccluster%EF%BC%8Ctoken%EF%BC%8Cauthorization-type%EF%BC%8Csecret-key-%EF%BC%9F) |123456789 |
| | | | \
|X-Api-Access-Key |使用火山引擎控制台获取的Access Token，可参考 [控制台使用FAQ-Q1](/docs/6561/196768#q1%EF%BC%9A%E5%93%AA%E9%87%8C%E5%8F%AF%E4%BB%A5%E8%8E%B7%E5%8F%96%E5%88%B0%E4%BB%A5%E4%B8%8B%E5%8F%82%E6%95%B0appid%EF%BC%8Ccluster%EF%BC%8Ctoken%EF%BC%8Cauthorization-type%EF%BC%8Csecret-key-%EF%BC%9F) |your-access-key |
| | | | \
|X-Api-Resource-Id |表示调用服务的资源信息 ID |豆包录音文件识别模型1.0 |\
| | | |\
| | |* volc.bigasr.auc |\
| | | |\
| | |豆包录音文件识别模型2.0 |\
| | | |\
| | |* volc.seedasr.auc |
| | | | \
|X-Api-Request-Id |用于提交和查询任务的任务ID，推荐传入随机生成的UUID |67ee89ba-7050-4c04-a3d7-ac61a63499b3 |
| | | | \
|X-Api-Sequence |发包序号，固定值，-1 | |

<span id="989b21ce"></span>
### 请求字段

| | | | | | | \
|字段 |说明 |层级 |格式 |是否必填 |备注 |
|---|---|---|---|---|---|
| | | | | | | \
|user |用户相关配置 |1 |dict | | |
| | | | | | | \
|uid |用户标识 |2 |string | |建议采用 IMEI 或 MAC。 |
| | | | | | | \
|audio |音频相关配置 |1 |dict |✓ | |
| | | | | | | \
|url |音频链接 |2 |string |✓ | |
| | | | | | | \
|language |指定可识别的语言 |2 |string | |当该键为空时，该模型支持**中英文、上海话、闽南语，四川、陕西、粤语**识别。当将其设置为下方特定键时，它可以识别指定语言。 |\
| | | | | |英语：en-US |\
| | | | | |日语：ja-JP |\
| | | | | |印尼语：id-ID |\
| | | | | |西班牙语：es-MX |\
| | | | | |葡萄牙语：pt-BR |\
| | | | | |德语：de-DE |\
| | | | | |法语：fr-FR |\
| | | | | |韩语：ko-KR |\
| | | | | |菲律宾语：fil-PH |\
| | | | | |马来语：ms-MY |\
| | | | | |泰语：th-TH |\
| | | | | |阿拉伯语：ar-SA |\
| | | | | |例如，如果输入音频是德语，则此参数传入de-DE |
| | | | | | | \
|format |音频容器格式 |2 |string |✓ |raw / wav / mp3 / ogg |
| | | | | | | \
|codec |音频编码格式 |2 |string | |raw / opus，默认为 raw(pcm) 。 |
| | | | | | | \
|rate |音频采样率 |2 |int | |默认为 16000。 |
| | | | | | | \
|bits |音频采样点位数 |2 |int | |默认为 16，暂只支持16bits。 |
| | | | | | | \
|channel |音频声道数 |2 |int | |1(mono) / 2(stereo)，默认为1。 |
| | | | | | | \
|request |请求相关配置 |1 |dict |✓ | |
| | | | | | | \
|model_name |模型名称 |2 |string |✓ |目前只有bigmodel |
| | | | | | | \
|ssd_version |ssd版本 |2 |string | |仅在enable_speaker_info = True开启说话人分离能力，且不指定*language*字段或者*language指定为"zh-CN" 时生效* |\
| | | | | |示例： |\
| | | | | |```Python |\
| | | | | |ssd_version = "200" |\
| | | | | |``` |\
| | | | | | |
| | | | | | | \
|model_version |模型版本 |2 |string | |传model_version = "400" 使用400模型效果，不传时为默认310模型效果。 |\
| | | | | |400模型性能略有提升，且ITN有较大优化。 |
| | | | | | | \
|enable_itn |启用itn |2 |bool | |默认为true。 |\
| | | | | |文本规范化 (ITN) 是自动语音识别 (ASR) 后处理管道的一部分。 ITN 的任务是将 ASR 模型的原始语音输出转换为书面形式，以提高文本的可读性。 |\
| | | | | |例如，“一九七零年”->“1970年”和“一百二十三美元”->“$123”。 |
| | | | | | | \
|enable_punc |启用标点 |2 |bool | |默认为false。 |
| | | | | | | \
|enable_ddc |启用顺滑 |2 |bool | |默认为false。 |\
| | | | | |**++语义顺滑++**‌是一种技术，旨在提高自动语音识别（ASR）结果的文本可读性和流畅性。这项技术通过删除或修改ASR结果中的不流畅部分，如停顿词、语气词、语义重复词等，使得文本更加易于阅读和理解。 |
| | | | | | | \
|enable_speaker_info |启用说话人聚类分离 |2 |bool | |默认为false，开启后可返回说话人的信息，10人以内，效果较好。 |\
| | | | | |（如果音频存在音量、远近等明显变化，无法保证区分效果） |
| | | | | | | \
|enable_channel_split |启用双声道识别 |2 |bool | |如果设为"True"，则会在返回结果中使用channel_id标记，1为左声道，2为右声道。默认 "False"默认为false |
| | | | | | | \
|show_utterances |输出语音停顿、分句、分词信息 |2 |bool | | |
| | | | | | | \
|show_speech_rate |分句信息携带语速 |2 |bool | |如果设为"True"，则会在分句additions信息中使用speech_rate标记，单位为 token/s。默认 "False" |
| | | | | | | \
|show_volume |分句信息携带音量 |2 |bool | |如果设为"True"，则会在分句additions信息中使用volume标记，单位为 分贝。默认 "False" |
| | | | | | | \
|enable_lid |启用语种识别 |2 |bool | |**目前支持语种：中英文、上海话、闽南语，四川、陕西、粤语** |\
| | | | | |如果设为"True"，则会在additions信息中使用lid_lang标记, 返回对应的语种标签。默认 "False" |\
| | | | | |支持的标签包括： |\
| | | | | | |\
| | | | | |*  singing_en：英文唱歌 |\
| | | | | |*  singing_mand：普通话唱歌 |\
| | | | | |*  singing_dia_cant：粤语唱歌 |\
| | | | | |*  speech_en：英文说话 |\
| | | | | |*  speech_mand：普通话说话 |\
| | | | | |*  speech_dia_nan：闽南语 |\
| | | | | |* speech_dia_wuu：吴语（含上海话） |\
| | | | | |*  speech_dia_cant：粤语说话 |\
| | | | | |*  speech_dia_xina：西南官话（含四川话） |\
| | | | | |*  speech_dia_zgyu：中原官话（含陕西话） |\
| | | | | |*  other_langs：其它语种（其它语种人声） |\
| | | | | |*  others：检测不出（非语义人声和非人声） |\
| | | | | |   空时代表无法判断（例如传入音频过短等） |\
| | | | | | |\
| | | | | |**实际不支持识别的语种（无识别结果），但该参数可检测并输出对应lang_code。对应的标签如下：** |\
| | | | | | |\
| | | | | |* singing_hi：印度语唱歌 |\
| | | | | |* singing_ja：日语唱歌 |\
| | | | | |* singing_ko：韩语唱歌 |\
| | | | | |* singing_th：泰语唱歌 |\
| | | | | |* speech_hi：印地语说话 |\
| | | | | |* speech_ja：日语说话 |\
| | | | | |* speech_ko：韩语说话 |\
| | | | | |* speech_th：泰语说话 |\
| | | | | |* speech_kk：哈萨克语说话 |\
| | | | | |* speech_bo：藏语说话 |\
| | | | | |* speech_ug：维语 |\
| | | | | |* speech_mn：蒙古语 |\
| | | | | |* speech_dia_ql：琼雷话 |\
| | | | | |* speech_dia_hsn：湘语 |\
| | | | | |* speech_dia_jin：晋语 |\
| | | | | |* speech_dia_hak：客家话 |\
| | | | | |* speech_dia_chao：潮汕话 |\
| | | | | |* speech_dia_juai：江淮官话 |\
| | | | | |* speech_dia_lany：兰银官话 |\
| | | | | |* speech_dia_dbiu：东北官话 |\
| | | | | |* speech_dia_jliu：胶辽官话 |\
| | | | | |* speech_dia_jlua：冀鲁官话 |\
| | | | | |* speech_dia_cdo：闽东话 |\
| | | | | |* speech_dia_gan：赣语 |\
| | | | | |* speech_dia_mnp：闽北语 |\
| | | | | |* speech_dia_czh：徽语 |
| | | | | | | \
|enable_emotion_detection |启用情绪检测 |2 |bool | |如果设为"True"，则会在分句additions信息中使用emotion标记, 返回对应的情绪标签。默认 "False" |\
| | | | | |支持的情绪标签包括： |\
| | | | | | |\
| | | | | |*  "angry"：表示情绪为生气 |\
| | | | | |*  "happy"：表示情绪为开心 |\
| | | | | |*  "neutral"：表示情绪为平静或中性 |\
| | | | | |*  "sad"：表示情绪为悲伤 |\
| | | | | |*  "surprise"：表示情绪为惊讶 |
| | | | | | | \
|enable_gender_detection |启用性别检测 |2 |bool | |如果设为"True"，则会在分句additions信息中使用gender标记, 返回对应的性别标签（male/female）。默认 "False" |
| | | | | | | \
|vad_segment |使用vad分句 |2 |bool | |默认为false，默认是语义分句。 |\
| | | | | |打开双声道识别时，通常需要使用vad分句，可同时打开此参数 |
| | | | | | | \
|end_window_size |强制判停时间 |2 |int | |范围300 - 5000ms，建议设置800ms或者1000ms，比较敏感的场景可以配置500ms或者更小。（如果配置的过小，则会导致分句过碎，配置过大会导致不容易将说话内容分开。建议依照自身场景按需配置） |\
| | | | | |配置该值，不使用语义分句，根据静音时长来分句。 |
| | | | | | | \
|sensitive_words_filter |敏感词过滤 |2 |string | |敏感词过滤功能,支持开启或关闭,支持自定义敏感词。该参数可实现：不处理(默认,即展示原文)、过滤、替换为*。 |\
| | | | | |示例： |\
| | | | | |system_reserved_filter  //是否使用系统敏感词，会替换成*(默认系统敏感词主要包含一些限制级词汇） |\
| | | | | |filter_with_empty   // 想要替换成空的敏感词  |\
| | | | | |filter_with_signed  // 想要替换成 * 的敏感词 |\
| | | | | |```Python |\
| | | | | |"sensitive_words_filter":{\"system_reserved_filter\":true,\"filter_with_empty\":[\"敏感词\"],\"filter_with_signed\":[\"敏感词\"]}", |\
| | | | | |``` |\
| | | | | | |
| | | | | | | \
|enable_poi_fc |开启 POI function call |2 |bool | |对于语音识别困难的词语，能调用专业的地图领域推荐词服务辅助识别 |\
| | | | | |示例： |\
| | | | | |```Python |\
| | | | | |"request": { |\
| | | | | |    "enable_poi_fc": true, |\
| | | | | |    "corpus": { |\
| | | | | |        "context": "{\"loc_info\":{\"city_name\":\"北京市\"}}" |\
| | | | | |    } |\
| | | | | |} |\
| | | | | |``` |\
| | | | | | |\
| | | | | |其中loc_info字段可选，传入该字段结果相对更精准，city_name单位为地级市。 |
| | | | | | | \
|enable_music_fc |开启音乐 function call |2 |bool | |对于语音识别困难的词语，能调用专业的音领域推荐词服务辅助识别 |\
| | | | | |示例： |\
| | | | | |```Python |\
| | | | | |"request": { |\
| | | | | |    "enable_music_fc": true |\
| | | | | |} |\
| | | | | |``` |\
| | | | | | |
| | | | | | | \
|corpus |语料/干预词等 |2 |string | | |
| | | | | | | \
|boosting_table_name |自学习平台上设置的热词词表名称 |3 |string | |热词功能和设置方法可以参考[文档](https://www.volcengine.com/docs/6561/155738) |
| | | | | | | \
|correct_table_name |自学习平台上设置的替换词词表名称 |3 |string | |替换词功能和设置方法可以参考[文档](https://www.volcengine.com/docs/6561/1206007) |
| | | | | | | \
|context |上下文功能 |3 |string | |1. 热词直传，支持5000个词 |\
| | | | | | |\
| | | | | |"context":"{\"hotwords\":[{\"word\":\"热词1号\"}, {\"word\":\"热词2号\"}]}" |\
| | | | | | |\
| | | | | | |\
| | | | | |2. 上下文，限制800 tokens及20轮（含）内，超出会按照时间顺序从新到旧截断，优先保留更新的对话 |\
| | | | | | |\
| | | | | |  context_data字段按照从新到旧的顺序排列，传入需要序列化为jsonstring（转义引号） |\
| | | | | |**豆包录音文件识别模型2.0，支持将上下文理解的范围从纯文本扩展到视觉层面，** |\
| | | | | |**通过理解图像内容，帮助模型更精准地完成语音转录。通过image_url传入图片，** |\
| | | | | |**图片限制传入1张，大小：500k以内（格式：jpeg、jpg、png ）** |\
| | | | | |```SQL |\
| | | | | |上下文:可以加入对话历史、聊天所在bot信息、个性化信息、业务场景信息等,如: |\
| | | | | |a.对话历史:把最近几轮的对话历史传进来 |\
| | | | | |b.聊天所在bot信息:如"我在和林黛玉聊天","我在使用A助手和手机对话" |\
| | | | | |c.个性化信息:"我当前在北京市海淀区","我有四川口音","我喜欢音乐" |\
| | | | | |d.业务场景信息:"当前是中国平安的营销人员针对外部客户采访的录音,可能涉及..." |\
| | | | | |{ |\
| | | | | |    \"context_type\": \"dialog_ctx\", |\
| | | | | |    \"context_data\":[ |\
| | | | | |        {\"text\": \"text1\"}, |\
| | | | | |        {\"image_url\": \"image_url\"}, |\
| | | | | |        {\"text\": \"text2\"}, |\
| | | | | |        {\"text\": \"text3\"}, |\
| | | | | |        {\"text\": \"text4\"}, |\
| | | | | |        ... |\
| | | | | |    ] |\
| | | | | |} |\
| | | | | |``` |\
| | | | | | |
| | | | | | | \
|callback |回调地址 |1 |string | |举例： |\
| | | | | |```Plain Text |\
| | | | | |"callback": "http://xxx" |\
| | | | | |``` |\
| | | | | | |\
| | | | | | |
| | | | | | | \
|callback_data |回调信息 |1 |string | |举例： |\
| | | | | | |\
| | | | | |```Plain Text |\
| | | | | |"callback_data":"$Request-Id" |\
| | | | | |``` |\
| | | | | | |\
| | | | | | |

请求示例：
```JSON
{
    "user": {
        "uid": "388808087185088"
    },
    "audio": {
        "format": "mp3",
        "url": "http://xxx.com/obj/sample.mp3"
    },
    "request": {
        "model_name": "bigmodel",
        "enable_itn": true
    }
}
```

<span id="950b1aef"></span>
## 应答
Response header如下：

| | | | \
|Key |说明 |Value 示例 |
|---|---|---|
| | | | \
|X-Tt-Logid |服务端返回的 logid，建议用户获取和打印方便定位问题 |202407261553070FACFE6D19421815D605 |
| | | | \
|X-Api-Status-Code |提交任务后服务端返回的状态码，20000000表示提交成功，其他表示失败 | |
| | | | \
|X-Api-Message |提交任务后服务端返回的信息，OK表示成功，其他表示失败 | |

Response body为空
<span id="aa510007"></span>
# 查询结果
<span id="9151467f"></span>
## 接口地址
火山地址：https://openspeech.bytedance.com/api/v3/auc/bigmodel/query
<span id="a5746fcf"></span>
## 请求
请求方式：HTTP POST。
请求和应答，均采用在 HTTP BODY 里面传输 JSON 格式字串的方式。
Header 需要加入内容类型标识：

| | | | \
|Key |说明 |Value 示例 |
|---|---|---|
| | | | \
|X-Api-App-Key |使用火山引擎控制台获取的APP ID，可参考 [控制台使用FAQ-Q1](/docs/6561/196768#q1%EF%BC%9A%E5%93%AA%E9%87%8C%E5%8F%AF%E4%BB%A5%E8%8E%B7%E5%8F%96%E5%88%B0%E4%BB%A5%E4%B8%8B%E5%8F%82%E6%95%B0appid%EF%BC%8Ccluster%EF%BC%8Ctoken%EF%BC%8Cauthorization-type%EF%BC%8Csecret-key-%EF%BC%9F) |123456789 |
| | | | \
|X-Api-Access-Key |使用火山引擎控制台获取的Access Token，可参考 [控制台使用FAQ-Q1](/docs/6561/196768#q1%EF%BC%9A%E5%93%AA%E9%87%8C%E5%8F%AF%E4%BB%A5%E8%8E%B7%E5%8F%96%E5%88%B0%E4%BB%A5%E4%B8%8B%E5%8F%82%E6%95%B0appid%EF%BC%8Ccluster%EF%BC%8Ctoken%EF%BC%8Cauthorization-type%EF%BC%8Csecret-key-%EF%BC%9F) |your-access-key |
| | | | \
|X-Api-Resource-Id |表示调用服务的资源信息 ID |豆包录音文件识别模型1.0 |\
| | | |\
| | |* volc.bigasr.auc |\
| | | |\
| | |豆包录音文件识别模型2.0 |\
| | | |\
| | |* volc.seedasr.auc |
| | | | \
|X-Api-Request-Id |用于提交和查询任务的任务ID。查询时需使用提交成功的任务Id |67ee89ba-7050-4c04-a3d7-ac61a63499b3 |

body为空json：
```Go
{}
```

<span id="311c9941"></span>
## 应答
Response header如下：

| | | | \
|Key |说明 |Value 示例 |
|---|---|---|
| | | | \
|X-Tt-Logid |服务端返回的 logid，建议用户获取和打印方便定位问题 |202407261553070FACFE6D19421815D605 |
| | | | \
|X-Api-Status-Code |提交任务后服务端返回的状态码，具体错误码参考下面错误码列表 | |
| | | | \
|X-Api-Message |提交任务后服务端返回的信息，OK表示成功，其他表示失败 | |

Response Body格式 ：JSON。
应答字段：

| | | | | | \
|字段 |说明 |层级 |格式 |备注 |
|---|---|---|---|---|
| | | | | | \
|result |识别结果 |1 |list |仅当识别成功时填写 |
| | | | | | \
|text |整个音频的识别结果文本 |2 |string |仅当识别成功时填写。 |
| | | | | | \
|utterances |识别结果语音分句信息 |2 |list |仅当识别成功且开启show_utterances时填写。 |
| | | | | | \
|text |utterance级的文本内容 |3 |string |仅当识别成功且开启show_utterances时填写。 |
| | | | | | \
|start_time |起始时间（毫秒） |3 |int |仅当识别成功且开启show_utterances时填写。 |
| | | | | | \
|end_time |结束时间（毫秒） |3 |int |仅当识别成功且开启show_utterances时填写。 |

应答示例：
返回文本的形式：
```JSON
{
  "audio_info": {"duration": 10000},
  "result": {
      "text": "这是字节跳动， 今日头条母公司。",
      "utterances": [
        {
          "definite": true,
          "end_time": 1705,
          "start_time": 0,
          "text": "这是字节跳动，",
          "words": [
            {
              "blank_duration": 0,
              "end_time": 860,
              "start_time": 740,
              "text": "这"
            },
            {
              "blank_duration": 0,
              "end_time": 1020,
              "start_time": 860,
              "text": "是"
            },
            {
              "blank_duration": 0,
              "end_time": 1200,
              "start_time": 1020,
              "text": "字"
            },
            {
              "blank_duration": 0,
              "end_time": 1400,
              "start_time": 1200,
              "text": "节"
            },
            {
              "blank_duration": 0,
              "end_time": 1560,
              "start_time": 1400,
              "text": "跳"
            },
            {
              "blank_duration": 0,
              "end_time": 1640,
              "start_time": 1560,
              "text": "动"
            }
          ]
        },
        {
          "definite": true,
          "end_time": 3696,
          "start_time": 2110,
          "text": "今日头条母公司。",
          "words": [
            {
              "blank_duration": 0,
              "end_time": 3070,
              "start_time": 2910,
              "text": "今"
            },
            {
              "blank_duration": 0,
              "end_time": 3230,
              "start_time": 3070,
              "text": "日"
            },
            {
              "blank_duration": 0,
              "end_time": 3390,
              "start_time": 3230,
              "text": "头"
            },
            {
              "blank_duration": 0,
              "end_time": 3550,
              "start_time": 3390,
              "text": "条"
            },
            {
              "blank_duration": 0,
              "end_time": 3670,
              "start_time": 3550,
              "text": "母"
            },
            {
              "blank_duration": 0,
              "end_time": 3696,
              "start_time": 3670,
              "text": "公"
            },
            {
              "blank_duration": 0,
              "end_time": 3696,
              "start_time": 3696,
              "text": "司"
            }
          ]
        }
      ]
   },
  "audio_info": {
    "duration": 3696
  }
}
```

<span id="e56afc6c"></span>
## 错误码

| | | | \
|错误码 |含义 |说明 |
|---|---|---|
| | | | \
|20000000 |成功 | |
| | | | \
|20000001 |正在处理中 | |
| | | | \
|20000002 |任务在队列中 | |
| | | | \
|20000003 |静音音频 |返回该错误码无需重新query，直接重新submit |
| | | | \
|45000001 |请求参数无效 |请求参数缺失必需字段 / 字段值无效 / 重复请求。 |
| | | | \
|45000002 |空音频 | |
| | | | \
|45000151 |音频格式不正确 | |
| | | | \
|550xxxx |服务内部处理错误 | |
| | | | \
|55000031 |服务器繁忙 |服务过载，无法处理当前请求。 |

<span id="f4d1642c"></span>
# Demo
python:
<Attachment link="https://p9-arcosite.byteimg.com/tos-cn-i-goo7wpa0wc/654bf568842f4d3b9ea6e63e311955a5~tplv-goo7wpa0wc-image.image" name="auc_python.zip" ></Attachment>
Go：
<Attachment link="https://p9-arcosite.byteimg.com/tos-cn-i-goo7wpa0wc/de7039ce567b46618e0a4f330ce8a72e~tplv-goo7wpa0wc-image.image" name="auc_go.zip" ></Attachment>
Java:
<Attachment link="https://p9-arcosite.byteimg.com/tos-cn-i-goo7wpa0wc/122ebba6af6e4313a194cc07dc0f0547~tplv-goo7wpa0wc-image.image" name="java_client.zip" ></Attachment>







