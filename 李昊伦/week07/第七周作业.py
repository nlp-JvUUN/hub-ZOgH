{
  "model": "qwen-plus",
  "n_samples": 100,
  "zero_shot": {
    "precision": 0.859504132231405,
    "recall": 0.4748858447488584,
    "f1": 0.6117647058823529,
    "tp": 104,
    "pred_total": 121,
    "gold_total": 219
  },
  "few_shot": {
    "precision": 0.875,
    "recall": 0.5114155251141552,
    "f1": 0.6455331412103746,
    "tp": 112,
    "pred_total": 128,
    "gold_total": 219
  },
  "detail": [
    {
      "text": "愿中国的女飞人与世界高手比翼齐飞。",
      "gold": [
        {
          "text": "中国",
          "type": "location"
        }
      ],
      "zero_shot": [],
      "few_shot": [
        {
          "text": "中国",
          "type": "location"
        }
      ]
    },
    {
      "text": "根据中央《通知》要求，省委决定今年第四季度，对全省的县委书记集中进行一次以讲学习、讲政治、讲正气为主要内容的党性党风教育，以整风的精神开展批评和自我批评，着重解决密切联系群众和反腐倡廉的问题。",
      "gold": [
        {
          "text": "省委",
          "type": "organization"
        }
      ],
      "zero_shot": [],
      "few_shot": []
    },
    {
      "text": "此案经海口市中级人民法院一审，以受贿罪判处辛业江有期徒刑5年，依法追缴其非法所得。",
      "gold": [
        {
          "text": "海口市中级人民法院",
          "type": "organization"
        },
        {
          "text": "辛业江",
          "type": "person"
        }
      ],
      "zero_shot": [
        {
          "text": "海口市中级人民法院",
          "type": "organization"
        },
        {
          "text": "辛业江",
          "type": "person"
        }
      ],
      "few_shot": [
        {
          "text": "海口市中级人民法院",
          "type": "organization"
        },
        {
          "text": "辛业江",
          "type": "person"
        }
      ]
    },
    {
      "text": "1937年12月13日，中共中央政治局作出《对于南方游击区工作的决议》，对项英同志和南方各游击区军民的斗争给予高度评价。",
      "gold": [
        {
          "text": "中共中央政治局",
          "type": "organization"
        },
        {
          "text": "项英",
          "type": "person"
        }
      ],
      "zero_shot": [
        {
          "text": "中共中央政治局",
          "type": "organization"
        },
        {
          "text": "项英",
          "type": "person"
        }
      ],
      "few_shot": [
        {
          "text": "中共中央政治局",
          "type": "organization"
        },
        {
          "text": "项英",
          "type": "person"
        }
      ]
    },
    {
      "text": "通辽分局为职工再就业创造条件本报讯沈铁通辽分局在深化企业改革中，大胆实践，充分开发和利用千里铁路沿线的土地资源和闲置的房屋，实施了“千亩良田、万头家禽工程”，为企业扭亏增盈，减人增效，安置下岗分流人员，职工再就业创造条件。",
      "gold": [
        {
          "text": "通辽分局",
          "type": "organization"
        },
        {
          "text": "沈铁通辽分局",
          "type": "organization"
        }
      ],
      "zero_shot": [],
      "few_shot": [
        {
          "text": "通辽分局",
          "type": "organization"
        },
        {
          "text": "沈铁通辽分局",
          "type": "organization"
        }
      ]
    },
    {
      "text": "北京雷阵雨转晴17℃／30℃天津雷阵雨转多云19℃／31℃石家庄晴20℃／34℃太原晴15℃／30℃呼和浩特多云转晴13℃／26℃沈阳雷阵雨14℃／22℃大连多云转阴14℃／20℃长春小雨13℃／22℃哈尔滨雷阵雨10℃／23℃上海多云19℃／25℃南京阴20℃／27℃杭州多云转阴20℃／27℃合肥多云20℃／30℃福州小雨20℃／25℃南昌多云转小雨20℃／26℃济南晴转多云21℃／30℃青岛多云转晴16℃／22℃郑州晴转多云16℃／28℃武汉多云21℃／28℃长沙多云转小雨22℃／32℃广州中雨21℃／27℃南宁大雨转中雨24℃／29℃海口多云转雷阵雨27℃／34℃成都多云转阴20℃／26℃重庆小雨20℃／25℃贵阳阴转多云17℃／25℃昆明小雨16℃／24℃拉萨雷阵雨转多云10℃／25℃西安多云19℃／30℃兰州小雨15℃／26℃西宁小雨8℃／20℃银川晴转阴17℃／24℃乌鲁木齐多云转阴17℃／26℃台北小雨23℃／28℃香港中雨转小雨21℃／28℃澳门中雨转小雨21℃／27℃东京小雨14℃／20℃曼谷小雨26℃／35℃悉尼晴11℃／19℃卡拉奇多云29℃／34℃开罗晴21℃／34℃莫斯科多云14℃／25℃法兰克福阴转小雨14℃／21℃巴黎阴转小雨12℃／18℃伦敦小雨12℃／19℃纽约多云14℃／20℃",
      "gold": [
        {
          "text": "长春",
          "type": "location"
        },
        {
          "text": "郑州",
          "type": "location"
        },
        {
          "text": "东京",
          "type": "location"
        },
        {
          "text": "哈尔滨",
          "type": "location"
        },
        {
          "text": "杭州",
          "type": "location"
        },
        {
          "text": "南宁",
          "type": "location"
        },
        {
          "text": "卡拉奇",
          "type": "location"
        },
        {
          "text": "成都",
          "type": "location"
        },
        {
          "text": "兰州",
          "type": "location"
        },
        {
          "text": "上海",
          "type": "location"
        },
        {
          "text": "济南",
          "type": "location"
        },
        {
          "text": "广州",
          "type": "location"
        },
        {
          "text": "纽约",
          "type": "location"
        },
        {
          "text": "青岛",
          "type": "location"
        },
        {
          "text": "贵阳",
          "type": "location"
        },
        {
          "text": "石家庄",
          "type": "location"
        },
        {
          "text": "拉萨",
          "type": "location"
        },
        {
          "text": "银川",
          "type": "location"
        },
        {
          "text": "武汉",
          "type": "location"
        },
        {
          "text": "香港",
          "type": "location"
        },
        {
          "text": "沈阳",
          "type": "location"
        },
        {
          "text": "南京",
          "type": "location"
        },
        {
          "text": "重庆",
          "type": "location"
        },
        {
          "text": "太原",
          "type": "location"
        },
        {
          "text": "悉尼",
          "type": "location"
        },
        {
          "text": "开罗",
          "type": "location"
        },
        {
          "text": "天津",
          "type": "location"
        },
        {
          "text": "长沙",
          "type": "location"
        },
        {
          "text": "北京",
          "type": "location"
        },
        {
          "text": "南昌",
          "type": "location"
        },
        {
          "text": "西宁",
          "type": "location"
        },
        {
          "text": "法兰克福",
          "type": "location"
        },
        {
          "text": "台北",
          "type": "location"
        },
        {
          "text": "乌鲁木齐",
          "type": "location"
        },
        {
          "text": "莫斯科",
          "type": "location"
        },
        {
          "text": "伦敦",
          "type": "location"
        },
        {
          "text": "合肥",
          "type": "location"
        },
        {
          "text": "巴黎",
          "type": "location"
        },
        {
          "text": "呼和浩特",
          "type": "location"
        },
        {
          "text": "昆明",
          "type": "location"
        },
        {
          "text": "海口",
          "type": "location"
        },
        {
          "text": "福州",
          "type": "location"
        },
        {
          "text": "大连",
          "type": "location"
        },
        {
          "text": "曼谷",
          "type": "location"
        },
        {
          "text": "西安",
          "type": "location"
        },
        {
          "text": "澳门",
          "type": "location"
        }
      ],
      "zero_shot": [],
      "few_shot": []
    },
    {
      "text": "地面控制中心一直在跟踪监测，今天晚些时候还将开会研究如何处理这一故障。",
      "gold": [
        {
          "text": "地面控制中心",
          "type": "location"
        }
      ],
      "zero_shot": [],
      "few_shot": []
    },
    {
      "text": "美国财政部长鲁宾16日在会见记者时表示，近日将派遣财政部副部长萨默斯前往日本，就日元贬值对策和日本经济问题举行协商。",
      "gold": [
        {
          "text": "鲁宾",
          "type": "person"
        },
        {
          "text": "日本",
          "type": "location"
        },
        {
          "text": "日本",
          "type": "location"
        },
        {
          "text": "美国财政部",
          "type": "organization"
        },
        {
          "text": "萨默斯",
          "type": "person"
        },
        {
          "text": "财政部",
          "type": "organization"
        }
      ],
      "zero_shot": [
        {
          "text": "美国财政部",
          "type": "organization"
        },
        {
          "text": "鲁宾",
          "type": "person"
        },
        {
          "text": "日本",
          "type": "location"
        },
        {
          "text": "萨默斯",
          "type": "person"
        }
      ],
      "few_shot": [
        {
          "text": "鲁宾",
          "type": "person"
        },
        {
          "text": "日本",
          "type": "location"
        },
        {
          "text": "萨默斯",
          "type": "person"
        },
        {
          "text": "美国",
          "type": "location"
        }
      ]
    },
    {
      "text": "被斧“砍”刀“宰”的外国游客，会误认为中国民风贪婪，目无法纪。",
      "gold": [
        {
          "text": "中国",
          "type": "location"
        }
      ],
      "zero_shot": [],
      "few_shot": []
    },
    {
      "text": "现在，赛龙舟不仅在中国各地风行，而且已传播到五大洲40多个国家和地区，成为世界人民喜爱的群众性体育活动。",
      "gold": [
        {
          "text": "中国",
          "type": "location"
        }
      ],
      "zero_shot": [
        {
          "text": "五大洲",
          "type": "location"
        },
        {
          "text": "中国",
          "type": "location"
        }
      ],
      "few_shot": [
        {
          "text": "中国",
          "type": "location"
        }
      ]
    },
    {
      "text": "德国队在开赛第9分钟的时候，由7号穆勒头球首开记录，而这个球正是由克林斯曼在禁区内头球摆渡成功交给穆勒的。",
      "gold": [
        {
          "text": "穆勒",
          "type": "person"
        },
        {
          "text": "克林斯曼",
          "type": "person"
        },
        {
          "text": "穆勒",
          "type": "person"
        },
        {
          "text": "德国队",
          "type": "organization"
        }
      ],
      "zero_shot": [
        {
          "text": "克林斯曼",
          "type": "person"
        },
        {
          "text": "穆勒",
          "type": "person"
        },
        {
          "text": "德国队",
          "type": "organization"
        }
      ],
      "few_shot": []
    },
    {
      "text": "中国驻奥地利大使刘昌业参加了以上的会见和会晤。",
      "gold": [
        {
          "text": "刘昌业",
          "type": "person"
        },
        {
          "text": "中国",
          "type": "location"
        },
        {
          "text": "奥地利",
          "type": "location"
        }
      ],
      "zero_shot": [
        {
          "text": "刘昌业",
          "type": "person"
        },
        {
          "text": "中国",
          "type": "location"
        },
        {
          "text": "奥地利",
          "type": "location"
        }
      ],
      "few_shot": [
        {
          "text": "刘昌业",
          "type": "person"
        },
        {
          "text": "中国",
          "type": "location"
        },
        {
          "text": "奥地利",
          "type": "location"
        }
      ]
    },
    {
      "text": "4月23日，编号为“准噶尔六号”的普氏野马产下小驹。",
      "gold": [
        {
          "text": "准噶尔",
          "type": "location"
        }
      ],
      "zero_shot": [
        {
          "text": "普氏野马",
          "type": "organization"
        },
        {
          "text": "准噶尔六号",
          "type": "person"
        }
      ],
      "few_shot": []
    },
    {
      "text": "郑州公安局针对张金柱案件在公安队伍中造成的恶劣影响，提出政治建警，把政治思想教育作为政治建警的首要任务。",
      "gold": [
        {
          "text": "张金柱",
          "type": "person"
        },
        {
          "text": "郑州公安局",
          "type": "organization"
        }
      ],
      "zero_shot": [
        {
          "text": "张金柱",
          "type": "person"
        },
        {
          "text": "郑州公安局",
          "type": "organization"
        }
      ],
      "few_shot": [
        {
          "text": "张金柱",
          "type": "person"
        },
        {
          "text": "郑州公安局",
          "type": "organization"
        }
      ]
    },
    {
      "text": "辽大注重外国留学生能力培养辽宁大学自1965年以来，招收外国留学生3000余人。",
      "gold": [
        {
          "text": "辽大",
          "type": "organization"
        },
        {
          "text": "辽宁大学",
          "type": "organization"
        }
      ],
      "zero_shot": [
        {
          "text": "辽宁大学",
          "type": "organization"
        }
      ],
      "few_shot": [
        {
          "text": "辽大",
          "type": "organization"
        },
        {
          "text": "辽宁大学",
          "type": "organization"
        }
      ]
    },
    {
      "text": "主教练马跃南说，队伍重新集中后人员还会有调整，他将在联赛期间做进一步观察。",
      "gold": [
        {
          "text": "马跃南",
          "type": "person"
        }
      ],
      "zero_shot": [
        {
          "text": "马跃南",
          "type": "person"
        }
      ],
      "few_shot": [
        {
          "text": "马跃南",
          "type": "person"
        }
      ]
    },
    {
      "text": "此前，巴勒斯坦方面已同意接受美国的计划，并希望美国促使以色列也接受该计划。",
      "gold": [
        {
          "text": "美国",
          "type": "location"
        },
        {
          "text": "以色列",
          "type": "location"
        },
        {
          "text": "美国",
          "type": "location"
        },
        {
          "text": "巴勒斯坦",
          "type": "location"
        }
      ],
      "zero_shot": [
        {
          "text": "以色列",
          "type": "location"
        },
        {
          "text": "美国",
          "type": "location"
        },
        {
          "text": "巴勒斯坦",
          "type": "location"
        }
      ],
      "few_shot": [
        {
          "text": "以色列",
          "type": "location"
        },
        {
          "text": "美国",
          "type": "location"
        },
        {
          "text": "巴勒斯坦",
          "type": "location"
        }
      ]
    },
    {
      "text": "西安电影制片厂厂长张丕民介绍，《惹事生非》在陕西工厂、农村、学校放映时，观众反应强烈，广大观众期盼好的农村题材影片，这是我们拍摄该片后最深切的感受。",
      "gold": [
        {
          "text": "张丕民",
          "type": "person"
        },
        {
          "text": "陕西",
          "type": "location"
        },
        {
          "text": "西安电影制片厂",
          "type": "organization"
        }
      ],
      "zero_shot": [
        {
          "text": "张丕民",
          "type": "person"
        },
        {
          "text": "陕西",
          "type": "location"
        },
        {
          "text": "西安电影制片厂",
          "type": "organization"
        }
      ],
      "few_shot": [
        {
          "text": "张丕民",
          "type": "person"
        },
        {
          "text": "陕西",
          "type": "location"
        },
        {
          "text": "西安电影制片厂",
          "type": "organization"
        }
      ]
    },
    {
      "text": "他说，在亚太地区安全、防止大规模杀伤性武器扩散、打击国际犯罪、保护环境、双边贸易与能源合作等各个领域，美国都需要与中国合作。",
      "gold": [
        {
          "text": "中国",
          "type": "location"
        },
        {
          "text": "亚太",
          "type": "location"
        },
        {
          "text": "美国",
          "type": "location"
        }
      ],
      "zero_shot": [],
      "few_shot": []
    },
    {
      "text": "金人庆强调，全国税务系统要按照国务院的要求，加强征管，挖掘潜力，标本兼治，确保增收目标的实现；要顾全大局，为国分忧，把收入任务及时落实到基层；要抓紧清理漏管户，对重点税源加强专项检查，大力清理欠税，严格期初库存抵扣；要加强加油站、出租车的税收征管；要采取得力措施，认真落实调整商业企业增值税一般纳税人和交通运输企业抵扣增值税比例的税收政策；要强化税务稽查，进一步加快税务稽查队伍建设，充分发挥稽查职能，严厉打击偷逃税行为；要根据税源结构变化，及时调整征管力量，确保新的经济增长点同时也成为新的税收增长点。",
      "gold": [
        {
          "text": "国务院",
          "type": "organization"
        },
        {
          "text": "金人庆",
          "type": "person"
        }
      ],
      "zero_shot": [
        {
          "text": "全国税务系统",
          "type": "organization"
        },
        {
          "text": "国务院",
          "type": "organization"
        },
        {
          "text": "金人庆",
          "type": "person"
        }
      ],
      "few_shot": [
        {
          "text": "国务院",
          "type": "organization"
        },
        {
          "text": "金人庆",
          "type": "person"
        }
      ]
    },
    {
      "text": "1993年作为中国代表参加了东京国际法学会，其发言受到国际法学界极大关注，破例被选为大会共同主席。",
      "gold": [
        {
          "text": "中国",
          "type": "location"
        },
        {
          "text": "东京国际法学会",
          "type": "organization"
        }
      ],
      "zero_shot": [
        {
          "text": "中国",
          "type": "location"
        },
        {
          "text": "东京",
          "type": "location"
        }
      ],
      "few_shot": [
        {
          "text": "中国",
          "type": "location"
        },
        {
          "text": "东京",
          "type": "location"
        }
      ]
    },
    {
      "text": "今年4月，北京市技术监督局在查处中竟发现假“大宝”中有真“大宝”从没生产过的产品。",
      "gold": [
        {
          "text": "北京市技术监督局",
          "type": "organization"
        }
      ],
      "zero_shot": [
        {
          "text": "北京市技术监督局",
          "type": "organization"
        },
        {
          "text": "北京",
          "type": "location"
        }
      ],
      "few_shot": [
        {
          "text": "北京市技术监督局",
          "type": "organization"
        },
        {
          "text": "北京",
          "type": "location"
        }
      ]
    },
    {
      "text": "新华社北京5月10日电外交部长唐家璇今天下午前往泰国驻华使馆吊唁泰国前总理、泰中友协会长差猜·春哈旺。",
      "gold": [
        {
          "text": "新华社",
          "type": "organization"
        },
        {
          "text": "泰中友协会",
          "type": "organization"
        },
        {
          "text": "泰国",
          "type": "location"
        },
        {
          "text": "北京",
          "type": "location"
        },
        {
          "text": "唐家璇",
          "type": "person"
        },
        {
          "text": "泰国",
          "type": "location"
        },
        {
          "text": "华",
          "type": "location"
        },
        {
          "text": "差猜·春哈旺",
          "type": "person"
        },
        {
          "text": "外交部",
          "type": "organization"
        }
      ],
      "zero_shot": [
        {
          "text": "新华社",
          "type": "organization"
        },
        {
          "text": "泰国",
          "type": "location"
        },
        {
          "text": "北京",
          "type": "location"
        },
        {
          "text": "唐家璇",
          "type": "person"
        },
        {
          "text": "泰国驻华使馆",
          "type": "organization"
        },
        {
          "text": "差猜·春哈旺",
          "type": "person"
        }
      ],
      "few_shot": [
        {
          "text": "新华社",
          "type": "organization"
        },
        {
          "text": "北京",
          "type": "location"
        },
        {
          "text": "唐家璇",
          "type": "person"
        },
        {
          "text": "泰国驻华使馆",
          "type": "organization"
        },
        {
          "text": "差猜·春哈旺",
          "type": "person"
        }
      ]
    },
    {
      "text": "本报北京6月4日讯记者贾西平报道：集中展示中国工程技术最高水平的《中国科学技术前沿———1997中国工程院版》一书首发式，今晚在京举行。",
      "gold": [
        {
          "text": "贾西平",
          "type": "person"
        },
        {
          "text": "中国",
          "type": "location"
        },
        {
          "text": "京",
          "type": "location"
        },
        {
          "text": "中国",
          "type": "location"
        },
        {
          "text": "中国工程院",
          "type": "organization"
        },
        {
          "text": "北京",
          "type": "location"
        }
      ],
      "zero_shot": [
        {
          "text": "中国工程院",
          "type": "organization"
        },
        {
          "text": "北京",
          "type": "location"
        }
      ],
      "few_shot": [
        {
          "text": "中国工程院",
          "type": "organization"
        },
        {
          "text": "贾西平",
          "type": "person"
        },
        {
          "text": "北京",
          "type": "location"
        }
      ]
    },
    {
      "text": "美国国务院在一项声明中说，这只能进一步加剧南亚的紧张局势并破坏全球有关核不扩散的协调一致。",
      "gold": [
        {
          "text": "南亚",
          "type": "location"
        },
        {
          "text": "美国国务院",
          "type": "organization"
        }
      ],
      "zero_shot": [
        {
          "text": "美国国务院",
          "type": "organization"
        }
      ],
      "few_shot": [
        {
          "text": "南亚",
          "type": "location"
        },
        {
          "text": "美国国务院",
          "type": "organization"
        }
      ]
    },
    {
      "text": "在他的家里，他指着满满一柜子的英文书说，离开中国时，他要把这些书全部捐给清华的图书馆。",
      "gold": [
        {
          "text": "清华",
          "type": "organization"
        },
        {
          "text": "中国",
          "type": "location"
        }
      ],
      "zero_shot": [
        {
          "text": "清华",
          "type": "organization"
        }
      ],
      "few_shot": [
        {
          "text": "清华",
          "type": "organization"
        }
      ]
    },
    {
      "text": "如果有一种物理过程能够把质量m全部压到史瓦西半径，物质就达到黑洞的状态。",
      "gold": [
        {
          "text": "史瓦西",
          "type": "person"
        }
      ],
      "zero_shot": [],
      "few_shot": []
    },
    {
      "text": "本报济南6月11日电田里的秸秆能变成燃气，农民也可以像城里人一样用上管道燃气。",
      "gold": [
        {
          "text": "济南",
          "type": "location"
        }
      ],
      "zero_shot": [],
      "few_shot": [
        {
          "text": "济南",
          "type": "location"
        }
      ]
    },
    {
      "text": "正如学者丁东所言：“这一代人的真情实感十分复杂，有理想的狂热，也有觉醒的怀疑，有深沉的悲歌，也有低吟的私语，但这毕竟都是真实的，唯其真实，才可称为精神化石。”",
      "gold": [
        {
          "text": "丁东",
          "type": "person"
        }
      ],
      "zero_shot": [
        {
          "text": "丁东",
          "type": "person"
        }
      ],
      "few_shot": [
        {
          "text": "丁东",
          "type": "person"
        }
      ]
    },
    {
      "text": "关于台湾问题，克林顿说，他将向中国领导人重申美国向中方作出的承诺，再次表明美国坚持一个中国的政策，遵守中美三个联合公报的原则。",
      "gold": [
        {
          "text": "中",
          "type": "location"
        },
        {
          "text": "中国",
          "type": "location"
        },
        {
          "text": "美",
          "type": "location"
        },
        {
          "text": "台湾",
          "type": "location"
        },
        {
          "text": "中国",
          "type": "location"
        },
        {
          "text": "美国",
          "type": "location"
        },
        {
          "text": "克林顿",
          "type": "person"
        },
        {
          "text": "中",
          "type": "location"
        },
        {
          "text": "美国",
          "type": "location"
        }
      ],
      "zero_shot": [
        {
          "text": "克林顿",
          "type": "person"
        },
        {
          "text": "中美三个联合公报",
          "type": "organization"
        },
        {
          "text": "中国",
          "type": "location"
        },
        {
          "text": "美国",
          "type": "location"
        }
      ],
      "few_shot": [
        {
          "text": "中国",
          "type": "location"
        },
        {
          "text": "美国",
          "type": "location"
        },
        {
          "text": "台湾",
          "type": "location"
        },
        {
          "text": "克林顿",
          "type": "person"
        },
        {
          "text": "中美三个联合公报",
          "type": "organization"
        }
      ]
    },
    {
      "text": "英国内政部为这次试验提供了资助，它将根据该系统在肯特郡的试验情况决定是否进一步在全国其它地区推广这种技术。",
      "gold": [
        {
          "text": "英国内政部",
          "type": "organization"
        },
        {
          "text": "肯特郡",
          "type": "location"
        }
      ],
      "zero_shot": [
        {
          "text": "英国内政部",
          "type": "organization"
        },
        {
          "text": "肯特郡",
          "type": "location"
        }
      ],
      "few_shot": [
        {
          "text": "英国内政部",
          "type": "organization"
        },
        {
          "text": "肯特郡",
          "type": "location"
        }
      ]
    },
    {
      "text": "李瑞环首先对斯卡尔法罗总统访华表示欢迎，对不久前他访意期间受到的热情接待表示感谢。",
      "gold": [
        {
          "text": "斯卡尔法罗",
          "type": "person"
        },
        {
          "text": "李瑞环",
          "type": "person"
        },
        {
          "text": "意",
          "type": "location"
        },
        {
          "text": "华",
          "type": "location"
        }
      ],
      "zero_shot": [
        {
          "text": "斯卡尔法罗",
          "type": "person"
        },
        {
          "text": "李瑞环",
          "type": "person"
        }
      ],
      "few_shot": [
        {
          "text": "斯卡尔法罗",
          "type": "person"
        },
        {
          "text": "李瑞环",
          "type": "person"
        }
      ]
    },
    {
      "text": "敦煌的西边，至今仍保留一段汉长城。",
      "gold": [
        {
          "text": "长城",
          "type": "location"
        },
        {
          "text": "敦煌",
          "type": "location"
        }
      ],
      "zero_shot": [
        {
          "text": "汉长城",
          "type": "location"
        },
        {
          "text": "敦煌",
          "type": "location"
        }
      ],
      "few_shot": [
        {
          "text": "汉长城",
          "type": "location"
        },
        {
          "text": "敦煌",
          "type": "location"
        }
      ]
    },
    {
      "text": "自由党愿意同中国共产党发展真正相互信赖的关系。",
      "gold": [
        {
          "text": "中国共产党",
          "type": "organization"
        },
        {
          "text": "自由党",
          "type": "organization"
        }
      ],
      "zero_shot": [
        {
          "text": "中国共产党",
          "type": "organization"
        },
        {
          "text": "自由党",
          "type": "organization"
        }
      ],
      "few_shot": [
        {
          "text": "中国共产党",
          "type": "organization"
        },
        {
          "text": "自由党",
          "type": "organization"
        }
      ]
    },
    {
      "text": "思科在中国拓展着生存空间，而中国也在网络时代寻找着新的机会。",
      "gold": [
        {
          "text": "中国",
          "type": "location"
        },
        {
          "text": "中国",
          "type": "location"
        },
        {
          "text": "思科",
          "type": "organization"
        }
      ],
      "zero_shot": [
        {
          "text": "中国",
          "type": "location"
        },
        {
          "text": "思科",
          "type": "organization"
        }
      ],
      "few_shot": [
        {
          "text": "中国",
          "type": "location"
        },
        {
          "text": "思科",
          "type": "organization"
        }
      ]
    },
    {
      "text": "因此，中国印刷博物馆的历史价值和启发意义将是永远的。",
      "gold": [
        {
          "text": "中国印刷博物馆",
          "type": "location"
        }
      ],
      "zero_shot": [
        {
          "text": "中国印刷博物馆",
          "type": "organization"
        }
      ],
      "few_shot": [
        {
          "text": "中国印刷博物馆",
          "type": "organization"
        }
      ]
    },
    {
      "text": "在韶山路采访时，一位中年妇女对记者说，讲文明树新风不能搞“一阵风”，我们希望市里重新拿出去年搞“讲文明树新风”活动的劲头来。",
      "gold": [
        {
          "text": "韶山路",
          "type": "location"
        }
      ],
      "zero_shot": [
        {
          "text": "韶山路",
          "type": "location"
        }
      ],
      "few_shot": [
        {
          "text": "韶山路",
          "type": "location"
        }
      ]
    },
    {
      "text": "下岗职工潘维军抱着出生才50天的女儿潘研，从老家辽宁阜新市来到北京求医，治疗右下眼睑长出的血管瘤。",
      "gold": [
        {
          "text": "阜新市",
          "type": "location"
        },
        {
          "text": "辽宁",
          "type": "location"
        },
        {
          "text": "潘研",
          "type": "person"
        },
        {
          "text": "北京",
          "type": "location"
        },
        {
          "text": "潘维军",
          "type": "person"
        }
      ],
      "zero_shot": [
        {
          "text": "潘维军",
          "type": "person"
        },
        {
          "text": "潘研",
          "type": "person"
        },
        {
          "text": "辽宁阜新市",
          "type": "location"
        },
        {
          "text": "北京",
          "type": "location"
        }
      ],
      "few_shot": [
        {
          "text": "潘维军",
          "type": "person"
        },
        {
          "text": "潘研",
          "type": "person"
        },
        {
          "text": "辽宁阜新市",
          "type": "location"
        },
        {
          "text": "北京",
          "type": "location"
        }
      ]
    },
    {
      "text": "西藏自治区科协正在这里举办以“普及科学知识，破除封建迷信，促进两个文明建设”为主题的“科普一条街”活动。",
      "gold": [
        {
          "text": "西藏自治区科协",
          "type": "organization"
        }
      ],
      "zero_shot": [
        {
          "text": "西藏自治区科协",
          "type": "organization"
        }
      ],
      "few_shot": [
        {
          "text": "西藏自治区科协",
          "type": "organization"
        }
      ]
    },
    {
      "text": "报告还指出，应对此事负责的人员主要有：前总理差瓦立、前副总理兼财政部长安雷、前国家银行行长伦差等7人。",
      "gold": [
        {
          "text": "国家银行",
          "type": "organization"
        },
        {
          "text": "安雷",
          "type": "person"
        },
        {
          "text": "差瓦立",
          "type": "person"
        },
        {
          "text": "财政部",
          "type": "organization"
        },
        {
          "text": "伦差",
          "type": "person"
        }
      ],
      "zero_shot": [
        {
          "text": "差瓦立",
          "type": "person"
        },
        {
          "text": "安雷",
          "type": "person"
        },
        {
          "text": "伦差",
          "type": "person"
        }
      ],
      "few_shot": [
        {
          "text": "差瓦立",
          "type": "person"
        },
        {
          "text": "安雷",
          "type": "person"
        },
        {
          "text": "伦差",
          "type": "person"
        }
      ]
    },
    {
      "text": "7日其他各场比赛，八一队主场0∶0逼平四川全兴队。",
      "gold": [
        {
          "text": "八一队",
          "type": "organization"
        },
        {
          "text": "四川全兴队",
          "type": "organization"
        }
      ],
      "zero_shot": [
        {
          "text": "八一队",
          "type": "organization"
        },
        {
          "text": "四川全兴队",
          "type": "organization"
        }
      ],
      "few_shot": [
        {
          "text": "八一队",
          "type": "organization"
        },
        {
          "text": "四川全兴队",
          "type": "organization"
        }
      ]
    },
    {
      "text": "看完第二个景点———“大禹塑像”返回时，被三皇山卖门票的老者拦住，非要我们买两张门票不可。",
      "gold": [
        {
          "text": "三皇山",
          "type": "location"
        },
        {
          "text": "大禹",
          "type": "person"
        }
      ],
      "zero_shot": [
        {
          "text": "三皇山",
          "type": "location"
        },
        {
          "text": "大禹",
          "type": "person"
        }
      ],
      "few_shot": [
        {
          "text": "三皇山",
          "type": "location"
        },
        {
          "text": "大禹塑像",
          "type": "location"
        }
      ]
    },
    {
      "text": "从1996年起，威海以“提高覆盖率、扩大受益面、增强保障能力”为目标，对原有合作医疗进行大胆改革。",
      "gold": [
        {
          "text": "威海",
          "type": "location"
        }
      ],
      "zero_shot": [
        {
          "text": "威海",
          "type": "location"
        }
      ],
      "few_shot": [
        {
          "text": "威海",
          "type": "location"
        }
      ]
    },
    {
      "text": "黄卫平新著《中国政治体制改革纵横谈》，已由中央编译出版社出版。",
      "gold": [
        {
          "text": "黄卫平",
          "type": "person"
        },
        {
          "text": "中国",
          "type": "location"
        },
        {
          "text": "中央编译出版社",
          "type": "organization"
        }
      ],
      "zero_shot": [
        {
          "text": "黄卫平",
          "type": "person"
        },
        {
          "text": "中央编译出版社",
          "type": "organization"
        }
      ],
      "few_shot": [
        {
          "text": "黄卫平",
          "type": "person"
        },
        {
          "text": "中央编译出版社",
          "type": "organization"
        }
      ]
    },
    {
      "text": "如果后来换了诸葛亮，他也会像周瑜那样开解孙权，修正先前的说法。",
      "gold": [
        {
          "text": "周瑜",
          "type": "person"
        },
        {
          "text": "孙权",
          "type": "person"
        },
        {
          "text": "诸葛亮",
          "type": "person"
        }
      ],
      "zero_shot": [
        {
          "text": "周瑜",
          "type": "person"
        },
        {
          "text": "孙权",
          "type": "person"
        },
        {
          "text": "诸葛亮",
          "type": "person"
        }
      ],
      "few_shot": [
        {
          "text": "周瑜",
          "type": "person"
        },
        {
          "text": "孙权",
          "type": "person"
        },
        {
          "text": "诸葛亮",
          "type": "person"
        }
      ]
    },
    {
      "text": "他对中国的访问，是他6年总统任期中时间最长的一次出国访问，又是他一次直飞中国、中途不在任何国家停留，访问结束以后也同样直返美国的专程访问。",
      "gold": [
        {
          "text": "中国",
          "type": "location"
        },
        {
          "text": "美国",
          "type": "location"
        },
        {
          "text": "中国",
          "type": "location"
        }
      ],
      "zero_shot": [
        {
          "text": "美国",
          "type": "location"
        },
        {
          "text": "中国",
          "type": "location"
        }
      ],
      "few_shot": [
        {
          "text": "美国",
          "type": "location"
        },
        {
          "text": "中国",
          "type": "location"
        }
      ]
    },
    {
      "text": "武汉市利用空气质量周报分析出全市空气污染的特点是：交通稠密区以氮氧化物为首要污染物，人口密集区以总悬浮颗粒物为首要污染物。",
      "gold": [
        {
          "text": "武汉市",
          "type": "location"
        }
      ],
      "zero_shot": [
        {
          "text": "武汉市",
          "type": "location"
        }
      ],
      "few_shot": [
        {
          "text": "武汉市",
          "type": "location"
        }
      ]
    },
    {
      "text": "据中国美术馆工作人员介绍，这次展览自12日开展以来，参观者已逾万人。",
      "gold": [
        {
          "text": "中国美术馆",
          "type": "location"
        }
      ],
      "zero_shot": [
        {
          "text": "中国美术馆",
          "type": "organization"
        }
      ],
      "few_shot": [
        {
          "text": "中国美术馆",
          "type": "organization"
        }
      ]
    },
    {
      "text": "●温家宝会见芬兰农林部长据新华社北京6月10日电国务院副总理温家宝今天在京会见了来访的芬兰农林部长卡莱维·海米莱。",
      "gold": [
        {
          "text": "国务院",
          "type": "organization"
        },
        {
          "text": "京",
          "type": "location"
        },
        {
          "text": "北京",
          "type": "location"
        },
        {
          "text": "温家宝",
          "type": "person"
        },
        {
          "text": "芬兰农林部",
          "type": "organization"
        },
        {
          "text": "温家宝",
          "type": "person"
        },
        {
          "text": "卡莱维·海米莱",
          "type": "person"
        },
        {
          "text": "新华社",
          "type": "organization"
        },
        {
          "text": "芬兰农林部",
          "type": "organization"
        }
      ],
      "zero_shot": [
        {
          "text": "国务院",
          "type": "organization"
        },
        {
          "text": "芬兰农林部长",
          "type": "organization"
        },
        {
          "text": "北京",
          "type": "location"
        },
        {
          "text": "温家宝",
          "type": "person"
        },
        {
          "text": "卡莱维·海米莱",
          "type": "person"
        }
      ],
      "few_shot": [
        {
          "text": "国务院",
          "type": "organization"
        },
        {
          "text": "北京",
          "type": "location"
        },
        {
          "text": "芬兰",
          "type": "location"
        },
        {
          "text": "温家宝",
          "type": "person"
        },
        {
          "text": "卡莱维·海米莱",
          "type": "person"
        }
      ]
    },
    {
      "text": "这是由北京台资企业九鼎轩置业企划有限公司组织一批从事幼儿教育多年，并在文字编辑、信息采集、计算机培训等领域具有丰富经验的人士开发策划而成的。",
      "gold": [
        {
          "text": "北京",
          "type": "location"
        },
        {
          "text": "九鼎轩置业企划有限公司",
          "type": "organization"
        },
        {
          "text": "台",
          "type": "location"
        }
      ],
      "zero_shot": [
        {
          "text": "北京",
          "type": "location"
        },
        {
          "text": "九鼎轩置业企划有限公司",
          "type": "organization"
        }
      ],
      "few_shot": [
        {
          "text": "北京",
          "type": "location"
        },
        {
          "text": "九鼎轩置业企划有限公司",
          "type": "organization"
        }
      ]
    },
    {
      "text": "新当选的非统组织执行主席、布基纳法索总统孔波雷在闭幕式上发表讲话指出，制止和解决冲突、实现和平是非洲面临的重大课题。",
      "gold": [
        {
          "text": "非洲",
          "type": "location"
        },
        {
          "text": "非统组织",
          "type": "organization"
        },
        {
          "text": "布基纳法索",
          "type": "location"
        },
        {
          "text": "孔波雷",
          "type": "person"
        }
      ],
      "zero_shot": [
        {
          "text": "非统组织",
          "type": "organization"
        },
        {
          "text": "布基纳法索",
          "type": "location"
        },
        {
          "text": "孔波雷",
          "type": "person"
        }
      ],
      "few_shot": [
        {
          "text": "非统组织",
          "type": "organization"
        },
        {
          "text": "布基纳法索",
          "type": "location"
        },
        {
          "text": "孔波雷",
          "type": "person"
        }
      ]
    },
    {
      "text": "第二部分“人物的简介”，是编著者对毛泽东评点的人物的介绍。",
      "gold": [
        {
          "text": "毛泽东",
          "type": "person"
        }
      ],
      "zero_shot": [
        {
          "text": "毛泽东",
          "type": "person"
        }
      ],
      "few_shot": [
        {
          "text": "毛泽东",
          "type": "person"
        }
      ]
    },
    {
      "text": "东营市委、市政府认为，前邵村创造的“上农下渔”模式，是盐碱地耕作制度的重大变革，是盐碱地农业开发的根本出路，不失时机地采取了思想发动、政策引动、服务推动等有力措施，在全市大力推广。",
      "gold": [
        {
          "text": "东营市委",
          "type": "organization"
        },
        {
          "text": "前邵村",
          "type": "location"
        }
      ],
      "zero_shot": [
        {
          "text": "市政府",
          "type": "organization"
        },
        {
          "text": "东营市委",
          "type": "organization"
        },
        {
          "text": "前邵村",
          "type": "location"
        }
      ],
      "few_shot": [
        {
          "text": "市政府",
          "type": "organization"
        },
        {
          "text": "东营市委",
          "type": "organization"
        },
        {
          "text": "前邵村",
          "type": "location"
        }
      ]
    },
    {
      "text": "双方就当前的东南亚危机深入地交换了意见。",
      "gold": [
        {
          "text": "东南亚",
          "type": "location"
        }
      ],
      "zero_shot": [
        {
          "text": "东南亚",
          "type": "location"
        }
      ],
      "few_shot": [
        {
          "text": "东南亚",
          "type": "location"
        }
      ]
    },
    {
      "text": "杭州市要以国家的总体要求为目标，以国际国内两个市场为导向，充分发挥全国重点风景旅游城市和历史文化名城的优势和优越的区位条件，统一认识，强化旅游大产业观念；大力加快城市的现代化、国际化功能建设和整体旅游环境建设；大力加强旅游资源的保护与开发建设及特色创新；大力发展全方位、多元化的旅游经济产业体系；加快城乡一体化进程，构筑大杭州旅游新格局，使旅游业成为全市经济发展新的增长点和重要支柱产业。",
      "gold": [
        {
          "text": "杭州",
          "type": "location"
        },
        {
          "text": "杭州市",
          "type": "location"
        }
      ],
      "zero_shot": [
        {
          "text": "杭州市",
          "type": "location"
        }
      ],
      "few_shot": [
        {
          "text": "杭州市",
          "type": "location"
        }
      ]
    },
    {
      "text": "但韩国经济仍未摆脱困难局面，内需市场萎缩，设备投资下降，失业人数增加，企业倒闭严重，企业和金融结构调整缓慢，社会不安定因素上升。",
      "gold": [
        {
          "text": "韩国",
          "type": "location"
        }
      ],
      "zero_shot": [],
      "few_shot": [
        {
          "text": "韩国",
          "type": "location"
        }
      ]
    },
    {
      "text": "1996年新一轮的全球并购风潮乍起之时，美国的联邦贸易委员会和司法部的反托拉斯局以及欧盟的竞争专员绷紧了神经，对一个十亿和几十亿美元的合并讨论了又讨论，谈判了又谈判，生怕一不小心就“制造”了某个行业的“一霸”。",
      "gold": [
        {
          "text": "联邦贸易委员会",
          "type": "organization"
        },
        {
          "text": "司法部",
          "type": "organization"
        },
        {
          "text": "美国",
          "type": "location"
        },
        {
          "text": "欧盟",
          "type": "organization"
        },
        {
          "text": "反托拉斯局",
          "type": "organization"
        }
      ],
      "zero_shot": [],
      "few_shot": [
        {
          "text": "竞争专员",
          "type": "organization"
        },
        {
          "text": "欧盟",
          "type": "location"
        },
        {
          "text": "联邦贸易委员会",
          "type": "organization"
        },
        {
          "text": "司法部",
          "type": "organization"
        },
        {
          "text": "美国",
          "type": "location"
        },
        {
          "text": "反托拉斯局",
          "type": "organization"
        }
      ]
    },
    {
      "text": "按照蒙古法律，国家大呼拉尔拥有最终作出政府辞职决定的权力。",
      "gold": [
        {
          "text": "国家大呼拉尔",
          "type": "organization"
        },
        {
          "text": "蒙古",
          "type": "location"
        }
      ],
      "zero_shot": [
        {
          "text": "国家大呼拉尔",
          "type": "organization"
        },
        {
          "text": "蒙古",
          "type": "location"
        }
      ],
      "few_shot": [
        {
          "text": "国家大呼拉尔",
          "type": "organization"
        },
        {
          "text": "蒙古",
          "type": "location"
        }
      ]
    },
    {
      "text": "美国铁路客运量竟下降到仅占客运总量的0．65％。",
      "gold": [
        {
          "text": "美国",
          "type": "location"
        }
      ],
      "zero_shot": [],
      "few_shot": [
        {
          "text": "美国",
          "type": "location"
        }
      ]
    },
    {
      "text": "学校教师每人每年有一个多月时间深入企业咨询、调研。",
      "gold": [
        {
          "text": "学校",
          "type": "organization"
        }
      ],
      "zero_shot": [],
      "few_shot": []
    },
    {
      "text": "当年柳青、李昕那种一上场就拼命的劲头现在已很难见到。",
      "gold": [
        {
          "text": "柳青",
          "type": "person"
        },
        {
          "text": "李昕",
          "type": "person"
        }
      ],
      "zero_shot": [
        {
          "text": "柳青",
          "type": "person"
        },
        {
          "text": "李昕",
          "type": "person"
        }
      ],
      "few_shot": [
        {
          "text": "柳青",
          "type": "person"
        },
        {
          "text": "李昕",
          "type": "person"
        }
      ]
    },
    {
      "text": "这次是最后一次试运作，也是规模最大的一次，共有1·5万人参加，其中有1·2万人扮演旅客及接机者，其余3000人是35家航空公司及机场管理局职员。",
      "gold": [
        {
          "text": "机场管理局",
          "type": "organization"
        }
      ],
      "zero_shot": [],
      "few_shot": []
    },
    {
      "text": "1995年11月，卢亮以发展事业为由提出书面辞职报告，水泥厂不同意。",
      "gold": [
        {
          "text": "卢亮",
          "type": "person"
        },
        {
          "text": "水泥厂",
          "type": "organization"
        }
      ],
      "zero_shot": [
        {
          "text": "卢亮",
          "type": "person"
        },
        {
          "text": "水泥厂",
          "type": "organization"
        }
      ],
      "few_shot": [
        {
          "text": "卢亮",
          "type": "person"
        }
      ]
    },
    {
      "text": "33年前，年仅6岁的小俊平被连日高烧折磨得奄奄一息。",
      "gold": [
        {
          "text": "俊平",
          "type": "person"
        }
      ],
      "zero_shot": [
        {
          "text": "小俊平",
          "type": "person"
        }
      ],
      "few_shot": [
        {
          "text": "小俊平",
          "type": "person"
        }
      ]
    },
    {
      "text": "的确，在电力城随意一抬头，就可以看到“建设一阵子，管理一辈子”的标语。",
      "gold": [
        {
          "text": "电力城",
          "type": "location"
        }
      ],
      "zero_shot": [],
      "few_shot": []
    },
    {
      "text": "创刊号发表了罗点点回忆父亲罗瑞卿将军的长篇纪实文学，有『八一三抗战专题』，有新老作家肖乾、冯骥才、余华等人的作品，栏目有《过去的新闻》、《永远的童年》等，由华夏杂志社出版发行。",
      "gold": [
        {
          "text": "肖乾",
          "type": "person"
        },
        {
          "text": "余华",
          "type": "person"
        },
        {
          "text": "罗瑞卿",
          "type": "person"
        },
        {
          "text": "冯骥才",
          "type": "person"
        },
        {
          "text": "华夏杂志社",
          "type": "organization"
        },
        {
          "text": "罗点点",
          "type": "person"
        }
      ],
      "zero_shot": [
        {
          "text": "肖乾",
          "type": "person"
        },
        {
          "text": "余华",
          "type": "person"
        },
        {
          "text": "罗瑞卿",
          "type": "person"
        },
        {
          "text": "冯骥才",
          "type": "person"
        },
        {
          "text": "华夏杂志社",
          "type": "organization"
        },
        {
          "text": "罗点点",
          "type": "person"
        }
      ],
      "few_shot": [
        {
          "text": "肖乾",
          "type": "person"
        },
        {
          "text": "余华",
          "type": "person"
        },
        {
          "text": "罗瑞卿",
          "type": "person"
        },
        {
          "text": "冯骥才",
          "type": "person"
        },
        {
          "text": "华夏杂志社",
          "type": "organization"
        },
        {
          "text": "罗点点",
          "type": "person"
        }
      ]
    },
    {
      "text": "在燕京秀丽幽静的群山中，有许多鲜为人知的辽金时期的佛教遗迹：古庙、灯塔、老树、香道。",
      "gold": [
        {
          "text": "燕京",
          "type": "location"
        }
      ],
      "zero_shot": [],
      "few_shot": [
        {
          "text": "燕京",
          "type": "location"
        }
      ]
    },
    {
      "text": "穆巴拉克抵拉塔基亚后立即与叙总统阿萨德就当前地区形势的最新发展，特别是威胁中东和平进程的“严峻形势”举行了会谈。",
      "gold": [
        {
          "text": "穆巴拉克",
          "type": "person"
        },
        {
          "text": "阿萨德",
          "type": "person"
        },
        {
          "text": "叙",
          "type": "location"
        },
        {
          "text": "拉塔基亚",
          "type": "location"
        },
        {
          "text": "中东",
          "type": "location"
        }
      ],
      "zero_shot": [
        {
          "text": "叙",
          "type": "organization"
        },
        {
          "text": "穆巴拉克",
          "type": "person"
        },
        {
          "text": "阿萨德",
          "type": "person"
        },
        {
          "text": "拉塔基亚",
          "type": "location"
        }
      ],
      "few_shot": [
        {
          "text": "穆巴拉克",
          "type": "person"
        },
        {
          "text": "阿萨德",
          "type": "person"
        },
        {
          "text": "拉塔基亚",
          "type": "location"
        }
      ]
    },
    {
      "text": "1月15日，她将一纸申诉状递到了北京市劳动争议仲裁委员会，2月12日，北京市劳动争议仲裁委员会开庭审理此案。",
      "gold": [
        {
          "text": "北京市劳动争议仲裁委员会",
          "type": "organization"
        },
        {
          "text": "北京市劳动争议仲裁委员会",
          "type": "organization"
        }
      ],
      "zero_shot": [
        {
          "text": "北京",
          "type": "location"
        },
        {
          "text": "北京市劳动争议仲裁委员会",
          "type": "organization"
        }
      ],
      "few_shot": [
        {
          "text": "北京市",
          "type": "location"
        },
        {
          "text": "北京市劳动争议仲裁委员会",
          "type": "organization"
        }
      ]
    },
    {
      "text": "在这次由印度核试验引发的紧张局势中，双方在克什米尔问题上，既有“舌战”，也有“热战”，双方在实际控制线地区曾有过交火。",
      "gold": [
        {
          "text": "印度",
          "type": "location"
        },
        {
          "text": "克什米尔",
          "type": "location"
        }
      ],
      "zero_shot": [
        {
          "text": "印度",
          "type": "location"
        },
        {
          "text": "克什米尔",
          "type": "location"
        }
      ],
      "few_shot": [
        {
          "text": "印度",
          "type": "location"
        },
        {
          "text": "克什米尔",
          "type": "location"
        }
      ]
    },
    {
      "text": "五十年代初期和中期，现实生活中涌现许多好人好事，形成良好的社会风尚。",
      "gold": [],
      "zero_shot": [],
      "few_shot": []
    },
    {
      "text": "要着力引导广大公务员树立为人民服务的价值观，用公务员全心全意为人民服务的实际行动，推动社会公德、职业道德、家庭美德建设，促进社会主义市场经济健康发展。",
      "gold": [],
      "zero_shot": [],
      "few_shot": []
    },
    {
      "text": "但在工厂里干了20多年的工人活，她怎么也适应不了商业这行，只好又一次下岗。",
      "gold": [],
      "zero_shot": [],
      "few_shot": []
    },
    {
      "text": "在5月26日的大搜捕中，56名恐怖分子被一网打尽；6月3日又有3人被缉拿归案。",
      "gold": [],
      "zero_shot": [],
      "few_shot": []
    },
    {
      "text": "精明的商人根据足球比赛90分钟再加上15分钟休息，设计出一种长达105分钟的电池组。",
      "gold": [],
      "zero_shot": [],
      "few_shot": []
    },
    {
      "text": "所谓智慧资本，说白了就是给脑袋定价。",
      "gold": [],
      "zero_shot": [],
      "few_shot": []
    },
    {
      "text": "亏损的大背景，就是外部环境的变化，买方市场的形成。",
      "gold": [],
      "zero_shot": [],
      "few_shot": []
    },
    {
      "text": "他腰硬腿硬，专业条件并不理想；加之从艺晚，意味着必须付出成倍的努力以夺取时间。",
      "gold": [],
      "zero_shot": [],
      "few_shot": []
    },
    {
      "text": "每次赛季横跨4个月，12支球队要打一百四五十场球，联赛的水平有没有提高？",
      "gold": [],
      "zero_shot": [],
      "few_shot": []
    },
    {
      "text": "有关统计表明，2000年全球汽车生产能力将达8000万辆，而那时的汽车总需求很可能只有6000万辆上下。",
      "gold": [],
      "zero_shot": [],
      "few_shot": []
    },
    {
      "text": "党政『一把手』要自觉营造团结氛围，尽可能地做到『素质互补，气质互补，优势互补』，同唱一台富民『戏』，同下一盘经济发展『棋』，同操一把量人用人『尺』。",
      "gold": [],
      "zero_shot": [],
      "few_shot": []
    },
    {
      "text": "司法部门已建立劳教戒毒场所86个，在所劳教戒毒人员9万多人，是历年来人数最多的。",
      "gold": [],
      "zero_shot": [],
      "few_shot": []
    },
    {
      "text": "两条主钢缆各长约4公里、重约5万吨，被称为大桥的生命线。",
      "gold": [],
      "zero_shot": [],
      "few_shot": []
    },
    {
      "text": "越吃嘴越馋，越馋就越是变着法儿向农民收钱再吃再喝。",
      "gold": [],
      "zero_shot": [],
      "few_shot": []
    },
    {
      "text": "而企业每月为职工缴纳的养老保险费将从今年的24％，逐年减少到22％。",
      "gold": [],
      "zero_shot": [],
      "few_shot": []
    },
    {
      "text": "他们身上未着任何雨具，一任雨淋风吹，队形不乱，笑颜不改。",
      "gold": [],
      "zero_shot": [],
      "few_shot": []
    },
    {
      "text": "打响名牌，选择市场突破口是关键。",
      "gold": [],
      "zero_shot": [],
      "few_shot": []
    },
    {
      "text": "“我不仅教给他们我所掌握的知识，我也从他们那里学到很多。”",
      "gold": [],
      "zero_shot": [],
      "few_shot": []
    },
    {
      "text": "符合饮用和工农业用水标准的河段不足15％。",
      "gold": [],
      "zero_shot": [],
      "few_shot": []
    },
    {
      "text": "单位的电灯，不少是开时容易关时难，人走灯仍亮，天亮灯不灭；企业的物资管理混乱，大手大脚，不计工本，有用的零配件、原材料，在垃圾堆里比比皆是，造就了不少『垃圾暴发户』。",
      "gold": [],
      "zero_shot": [],
      "few_shot": []
    },
    {
      "text": "母亲小心翼翼地把母爱沏入儿女们的茶壶里，使原本平淡如水的人生变成了一杯酽酽的香茗。",
      "gold": [],
      "zero_shot": [],
      "few_shot": []
    },
    {
      "text": "他说，新美钞很难仿冒，但任何技术都不可能做到绝对无法仿冒。",
      "gold": [],
      "zero_shot": [],
      "few_shot": []
    },
    {
      "text": "在e组和f组各队25日比赛后，几支队的教练相继对新闻界发表讲话。",
      "gold": [],
      "zero_shot": [],
      "few_shot": []
    },
    {
      "text": "因此，有必要也有可能消灭资本主义私有制，建立社会主义公有制，以解决社会化大生产与生产资料私人占有制的矛盾，使生产力进一步得到解放和发展。",
      "gold": [],
      "zero_shot": [],
      "few_shot": []
    },
    {
      "text": "不过，人世间的许多事，往往相生相克，有一害便有一治。",
      "gold": [],
      "zero_shot": [],
      "few_shot": []
    },
    {
      "text": "据说，由于现代科学技术发展太快，法官审理案件时也经常遇到自己不懂的专业知识。",
      "gold": [],
      "zero_shot": [],
      "few_shot": []
    },
    {
      "text": "知识经济和信息经济也有区别的一面。",
      "gold": [],
      "zero_shot": [],
      "few_shot": []
    },
    {
      "text": "从长远看，欧元将有可能挑战、甚至取代美元的主导地位，促使国际金融格局进行重大调整。",
      "gold": [],
      "zero_shot": [],
      "few_shot": []
    },
    {
      "text": "比如，她会冷不丁弄出声巨响，毛手毛脚地打碎点儿什么。",
      "gold": [],
      "zero_shot": [],
      "few_shot": []
    },
    {
      "text": "据说在15世纪，有个王子放飞一个用粗绳子牵引的特大风筝，肆意割断村民们的细线风筝，以此取乐。",
      "gold": [],
      "zero_shot": [],
      "few_shot": []
    }
  ]
}
