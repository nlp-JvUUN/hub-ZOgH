(py312) ➜  function_call_mcp_cli python mode_function_call/run_function_call.py --question "查询下今北京的天气"    
[rag_backend] 就绪：10353 个向量，10353 条元数据
[Function Call] provider=deepseek model=glm-5.1

============================================================
Q1：查询下今北京的天气
============================================================
--tool_calls->$[ChatCompletionMessageFunctionToolCall(id='call_2db7381f5c6441389690fadb', function=Function(arguments='{"city": "北京"}', name='get_city_latAndlon'), type='function', index=0)]
  → [tool] get_city_latAndlon({'city': '北京'})
    ↩ {"latitude": 39.9075, "longitude": 116.39723, "city_name": "北京", "country": "中国", "admin1": "北京市"}

--tool_calls->$[ChatCompletionMessageFunctionToolCall(id='call_e115fd6da9f94b078a35ea5d', function=Function(arguments='{"lat": 39.9075, "lon": 116.39723, "city_name": "北京"}', name='get_weather_by_latlon'), type='function', index=0)]
  → [tool] get_weather_by_latlon({'lat': 39.9075, 'lon': 116.39723, 'city_name': '北京'})
    ↩ 【北京】天气报告 坐标：39.91°N, 116.40°E  当前天气：阴天   温度：26.9°C   相对湿度：77%   风速：5.8 km/h  未来3天预报：   2026-07-16：小毛毛雨，33.2°C / 25.8°C，降...

--tool_calls->$None
  → [llm] 最终回答（18.6s）

最终回答：
以下是**北京**的天气情况：

---

### 🌫 当前天气
| 项目 | 数据 |
|------|------|
| 天气状况 | 阴天 |
| 温度 | **26.9°C** |
| 相对湿度 | **77%** |
| 风速 | **5.8 km/h** |

---

### 📅 未来3天预报

| 日期 | 天气 | 最高温 / 最低温 | 降水量 |
|------|------|------------------|--------|
| 7月16日 | 小毛毛雨 | 33.2°C / 25.8°C | 0.1 mm |
| 7月17日 | 阴天 | 32.9°C / 25.5°C | 0.0 mm |
| 7月18日 | 大毛毛雨 | 30.7°C / 23.9°C | 3.8 mm |

---

整体来看，北京近几天气温偏高，湿度较大，未来几天以阴天和零星小雨为主，建议出行时随身携带雨伞，注意防暑降温。 ☂️
