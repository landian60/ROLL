"""
Personal Proxy Web Demo
使用 Flask 创建 Web 界面，接入 DeepSeek API
"""
import json
import os
import random
from datetime import datetime
from typing import Dict, List, Optional
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# 数据文件路径
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
PERSONAL_FACTS_FILE = os.path.join(DATA_DIR, 'personal_facts.json')
PERSONAL_PREFERENCES_FILE = os.path.join(DATA_DIR, 'personal_preferences.json')
INTENT_HISTORY_FILE = os.path.join(DATA_DIR, 'intent_history.json')

# DeepSeek API 配置（阿里云）
DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY', '')
DEEPSEEK_BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
DEEPSEEK_MODEL = 'deepseek-v3.2-exp'

# 初始化数据目录
os.makedirs(DATA_DIR, exist_ok=True)


class PersonalProxyWeb:
    """Personal Proxy Web 版本的核心逻辑"""
    
    def __init__(self):
        self.client = None
        if DASHSCOPE_API_KEY:
            self.client = OpenAI(
                api_key=DASHSCOPE_API_KEY,
                base_url=DEEPSEEK_BASE_URL
            )
        
        # 知识搜索场景
        self.knowledge_search_scenarios = [
            {
                "context": "查询上海旅游规划",
                "description": "用户想要了解上海旅游的详细规划，包括景点推荐、行程安排、美食推荐等"
            },
            {
                "context": "查找强化学习的代码库的相关总结",
                "description": "用户想要查找关于强化学习相关的代码库，并需要相关的总结和推荐"
            },
            {
                "context": "了解Python异步编程的最佳实践",
                "description": "用户想要学习Python异步编程的相关知识和最佳实践"
            },
            {
                "context": "查询机器学习模型部署方案",
                "description": "用户想要了解如何部署机器学习模型，包括各种部署方案和工具"
            },
            {
                "context": "查找自然语言处理的最新研究进展",
                "description": "用户想要了解自然语言处理领域的最新研究进展和论文"
            }
        ]
    
    def load_personal_facts(self) -> Dict:
        """加载个人事实"""
        if os.path.exists(PERSONAL_FACTS_FILE):
            with open(PERSONAL_FACTS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            "age": "",
            "gender": "",
            "employment_status": "",
            "education_level": "",
            "residence_area": ""
        }
    
    def save_personal_facts(self, facts: Dict):
        """保存个人事实"""
        with open(PERSONAL_FACTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(facts, f, ensure_ascii=False, indent=2)
    
    def load_personal_preferences(self) -> Dict:
        """加载个人偏好"""
        if os.path.exists(PERSONAL_PREFERENCES_FILE):
            with open(PERSONAL_PREFERENCES_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            "like_long_text": False,
            "like_outline_navigation": False
        }
    
    def save_personal_preferences(self, preferences: Dict):
        """保存个人偏好"""
        with open(PERSONAL_PREFERENCES_FILE, 'w', encoding='utf-8') as f:
            json.dump(preferences, f, ensure_ascii=False, indent=2)
    
    def load_intent_history(self) -> List[Dict]:
        """加载意图判断历史"""
        if os.path.exists(INTENT_HISTORY_FILE):
            with open(INTENT_HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def save_intent_history(self, history: List[Dict]):
        """保存意图判断历史"""
        with open(INTENT_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    
    def generate_random_personal_facts(self) -> Dict:
        """生成随机个人事实"""
        ages = ["20-25", "26-30", "31-35", "36-40", "41-45", "46-50"]
        genders = ["男", "女", "其他"]
        employment_statuses = ["在校学生", "在职", "待业", "自由职业", "退休"]
        education_levels = ["高中", "大专", "本科", "硕士", "博士"]
        residence_areas = ["北京", "上海", "广州", "深圳", "杭州", "成都", "其他"]
        
        return {
            "age": random.choice(ages),
            "gender": random.choice(genders),
            "employment_status": random.choice(employment_statuses),
            "education_level": random.choice(education_levels),
            "residence_area": random.choice(residence_areas)
        }
    
    def generate_random_personal_preferences(self) -> Dict:
        """生成随机个人偏好"""
        return {
            "like_long_text": random.choice([True, False]),
            "like_outline_navigation": random.choice([True, False])
        }
    
    def generate_random_intent_history(self, count: int = 3) -> List[Dict]:
        """生成随机意图判断历史"""
        history = []
        scenarios = random.sample(self.knowledge_search_scenarios, min(count, len(self.knowledge_search_scenarios)))
        
        intents = [
            "获取详细的旅游攻略和行程安排",
            "查找高质量的代码库和项目总结",
            "学习技术知识和最佳实践",
            "了解最新的研究进展",
            "获取实用的部署方案和工具推荐"
        ]
        
        intent_explanations = [
            "用户希望通过搜索获得详细的、可执行的旅游规划信息",
            "用户需要找到相关的代码库并了解其特点和适用场景",
            "用户想要系统性地学习某个技术领域的知识",
            "用户希望了解该领域的最新研究动态和趋势",
            "用户需要实用的解决方案和工具推荐"
        ]
        
        for i, scenario in enumerate(scenarios):
            history.append({
                "id": f"intent_{i+1}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "context": scenario["context"],
                "user": "当前用户",
                "intent": intents[i % len(intents)],
                "intent_explanation": intent_explanations[i % len(intent_explanations)],
                "user_feedback": "",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            })
        
        return history
    
    def retrieve_intent_from_history(self, context: str) -> Optional[Dict]:
        """从历史中检索相似情景的意图"""
        history = self.load_intent_history()
        if not history:
            return None
        
        # 简单的文本匹配
        context_lower = context.lower()
        for h in history:
            if context_lower in h.get("context", "").lower() or h.get("context", "").lower() in context_lower:
                return h
        
        return None
    
    def retrieve_personal_memory(self, intent: str, personal_facts: Dict, personal_preferences: Dict) -> Dict:
        """
        使用Proxy模型根据意图从总体个人记忆中归纳出相关的个人记忆
        返回：{"memory": 相关记忆文本, "thinking": 推理过程}
        """
        # 构建完整的个人记忆
        all_facts = []
        if personal_facts.get("age"):
            all_facts.append(f"年龄：{personal_facts['age']}")
        if personal_facts.get("gender"):
            all_facts.append(f"性别：{personal_facts['gender']}")
        if personal_facts.get("employment_status"):
            all_facts.append(f"就业状态：{personal_facts['employment_status']}")
        if personal_facts.get("education_level"):
            all_facts.append(f"教育水平：{personal_facts['education_level']}")
        if personal_facts.get("residence_area"):
            all_facts.append(f"居住区域：{personal_facts['residence_area']}")
        
        all_preferences = []
        if personal_preferences.get("like_long_text"):
            all_preferences.append("喜欢阅读长文本")
        else:
            all_preferences.append("不喜欢阅读长文本，偏好简洁内容")
        if personal_preferences.get("like_outline_navigation"):
            all_preferences.append("喜欢大纲导览")
        else:
            all_preferences.append("不喜欢大纲导览，偏好直接内容")
        
        if not all_facts and not all_preferences:
            return {"memory": "暂无个人记忆", "thinking": ""}
        
        # 构建完整记忆文本
        all_memory_text = ""
        if all_facts:
            all_memory_text += "个人事实（长期）：\n" + "\n".join(f"- {f}" for f in all_facts) + "\n\n"
        if all_preferences:
            all_memory_text += "个人偏好（短期）：\n" + "\n".join(f"- {p}" for p in all_preferences)
        
        # 使用Proxy模型归纳相关记忆
        memory_prompt = f"""基于以下用户意图，从总体个人记忆中归纳出与当前意图相关的个人记忆。

用户意图：{intent}

总体个人记忆：
{all_memory_text}

请分析并归纳出与当前意图最相关的个人记忆，只保留真正相关的信息。
如果某些个人记忆与当前意图无关，请忽略它们。

请给出：
1. 相关的个人记忆（只包含与意图相关的部分）
2. 简要说明为什么这些记忆与意图相关

格式：
相关记忆：
[归纳后的相关记忆]

相关性说明：
[说明]"""
        
        result = self.call_deepseek_api(memory_prompt, "你是一个个人记忆分析专家，擅长从总体记忆中提取与特定意图相关的信息。", enable_thinking=True)
        
        # 解析相关记忆
        import re
        memory_match = re.search(r"相关记忆[：:]\s*(.+?)(?=相关性说明|$)", result["content"], re.DOTALL)
        if memory_match:
            relevant_memory = memory_match.group(1).strip()
        else:
            # 如果没有匹配到，使用全部记忆
            relevant_memory = all_memory_text
        
        return {
            "memory": relevant_memory,
            "thinking": result["thinking"]
        }
    
    def generate_personal_description(self, context: str, intent: str, intent_explanation: str, 
                                     personal_memory: str) -> str:
        """生成个性化描述"""
        description = f"""当前情景：{context}

用户意图：{intent}

意图解释：{intent_explanation}

{personal_memory}

基于以上信息，这是一个关于用户在当前知识搜索情景下的个性化描述。"""
        return description
    
    def call_deepseek_api(self, prompt: str, system_prompt: str = None, enable_thinking: bool = False) -> Dict:
        """调用 DeepSeek API，支持 thinking 过程"""
        if not self.client:
            return {
                "content": "DeepSeek API 未配置，请设置 DASHSCOPE_API_KEY 环境变量",
                "thinking": ""
            }
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            extra_body = {}
            if enable_thinking:
                extra_body["extend_fields"] = {"chat_template_kwargs": {"enable_thinking": True}}
            
            response = self.client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=4000,
                extra_body=extra_body if extra_body else None
            )
            
            message = response.choices[0].message
            content = message.content.strip() if message.content else ""
            
            # 提取 thinking 内容（deepseek-v3的thinking可能在多个位置）
            thinking = ""
            # 方法1：检查message对象是否有thinking属性
            if hasattr(message, 'thinking') and message.thinking:
                thinking = message.thinking.strip()
            # 方法2：检查是否有reasoning_content
            elif hasattr(message, 'reasoning_content') and message.reasoning_content:
                thinking = message.reasoning_content.strip()
            # 方法3：检查response对象本身
            elif hasattr(response, 'thinking') and response.thinking:
                thinking = response.thinking.strip()
            # 方法4：检查choices[0]是否有thinking
            elif hasattr(response.choices[0], 'thinking') and response.choices[0].thinking:
                thinking = response.choices[0].thinking.strip()
            # 方法5：尝试从content中提取（如果格式是<think>...</think>）
            if not thinking and '<think>' in content.lower():
                import re
                think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL | re.IGNORECASE)
                if think_match:
                    thinking = think_match.group(1).strip()
                    # 从content中移除thinking部分
                    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE).strip()
            
            return {
                "content": content,
                "thinking": thinking
            }
        except Exception as e:
            return {
                "content": f"API 调用失败：{str(e)}",
                "thinking": ""
            }
    
    def judge_personal_description(self, personal_description: str, context: str) -> Dict:
        """使用 DeepSeek API 对个人描述进行打分"""
        prompt = f"""请评估以下个人描述的质量，从0到1打分（1为最高分）。

当前情景：{context}

个人描述：
{personal_description}

评估标准：
1. 意图理解是否准确（0.3分）
2. 个人记忆是否相关且准确（0.3分）
3. 描述是否完整且有意义（0.2分）
4. 是否能够帮助理解用户在当前情景下的个性化需求（0.2分）

请给出分数（0-1之间的浮点数）和评分理由。
格式：
分数：[0-1之间的数字]
理由：[详细说明]"""
        
        system_prompt = "你是一个专业的评估专家，负责评估个人描述的质量。"
        result = self.call_deepseek_api(prompt, system_prompt)
        response = result["content"]
        
        # 解析分数和理由
        import re
        score_match = re.search(r"分数[：:]\s*([0-9.]+)", response)
        if score_match:
            score = float(score_match.group(1))
            score = max(0.0, min(1.0, score))
        else:
            score = 0.5
        
        reason_match = re.search(r"理由[：:]\s*(.+)", response, re.DOTALL)
        if reason_match:
            reason = reason_match.group(1).strip()
        else:
            reason = response
        
        return {
            "score": score,
            "reason": reason,
            "full_response": response
        }


# 初始化 PersonalProxyWeb
proxy = PersonalProxyWeb()


@app.route('/')
def index():
    """主页（个人资料管理）"""
    return render_template('index.html')


@app.route('/search')
def search():
    """搜索页面"""
    return render_template('search.html')


@app.route('/api/personal-facts', methods=['GET'])
def get_personal_facts():
    """获取个人事实"""
    facts = proxy.load_personal_facts()
    return jsonify({"success": True, "data": facts})


@app.route('/api/personal-facts', methods=['POST'])
def update_personal_facts():
    """更新个人事实"""
    data = request.json
    proxy.save_personal_facts(data)
    return jsonify({"success": True, "message": "个人事实已更新"})


@app.route('/api/personal-facts/random', methods=['POST'])
def generate_random_facts():
    """生成随机个人事实"""
    facts = proxy.generate_random_personal_facts()
    proxy.save_personal_facts(facts)
    return jsonify({"success": True, "data": facts})


@app.route('/api/personal-preferences', methods=['GET'])
def get_personal_preferences():
    """获取个人偏好"""
    preferences = proxy.load_personal_preferences()
    return jsonify({"success": True, "data": preferences})


@app.route('/api/personal-preferences', methods=['POST'])
def update_personal_preferences():
    """更新个人偏好"""
    data = request.json
    proxy.save_personal_preferences(data)
    return jsonify({"success": True, "message": "个人偏好已更新"})


@app.route('/api/personal-preferences/random', methods=['POST'])
def generate_random_preferences():
    """生成随机个人偏好"""
    preferences = proxy.generate_random_personal_preferences()
    proxy.save_personal_preferences(preferences)
    return jsonify({"success": True, "data": preferences})


@app.route('/api/intent-history', methods=['GET'])
def get_intent_history():
    """获取意图判断历史"""
    history = proxy.load_intent_history()
    return jsonify({"success": True, "data": history})


@app.route('/api/intent-history', methods=['POST'])
def add_intent_history():
    """添加意图判断历史"""
    data = request.json
    history = proxy.load_intent_history()
    
    new_item = {
        "id": f"intent_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "context": data.get("context", ""),
        "user": data.get("user", "当前用户"),
        "intent": data.get("intent", ""),
        "intent_explanation": data.get("intent_explanation", ""),
        "user_feedback": data.get("user_feedback", ""),
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }
    
    history.append(new_item)
    proxy.save_intent_history(history)
    return jsonify({"success": True, "data": new_item})


@app.route('/api/intent-history/<path:intent_id>', methods=['PUT'])
def update_intent_history(intent_id):
    """更新意图判断历史"""
    data = request.json
    history = proxy.load_intent_history()
    
    for i, item in enumerate(history):
        if str(item.get("id")) == str(intent_id):
            history[i].update({
                "context": data.get("context", item.get("context")),
                "user": data.get("user", item.get("user")),
                "intent": data.get("intent", item.get("intent")),
                "intent_explanation": data.get("intent_explanation", item.get("intent_explanation")),
                "user_feedback": data.get("user_feedback", item.get("user_feedback")),
                "updated_at": datetime.now().isoformat()
            })
            proxy.save_intent_history(history)
            return jsonify({"success": True, "data": history[i]})
    
    return jsonify({"success": False, "message": "未找到对应的意图历史"}), 404


@app.route('/api/intent-history/<path:intent_id>', methods=['DELETE'])
def delete_intent_history(intent_id):
    """删除意图判断历史"""
    history = proxy.load_intent_history()
    history = [h for h in history if str(h.get("id")) != str(intent_id)]
    proxy.save_intent_history(history)
    return jsonify({"success": True, "message": "已删除"})


@app.route('/api/intent-history/random', methods=['POST'])
def generate_random_history():
    """生成随机意图判断历史"""
    count = request.json.get("count", 3) if request.json else 3
    history = proxy.generate_random_intent_history(count)
    existing = proxy.load_intent_history()
    existing.extend(history)
    proxy.save_intent_history(existing)
    return jsonify({"success": True, "data": history})


@app.route('/api/scenarios', methods=['GET'])
def get_scenarios():
    """获取知识搜索场景列表"""
    return jsonify({"success": True, "data": proxy.knowledge_search_scenarios})


@app.route('/api/generate-description', methods=['POST'])
def generate_description():
    """生成个性化描述"""
    data = request.json
    context = data.get("context", "")
    
    # 加载数据
    personal_facts = proxy.load_personal_facts()
    personal_preferences = proxy.load_personal_preferences()
    
    # 检索意图
    intent_history_item = proxy.retrieve_intent_from_history(context)
    if intent_history_item:
        intent = intent_history_item.get("intent", "")
        intent_explanation = intent_history_item.get("intent_explanation", "")
    else:
        # 使用 DeepSeek 生成意图
        intent_prompt = f"""基于以下知识搜索情景，推断用户的意图和意图解释。

情景：{context}

请给出：
1. 用户意图（一句话概括）
2. 意图解释（详细说明为什么会有这个意图）

格式：
意图：[意图内容]
解释：[解释内容]"""
        
        result = proxy.call_deepseek_api(intent_prompt, "你是一个意图理解专家。")
        response = result["content"]
        
        # 解析意图和解释
        import re
        intent_match = re.search(r"意图[：:]\s*(.+)", response)
        explanation_match = re.search(r"解释[：:]\s*(.+)", response, re.DOTALL)
        
        intent = intent_match.group(1).strip() if intent_match else "获取相关信息"
        intent_explanation = explanation_match.group(1).strip() if explanation_match else "用户希望通过搜索获得相关信息"
    
    # 检索个人记忆
    memory_result = proxy.retrieve_personal_memory(intent, personal_facts, personal_preferences)
    personal_memory = memory_result["memory"]
    
    # 生成个性化描述
    personal_description = proxy.generate_personal_description(
        context, intent, intent_explanation, personal_memory
    )
    
    # 打分
    judge_result = proxy.judge_personal_description(personal_description, context)
    
    return jsonify({
        "success": True,
        "data": {
            "context": context,
            "intent": intent,
            "intent_explanation": intent_explanation,
            "personal_memory": personal_memory,
            "personal_description": personal_description,
            "score": judge_result["score"],
            "score_reason": judge_result["reason"]
        }
    })


@app.route('/api/generate-description', methods=['POST'])
def generate_personal_description_api():
    """
    生成个性化描述API（独立使用，可后续用于搜索）
    使用两轮推理：
    1. 第一轮：推理意图类别，进行历史检索
    2. 第二轮：给出意图类别判断和意图解释
    """
    data = request.json
    query = data.get("query", "")
    
    if not query:
        return jsonify({"success": False, "message": "查询内容不能为空"}), 400
    
    # 加载数据
    personal_facts = proxy.load_personal_facts()
    personal_preferences = proxy.load_personal_preferences()
    intent_history = proxy.load_intent_history()
    
    # 提取意图类别列表
    intent_categories = []
    for h in intent_history:
        intent_cat = h.get("intent", "").split("：")[0] if "：" in h.get("intent", "") else h.get("intent", "")
        if intent_cat and intent_cat not in intent_categories:
            intent_categories.append(intent_cat)
    
    intent_categories_text = "\n".join([f"- {cat}" for cat in intent_categories]) if intent_categories else "暂无历史意图类别"
    
    # ========== 第一轮：推理意图类别并进行历史检索 ==========
    round1_prompt = f"""基于以下知识搜索查询，推理用户的意图类别，并从历史意图中进行检索。

查询：{query}

历史意图类别：
{intent_categories_text}

历史意图判断记录：
{json.dumps(intent_history[:10], ensure_ascii=False, indent=2) if intent_history else "暂无历史记录"}

请：
1. 分析查询内容，推理可能的意图类别
2. 从历史意图类别中检索相似的类别
3. 从历史意图判断记录中检索相似的记录

格式：
意图类别：[推理出的意图类别]
相似历史类别：[检索到的相似类别，如果没有则写"无"]
相似历史记录：[检索到的相似记录ID或内容，如果没有则写"无"]"""
    
    round1_result = proxy.call_deepseek_api(round1_prompt, "你是一个意图理解专家，擅长分析用户的搜索意图并进行历史检索。", enable_thinking=True)
    round1_thinking = round1_result["thinking"]
    round1_content = round1_result["content"]
    
    # 解析第一轮结果
    import re
    category_match = re.search(r"意图类别[：:]\s*(.+)", round1_content)
    similar_category_match = re.search(r"相似历史类别[：:]\s*(.+)", round1_content)
    similar_history_match = re.search(r"相似历史记录[：:]\s*(.+)", round1_content, re.DOTALL)
    
    intent_category = category_match.group(1).strip() if category_match else "信息查询"
    similar_category = similar_category_match.group(1).strip() if similar_category_match else "无"
    similar_history = similar_history_match.group(1).strip() if similar_history_match else "无"
    
    # ========== 第二轮：给出意图类别判断和意图解释 ==========
    round2_prompt = f"""基于第一轮的分析结果，给出最终的意图类别判断和详细的意图解释。

查询：{query}

第一轮分析结果：
- 推理的意图类别：{intent_category}
- 相似历史类别：{similar_category}
- 相似历史记录：{similar_history}

请：
1. 确定最终的意图类别
2. 给出详细的意图解释（说明为什么用户会有这个意图，以及用户的真实需求）

格式：
最终意图类别：[确定的意图类别]
意图解释：[详细的意图解释]"""
    
    round2_result = proxy.call_deepseek_api(round2_prompt, "你是一个意图分析专家，能够基于历史信息给出准确的意图判断和解释。", enable_thinking=True)
    round2_thinking = round2_result["thinking"]
    round2_content = round2_result["content"]
    
    # 解析第二轮结果
    final_category_match = re.search(r"最终意图类别[：:]\s*(.+)", round2_content)
    explanation_match = re.search(r"意图解释[：:]\s*(.+)", round2_content, re.DOTALL)
    
    final_intent_category = final_category_match.group(1).strip() if final_category_match else intent_category
    intent_explanation = explanation_match.group(1).strip() if explanation_match else "用户希望通过搜索获得相关信息"
    
    intent = f"{final_intent_category}：{query}"
    
    # 检索相关个人记忆
    memory_result = proxy.retrieve_personal_memory(intent, personal_facts, personal_preferences)
    personal_memory = memory_result["memory"]
    memory_thinking = memory_result["thinking"]
    
    # 生成个性化描述
    personal_description = proxy.generate_personal_description(
        query, intent, intent_explanation, personal_memory
    )
    
    # 保存意图判断历史
    new_history = {
        "id": f"intent_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "context": query,
        "user": "当前用户",
        "intent": intent,
        "intent_explanation": intent_explanation,
        "user_feedback": "",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }
    intent_history.append(new_history)
    proxy.save_intent_history(intent_history)
    
    return jsonify({
        "success": True,
        "data": {
            "query": query,
            "intent_category": final_intent_category,
            "intent": intent,
            "intent_explanation": intent_explanation,
            "personal_memory": personal_memory,
            "personal_description": personal_description,
            "round1_thinking": round1_thinking,
            "round2_thinking": round2_thinking,
            "memory_thinking": memory_thinking,
            "similar_category": similar_category,
            "similar_history": similar_history
        }
    })


@app.route('/api/search', methods=['POST'])
def search_with_personalization():
    """
    知识搜索API：使用两个模型
    1. Proxy模型：根据意图历史生成个性化描述（两轮推理）
    2. LLM模型：接受Query+个性化描述，生成最终回答（带thinking过程）
    """
    data = request.json
    query = data.get("query", "")
    personal_description = data.get("personal_description", "")  # 可选：如果已生成个性化描述，直接使用
    
    if not query:
        return jsonify({"success": False, "message": "查询内容不能为空"}), 400
    
    # 如果提供了个性化描述，直接使用；否则生成
    if not personal_description:
        # 直接调用生成个性化描述的逻辑（复用代码）
        personal_facts = proxy.load_personal_facts()
        personal_preferences = proxy.load_personal_preferences()
        intent_history = proxy.load_intent_history()
        
        # 提取意图类别列表
        intent_categories = []
        for h in intent_history:
            intent_cat = h.get("intent", "").split("：")[0] if "：" in h.get("intent", "") else h.get("intent", "")
            if intent_cat and intent_cat not in intent_categories:
                intent_categories.append(intent_cat)
        
        intent_categories_text = "\n".join([f"- {cat}" for cat in intent_categories]) if intent_categories else "暂无历史意图类别"
        
        # 第一轮：推理意图类别并进行历史检索
        round1_prompt = f"""基于以下知识搜索查询，推理用户的意图类别，并从历史意图中进行检索。

查询：{query}

历史意图类别：
{intent_categories_text}

历史意图判断记录：
{json.dumps(intent_history[:10], ensure_ascii=False, indent=2) if intent_history else "暂无历史记录"}

请：
1. 分析查询内容，推理可能的意图类别
2. 从历史意图类别中检索相似的类别
3. 从历史意图判断记录中检索相似的记录

格式：
意图类别：[推理出的意图类别]
相似历史类别：[检索到的相似类别，如果没有则写"无"]
相似历史记录：[检索到的相似记录ID或内容，如果没有则写"无"]"""
        
        round1_result = proxy.call_deepseek_api(round1_prompt, "你是一个意图理解专家，擅长分析用户的搜索意图并进行历史检索。", enable_thinking=True)
        round1_thinking = round1_result["thinking"]
        round1_content = round1_result["content"]
        
        # 解析第一轮结果
        import re
        category_match = re.search(r"意图类别[：:]\s*(.+)", round1_content)
        similar_category_match = re.search(r"相似历史类别[：:]\s*(.+)", round1_content)
        similar_history_match = re.search(r"相似历史记录[：:]\s*(.+)", round1_content, re.DOTALL)
        
        intent_category = category_match.group(1).strip() if category_match else "信息查询"
        similar_category = similar_category_match.group(1).strip() if similar_category_match else "无"
        similar_history = similar_history_match.group(1).strip() if similar_history_match else "无"
        
        # 第二轮：给出意图类别判断和意图解释
        round2_prompt = f"""基于第一轮的分析结果，给出最终的意图类别判断和详细的意图解释。

查询：{query}

第一轮分析结果：
- 推理的意图类别：{intent_category}
- 相似历史类别：{similar_category}
- 相似历史记录：{similar_history}

请：
1. 确定最终的意图类别
2. 给出详细的意图解释（说明为什么用户会有这个意图，以及用户的真实需求）

格式：
最终意图类别：[确定的意图类别]
意图解释：[详细的意图解释]"""
        
        round2_result = proxy.call_deepseek_api(round2_prompt, "你是一个意图分析专家，能够基于历史信息给出准确的意图判断和解释。", enable_thinking=True)
        round2_thinking = round2_result["thinking"]
        round2_content = round2_result["content"]
        
        # 解析第二轮结果
        final_category_match = re.search(r"最终意图类别[：:]\s*(.+)", round2_content)
        explanation_match = re.search(r"意图解释[：:]\s*(.+)", round2_content, re.DOTALL)
        
        final_intent_category = final_category_match.group(1).strip() if final_category_match else intent_category
        intent_explanation = explanation_match.group(1).strip() if explanation_match else "用户希望通过搜索获得相关信息"
        
        intent = f"{final_intent_category}：{query}"
        
        # 检索相关个人记忆（根据用户选择的记忆）
        selected_memories = data.get("selected_memories", [])
        # 如果用户选择了特定记忆，只使用选中的
        if selected_memories:
            filtered_facts = {}
            filtered_preferences = {}
            for mem in selected_memories:
                if mem["type"] == "fact" and mem["key"] in personal_facts:
                    filtered_facts[mem["key"]] = personal_facts[mem["key"]]
                elif mem["type"] == "preference" and mem["key"] in personal_preferences:
                    filtered_preferences[mem["key"]] = personal_preferences[mem["key"]]
            memory_result = proxy.retrieve_personal_memory(intent, filtered_facts, filtered_preferences)
        else:
            # 使用全部记忆
            memory_result = proxy.retrieve_personal_memory(intent, personal_facts, personal_preferences)
        personal_memory = memory_result["memory"]
        memory_thinking = memory_result["thinking"]
        
        # 生成个性化描述
        personal_description = proxy.generate_personal_description(
            query, intent, intent_explanation, personal_memory
        )
    else:
        # 如果提供了个性化描述，需要从描述中提取信息（简化处理）
        intent = data.get("intent", "信息查询")
        intent_explanation = data.get("intent_explanation", "")
        personal_memory = data.get("personal_memory", "")
        round1_thinking = ""
        round2_thinking = ""
        memory_thinking = ""
    
    # ========== LLM模型生成最终回答（带thinking过程）==========
    llm_prompt = f"""用户查询：{query}

个性化描述：
{personal_description}

请基于以上信息，为用户提供详细的回答。在回答时，请考虑用户的个性化需求和偏好。"""
    
    system_prompt = """你是一个智能知识搜索助手，能够根据用户的个性化需求提供精准的回答。
在回答时，请考虑用户的个人背景、偏好和意图，提供最符合用户需求的答案。"""
    
    # 调用LLM模型，启用thinking过程
    llm_result = proxy.call_deepseek_api(llm_prompt, system_prompt, enable_thinking=True)
    
    return jsonify({
        "success": True,
        "data": {
            "query": query,
            "intent": intent,
            "intent_explanation": intent_explanation,
            "personal_memory": personal_memory,
            "personal_description": personal_description,
            "answer": llm_result["content"],
            "llm_thinking": llm_result["thinking"],
            "round1_thinking": round1_thinking,
            "round2_thinking": round2_thinking,
            "memory_thinking": memory_thinking
        }
    })


if __name__ == '__main__':
    # 初始化一些随机数据（如果数据文件不存在）
    if not os.path.exists(PERSONAL_FACTS_FILE):
        proxy.save_personal_facts(proxy.generate_random_personal_facts())
    if not os.path.exists(PERSONAL_PREFERENCES_FILE):
        proxy.save_personal_preferences(proxy.generate_random_personal_preferences())
    if not os.path.exists(INTENT_HISTORY_FILE):
        proxy.save_intent_history(proxy.generate_random_intent_history(3))
    
    print("Personal Proxy Web Demo 启动中...")
    print(f"DeepSeek API Key: {'已配置' if DASHSCOPE_API_KEY else '未配置（请设置环境变量 DASHSCOPE_API_KEY）'}")
    print("=" * 60)
    print("应用已启动，可以通过以下地址访问：")
    print("  - http://localhost:5000")
    print("  - http://127.0.0.1:5000")
    print("=" * 60)
    print("如果无法访问，请检查：")
    print("  1. 防火墙是否允许 5000 端口")
    print("  2. 是否有其他程序占用 5000 端口")
    print("  3. 尝试使用 127.0.0.1:5000 而不是 localhost:5000")
    print("=" * 60)
    try:
        app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"\n错误：端口 5000 已被占用")
            print("请尝试：")
            print("  1. 关闭占用端口的程序")
            print("  2. 或修改 app.py 中的端口号")
        else:
            print(f"\n启动失败：{e}")
        raise

