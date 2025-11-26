"""
Personal Proxy Web Demo
使用 Flask 创建 Web 界面，接入 DeepSeek API
"""

import base64
import io
import json
import os
import re
from collections import defaultdict
from datetime import datetime
from functools import wraps
from typing import Dict, List, Optional

from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from flask_cors import CORS
from openai import OpenAI
from werkzeug.security import check_password_hash, generate_password_hash

# Sentence Transformers for embedding-based retrieval
try:
    from sentence_transformers import SentenceTransformer, util

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("警告: sentence-transformers 未安装，将无法使用 embedding 检索功能")

try:
    from PyPDF2 import PdfReader  # type: ignore

    PDF_READER_AVAILABLE = True
except ImportError:
    PdfReader = None
    PDF_READER_AVAILABLE = False
    print("警告: PyPDF2 未安装，PDF 文本将无法自动抽取")

# Transformers for local LLM
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None
    torch = None
    TRANSFORMERS_AVAILABLE = False
    print("警告: transformers 未安装，将无法使用本地 LLM 模型")

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app, supports_credentials=True)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 最大16MB文件上传
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_HTTPONLY"] = True

# 数据文件路径
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
USER_DATA_DIR = os.path.join(DATA_DIR, "users")
USERS_FILE = os.path.join(DATA_DIR, "users.json")
DEFAULT_USER_ID = "default"
USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{3,32}$")

# DeepSeek API 配置（阿里云）
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DEEPSEEK_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEEPSEEK_MODEL = "deepseek-v3.2-exp"
# 多模态模型配置
MULTIMODAL_MODEL = "qwen3-omni-flash"

# Embedding 模型配置
EMBEDDING_RETRIEVAL_MODEL = os.getenv(
    "EMBEDDING_RETRIEVAL_MODEL", "BAAI/bge-small-zh-v1.5"
)

# 本地 LLM 模型配置（用于意图判断）
LOCAL_LLM_MODEL_PATH = os.path.expanduser(
    "~/.cache/modelscope/hub/qwen/Qwen2___5-14B-Instruct-GPTQ-Int8/"
)

# 初始化数据目录
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(USER_DATA_DIR, exist_ok=True)

# 服务端口配置
SERVER_PORT = int(os.getenv("FLASK_RUN_PORT", 5001))

# PDF 解析配置
MAX_PDF_PAGES = int(os.getenv("PREFERENCE_MAX_PDF_PAGES", "15"))
MAX_PDF_TEXT_LENGTH = int(os.getenv("PREFERENCE_MAX_PDF_TEXT_LENGTH", "20000"))


class EmbeddingModels:
    """管理 Embedding 模型的单例类，用于情景匹配检索"""

    _instance = None
    _retrieval_model = None  # 召回模型: BAAI/bge-large-zh (Bi-Encoder)

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_retrieval_model(self):
        """获取召回模型 (BAAI/bge-large-zh) - Bi-Encoder"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise RuntimeError("sentence-transformers 未安装，无法使用 embedding 检索")

        if self._retrieval_model is None:
            print(f"加载召回模型: {EMBEDDING_RETRIEVAL_MODEL} ...")
            self._retrieval_model = SentenceTransformer(EMBEDDING_RETRIEVAL_MODEL)
            print("召回模型加载完成")
        return self._retrieval_model


class LocalLLMModels:
    """管理本地 LLM 模型的单例类，用于意图判断"""

    _instance = None
    _local_model = None
    _local_tokenizer = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_local_llm(self):
        """获取本地 LLM 模型和 tokenizer"""
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers 未安装，无法使用本地 LLM 模型")

        if self._local_model is None or self._local_tokenizer is None:
            print(f"加载本地 LLM 模型: {LOCAL_LLM_MODEL_PATH} ...")
            try:
                self._local_tokenizer = AutoTokenizer.from_pretrained(
                    LOCAL_LLM_MODEL_PATH,
                    trust_remote_code=True
                )
                self._local_model = AutoModelForCausalLM.from_pretrained(
                    LOCAL_LLM_MODEL_PATH,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                )
                print("本地 LLM 模型加载完成")
            except Exception as e:
                print(f"本地 LLM 模型加载失败: {str(e)}")
                raise
        return self._local_model, self._local_tokenizer


# 全局 Embedding 模型实例
embedding_models = EmbeddingModels()

# 全局本地 LLM 模型实例
local_llm_models = LocalLLMModels()


class PersonalProxyWeb:
    """Personal Proxy Web 版本的核心逻辑"""

    def __init__(self):
        self.client = None
        if DASHSCOPE_API_KEY:
            self.client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DEEPSEEK_BASE_URL)

    def _resolve_user_dir(self, user_id: Optional[str] = None) -> str:
        effective_user = user_id or DEFAULT_USER_ID
        user_dir = os.path.join(USER_DATA_DIR, effective_user)
        os.makedirs(user_dir, exist_ok=True)
        return user_dir

    def _user_file(self, filename: str, user_id: Optional[str] = None) -> str:
        user_dir = self._resolve_user_dir(user_id)
        return os.path.join(user_dir, filename)

    def extract_pdf_text(
        self,
        file_data: bytes,
        max_pages: int = MAX_PDF_PAGES,
        max_chars: int = MAX_PDF_TEXT_LENGTH,
    ) -> str:
        """提取 PDF 文本，限制页数与字符数，失败时返回空字符串"""
        if not (PDF_READER_AVAILABLE and PdfReader):
            print("[PDF解析] PyPDF2 未就绪，跳过文本抽取")
            return ""

        try:
            print(f"[PDF解析] 初始化 PdfReader，文件大小 {len(file_data)} bytes")
            reader = PdfReader(io.BytesIO(file_data))
        except Exception as exc:
            print(f"[PDF解析] 加载失败: {exc}")
            return ""

        collected: List[str] = []
        total_chars = 0
        for page_idx, page in enumerate(reader.pages):
            if page_idx >= max_pages:
                break
            try:
                page_text = page.extract_text()
            except Exception as exc:
                print(f"[PDF解析] 第{page_idx + 1}页提取失败: {exc}")
                continue
            if not page_text:
                continue
            page_text = page_text.strip()
            if not page_text:
                continue
            remaining_chars = max_chars - total_chars
            if remaining_chars <= 0:
                break
            if len(page_text) > remaining_chars:
                page_text = page_text[:remaining_chars] + "..."
            collected.append(page_text)
            total_chars += len(page_text)
            if total_chars >= max_chars:
                break

        return "\n\n".join(collected).strip()

    def save_pdf_text_file(
        self,
        text: str,
        original_filename: str,
        user_id: Optional[str] = None,
    ) -> Optional[str]:
        """将 PDF 文本保存为 .txt 文件，返回路径"""
        if not text:
            return None

        base_name = os.path.splitext(original_filename or "pdf_document")[0]
        safe_base = re.sub(r"[^a-zA-Z0-9_.-]", "_", base_name).strip("_") or "pdf_document"
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        txt_filename = f"{safe_base}_{timestamp}.txt"

        user_dir = self._resolve_user_dir(user_id)
        output_dir = os.path.join(user_dir, "pdf_text")
        os.makedirs(output_dir, exist_ok=True)
        txt_path = os.path.join(output_dir, txt_filename)

        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
            return txt_path
        except OSError as exc:
            print(f"[PDF解析] 文本保存失败: {exc}")
            return None

    def load_personal_facts(self, user_id: Optional[str] = None) -> Dict:
        """加载个人事实"""
        file_path = self._user_file("personal_facts.json", user_id)
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "age": "",
            "gender": "",
            "employment_status": "",
            "education_level": "",
            "residence_area": "",
        }

    def save_personal_facts(self, facts: Dict, user_id: Optional[str] = None):
        """保存个人事实"""
        file_path = self._user_file("personal_facts.json", user_id)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(facts, f, ensure_ascii=False, indent=2)

    def load_personal_preferences(self, user_id: Optional[str] = None) -> List[Dict]:
        """加载个人偏好（短期）- 返回偏好列表"""
        file_path = self._user_file("personal_preferences.json", user_id)
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # 兼容旧格式
                if isinstance(data, dict) and not isinstance(data, list):
                    # 转换旧格式为新格式
                    preferences = []
                    if data.get("like_long_text"):
                        preferences.append(
                            {
                                "id": f"pref_{datetime.now().strftime('%Y%m%d%H%M%S')}_1",
                                "text": "喜欢阅读长文本",
                                "selected": True,
                                "preference_type": "like",
                            }
                        )
                    if data.get("like_outline_navigation"):
                        preferences.append(
                            {
                                "id": f"pref_{datetime.now().strftime('%Y%m%d%H%M%S')}_2",
                                "text": "喜欢大纲导览",
                                "selected": True,
                                "preference_type": "like",
                            }
                        )
                    self.save_personal_preferences(preferences, user_id)
                    return preferences
                # 确保旧数据有preference_type字段
                if isinstance(data, list):
                    for pref in data:
                        if "preference_type" not in pref:
                            pref["preference_type"] = "like"  # 默认为"喜欢"
                    return data
                return []
        # 默认偏好列表
        return []

    def save_personal_preferences(
        self, preferences: List[Dict], user_id: Optional[str] = None
    ):
        """保存个人偏好（短期）- 保存偏好列表"""
        file_path = self._user_file("personal_preferences.json", user_id)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(preferences, f, ensure_ascii=False, indent=2)

    def load_intent_history(self, user_id: Optional[str] = None) -> List[Dict]:
        """加载意图判断历史"""
        file_path = self._user_file("intent_history.json", user_id)
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def save_intent_history(self, history: List[Dict], user_id: Optional[str] = None):
        """保存意图判断历史"""
        file_path = self._user_file("intent_history.json", user_id)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    def load_intent_categories(self, user_id: Optional[str] = None) -> List[Dict]:
        file_path = self._user_file("intent_categories.json", user_id)
        user_categories: List[Dict] = []
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        user_categories = data
                        # 如果用户已有类别且数量为8个，直接返回
                        if len(user_categories) == 8:
                            return user_categories
            except json.JSONDecodeError:
                pass

        default_file = os.path.join(DATA_DIR, "default_intent_categories.json")
        default_categories: List[Dict] = []
        if os.path.exists(default_file):
            try:
                with open(default_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list) and len(data) == 8:
                        default_categories = data
            except json.JSONDecodeError:
                default_categories = []

        # 如果从文件加载失败或为空，使用硬编码的8个默认类别
        if not default_categories:
            default_categories = [
                {
                    "id": "intent_type_1",
                    "name": "定义-论证式",
                    "generic_explanation": '通过精确给出概念、对象或现象的定义，并配合分层论证，帮助读者建立清晰的认知框架。常用于回答"是什么、由什么构成、遵循什么基本原理"，行文客观、结构清晰，可由定义自然引出特征、结构与作用。',
                    "order": 1
                },
                {
                    "id": "intent_type_2",
                    "name": "故事-叙事式",
                    "generic_explanation": '以时间线或关键人物为线索，通过"起因—发展—转折—结果"的叙事结构呈现主题。强调情节与场景感，让读者在故事推进中自然理解背景、动机与影响，增强代入感和记忆度。',
                    "order": 2
                },
                {
                    "id": "intent_type_3",
                    "name": "示例-归纳式",
                    "generic_explanation": '先给出若干具有代表性的具体案例或情境，再从中归纳共同规律与结论。强调"从个体到整体"的逻辑，用细节支撑抽象观点，适合用于解释概念、提炼特征或总结经验。',
                    "order": 3
                },
                {
                    "id": "intent_type_4",
                    "name": "类比-关联式",
                    "generic_explanation": '将抽象或陌生的对象，与读者熟悉的事物建立类比，通过结构或功能上的对应关系帮助理解。强调"陌生概念—熟悉场景"的映射，使复杂内容变得直观易感知。',
                    "order": 4
                },
                {
                    "id": "intent_type_5",
                    "name": "结构-分解式",
                    "generic_explanation": '从整体出发，自上而下拆解主题的组成部分与层级关系，形成清晰的结构框架。强调"整体 → 层次 → 子项 → 细节"的展开方式，适用于系统说明、框架搭建和知识梳理。',
                    "order": 5
                },
                {
                    "id": "intent_type_6",
                    "name": "对比-对照式",
                    "generic_explanation": '将两个或多个对象、阶段或方案并列展示，通过差异与共性来突出变化、优劣或趋势。常采用表格或成对描述，强调"相同点—不同点—带来的影响"，帮助读者快速形成判断。',
                    "order": 6
                },
                {
                    "id": "intent_type_7",
                    "name": "模型-抽象式",
                    "generic_explanation": "用概念模型、图式、公式或伪代码等抽象形式，浓缩主题的核心机制或流程。突出变量、阶段、输入输出等要素之间的关系，适用于模式总结、原理提炼与逻辑建模。",
                    "order": 7
                },
                {
                    "id": "intent_type_8",
                    "name": "实验-验证式",
                    "generic_explanation": '围绕某个假设或观点，按照"假设—方法—过程—结果—结论"的路径展开，通过数据、观察或对照验证主张。强调可检验性与证据链，适用于论证可行性、有效性或因果关系。',
                    "order": 8
                }
            ]

        categories: List[Dict] = []
        for item in default_categories:
            category = {
                "id": item.get("id"),
                "name": item.get("name", ""),
                "generic_explanation": item.get("generic_explanation", ""),
                "order": item.get("order", 0),
            }
            categories.append(category)
        self.save_intent_categories(categories, user_id)
        return categories

    def save_intent_categories(
        self, categories: List[Dict], user_id: Optional[str] = None
    ):
        file_path = self._user_file("intent_categories.json", user_id)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(categories, f, ensure_ascii=False, indent=2)

    def identify_intent_category(
        self, llm_intent: str, intent_categories: List[Dict]
    ) -> str:
        """
        根据 LLM 的输出识别意图类别，支持模糊匹配，尽量保留 LLM 判断结果
        """
        if not intent_categories:
            return llm_intent or "未分类"

        allowed_intent_names = [cat["name"] for cat in intent_categories]
        if not llm_intent:
            return allowed_intent_names[0]

        llm_intent_clean = llm_intent.strip()

        # 1. 精确匹配
        if llm_intent_clean in allowed_intent_names:
            print(f"[意图识别] 精确匹配: '{llm_intent_clean}'")
            return llm_intent_clean

        # 2. 清洗字符后的匹配
        def clean_text(text: str) -> str:
            text = text.replace(" ", "").replace("　", "")
            for punct in "：:，,。.！!？?；;、,/\\|":
                text = text.replace(punct, "")
            return text

        llm_intent_cleaned = clean_text(llm_intent_clean)
        for name in allowed_intent_names:
            if clean_text(name) == llm_intent_cleaned:
                print(f"[意图识别] 模糊匹配: '{llm_intent_clean}' -> '{name}'")
                return name

        # 3. 包含关系匹配
        for name in allowed_intent_names:
            if name in llm_intent_clean or llm_intent_clean in name:
                print(f"[意图识别] 包含匹配: '{llm_intent_clean}' -> '{name}'")
                return name

        # 4. 简单相似度（字符集合重叠度）
        def similarity(s1: str, s2: str) -> float:
            s1_clean = clean_text(s1)
            s2_clean = clean_text(s2)
            if not s1_clean or not s2_clean:
                return 0.0
            set1 = set(s1_clean)
            set2 = set(s2_clean)
            if not set1 or not set2:
                return 0.0
            return len(set1 & set2) / len(set1 | set2)

        best_match = None
        best_score = 0.0
        for name in allowed_intent_names:
            score = similarity(llm_intent_clean, name)
            if score > best_score:
                best_score = score
                best_match = name

        if best_match and best_score >= 0.4:
            print(
                f"[意图识别] 相似度匹配: '{llm_intent_clean}' -> '{best_match}' "
                f"(score={best_score:.2f})"
            )
            return best_match

        # 5. 全部失败时，返回第一个类别，仍然保留LLM原始输出用于日志
        print(
            f"[意图识别] 无法匹配 '{llm_intent_clean}'，"
            f"使用默认类别 '{allowed_intent_names[0]}'"
        )
        return allowed_intent_names[0]

    def generate_random_personal_facts(self) -> Dict:
        """生成随机个人事实"""
        ages = ["20-25", "26-30", "31-35", "36-40", "41-45", "46-50"]
        genders = ["男", "女", "其他"]
        employment_statuses = ["在校学生", "在职", "待业", "自由职业", "退休"]
        education_levels = ["高中", "大专", "本科", "硕士", "博士"]
        residence_areas = ["北京", "上海", "广州", "深圳", "杭州", "成都", "其他"]

        return {
            # 改为使用固定示例值，避免随机行为
            "age": ages[0],
            "gender": genders[0],
            "employment_status": employment_statuses[0],
            "education_level": education_levels[0],
            "residence_area": residence_areas[0],
        }

    def generate_random_personal_preferences(self) -> List[Dict]:
        """生成随机个人偏好"""
        # 默认偏好列表为空
        return []

    def generate_random_intent_history(self, count: int = 3) -> List[Dict]:
        """生成随机意图判断历史"""
        history: List[Dict] = []
        return history

    def retrieve_intent_from_history(
        self, context: str, user_id: Optional[str] = None
    ) -> Optional[Dict]:
        """从历史中检索相似情景的意图"""
        history = self.load_intent_history(user_id)
        if not history:
            return None

        # 简单的文本匹配
        context_lower = context.lower()
        for h in history:
            if (
                context_lower in h.get("context", "").lower()
                or h.get("context", "").lower() in context_lower
            ):
                return h

        return None

    def find_similar_intent_history_with_embedding(
        self,
        context: str,
        user_id: Optional[str] = None,
        recall_k: int = 20,
        top_k: Optional[int] = None,
    ) -> tuple[list[Dict], list[Dict]]:
        """
        使用情景匹配从意图历史中找出最相似的记录
        通过 BAAI/bge-large-zh 计算上下文语义相似度
        并按意图类别分组，每个类别最多返回3条记录，交由 LLM 综合判断

        Args:
            context: 用户输入的搜索场景
            user_id: 用户ID
            recall_k: 用于 ranked_items 的最大候选数量（默认20）
            top_k: 最终返回的数量；默认 None 时返回所有类别筛选后的记录

        Returns:
            Tuple:
                - List[Dict]: 筛选后的历史记录（每个类别最多3条），包含相似度分数
                - List[Dict]: 总体排序后的相似度榜单（用于可视化，最多 recall_k 条）
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("警告: sentence-transformers 未安装，无法使用 embedding 检索")
            return [], []

        history = self.load_intent_history(user_id)

        if not history or len(history) == 0:
            return [], []

        try:
            print(
                f"[情景匹配] 从 {len(history)} 条历史记录中检索，"
                "按意图类别筛选每类最多3条"
            )

            # 按意图类别分组
            history_by_intent = defaultdict(list)
            for record in history:
                intent_full = record.get("intent", "") or ""
                if "：" in intent_full:
                    intent_category = intent_full.split("：")[0].strip()
                elif ":" in intent_full:
                    intent_category = intent_full.split(":")[0].strip()
                else:
                    intent_category = intent_full.strip()
                if not intent_category:
                    intent_category = "未分类"
                history_by_intent[intent_category].append(record)

            print(f"[情景匹配] 分组后共有 {len(history_by_intent)} 个意图类别")

            retrieval_model = embedding_models.get_retrieval_model()
            query_embedding = retrieval_model.encode(context, convert_to_tensor=True)

            all_ranked_items: list[Dict] = []
            for intent_name, intent_history in history_by_intent.items():
                if not intent_history:
                    continue
                history_texts = [h.get("context", "") for h in intent_history]
                history_embeddings = retrieval_model.encode(
                    history_texts, convert_to_tensor=True
                )
                cos_scores = util.cos_sim(query_embedding, history_embeddings)[0]

                category_items: list[tuple[float, Dict]] = []
                for record, score in zip(intent_history, cos_scores):
                    item = record.copy()
                    normalized_score = (float(score) + 1.0) / 2.0
                    normalized_score = max(0.0, min(1.0, normalized_score))
                    item["similarity_score"] = normalized_score
                    item["intent_category"] = intent_name
                    category_items.append((normalized_score, item))

                category_items.sort(key=lambda x: x[0], reverse=True)
                max_per_category = min(3, len(category_items))
                selected = category_items[:max_per_category]
                print(
                    f"[情景匹配] 类别 '{intent_name}' "
                    f"共有 {len(category_items)} 条，选取 {len(selected)} 条"
                )
                for _, item in selected:
                    all_ranked_items.append(item)

            all_ranked_items.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)

            if top_k is None:
                top_matches = all_ranked_items
            else:
                top_matches = all_ranked_items[: min(top_k, len(all_ranked_items))]

            recall_k_actual = min(recall_k, len(all_ranked_items))
            ranked_items = all_ranked_items[:recall_k_actual]

            print(
                f"[情景匹配] 最终返回 {len(top_matches)} 条记录 "
                f"(来自 {len(history_by_intent)} 个类别)"
            )
            return top_matches, ranked_items

        except Exception as e:
            print(f"Embedding 检索失败: {str(e)}")
            import traceback

            traceback.print_exc()
            return [], []

    def retrieve_personal_memory(
        self,
        context: str,
        intent: str,
        personal_facts: Dict,
        personal_preferences: List[Dict],
    ) -> Dict:
        """
        使用Proxy模型根据搜索场景和意图从总体个人记忆中归纳出相关的个人记忆

        Args:
            context: 用户输入的搜索场景
            intent: 识别出的用户意图
            personal_facts: 用户的个人事实（长期）
            personal_preferences: 用户的个人偏好（短期）

        Returns:
            {"memory": 相关记忆文本, "thinking": 推理过程}
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

        # 处理新格式的个人偏好（短期）- 只包含选中的偏好，区分喜欢和不喜欢
        all_preferences = []
        if isinstance(personal_preferences, list):
            for pref in personal_preferences:
                if pref.get("selected", False):
                    pref_text = pref.get("text", "")
                    pref_type = pref.get("preference_type", "like")
                    # 明确标注是"喜欢"还是"不喜欢"
                    if pref_type == "dislike":
                        all_preferences.append(f"不喜欢：{pref_text}")
                    else:
                        all_preferences.append(f"喜欢：{pref_text}")

        if not all_facts and not all_preferences:
            return {"memory": "暂无个人记忆", "thinking": ""}

        # 构建完整记忆文本
        all_memory_text = ""
        if all_facts:
            all_memory_text += (
                "个人事实（长期）：\n" + "\n".join(f"- {f}" for f in all_facts) + "\n\n"
            )
        if all_preferences:
            all_memory_text += "个人偏好（短期）：\n" + "\n".join(
                f"- {p}" for p in all_preferences
            )

        # 使用Proxy模型归纳相关记忆
        memory_prompt = f"""基于用户的搜索场景和识别出的意图，从总体个人记忆中归纳出与当前场景最相关的个人记忆。

【用户搜索场景】
{context}

【识别的用户意图】
{intent}

【总体个人记忆】
{all_memory_text}

【任务要求】
请综合考虑用户的搜索场景和意图，分析并归纳出与当前场景最相关的个人记忆。
1. 优先考虑能够影响搜索结果个性化的记忆（如与场景相关的偏好、背景等）
2. 只保留真正相关的信息，与场景无关的记忆应该忽略
3. 如果个人事实（如年龄、性别、职业等）对理解用户需求有帮助，也应该保留
4. 如果个人偏好能够指导搜索结果的呈现方式或内容侧重，应该重点关注

【输出格式】
相关记忆：
[归纳后的相关记忆，简洁明了地列出与搜索场景和意图相关的记忆]

相关性说明：
[简要说明这些记忆为什么与当前搜索场景和意图相关，它们如何影响用户的搜索需求]"""

        result = self.call_deepseek_api(
            memory_prompt,
            "你是一个个人记忆分析专家，擅长根据用户的搜索场景和意图，从总体记忆中提取最相关的个性化信息，帮助理解用户的真实需求。",
            enable_thinking=True,
        )

        # 解析相关记忆
        import re

        memory_match = re.search(
            r"相关记忆[：:]\s*(.+?)(?=相关性说明|$)", result["content"], re.DOTALL
        )
        if memory_match:
            relevant_memory = memory_match.group(1).strip()
        else:
            # 如果没有匹配到，使用全部记忆
            relevant_memory = all_memory_text

        return {"memory": relevant_memory, "thinking": result["thinking"]}

    def generate_personal_description(
        self, context: str, intent: str, intent_explanation: str, personal_memory: str
    ) -> str:
        """
        生成结构化的个性化描述，用于指导后续的搜索和回答生成

        Args:
            context: 用户输入的搜索场景
            intent: 识别出的用户意图
            intent_explanation: 意图的详细解释
            personal_memory: 相关的个人记忆

        Returns:
            结构化的个性化描述文本
        """
        description = f"""# 用户个性化搜索需求分析

## 搜索场景
{context}

## 用户意图
**意图类别**: {intent}

**意图分析**: {intent_explanation}

## 相关个人背景
{personal_memory}

## 个性化指导
基于以上信息，在回答用户查询时，应该：
1. 考虑用户的个人背景和偏好，调整内容的深度和呈现方式
2. 根据用户意图，突出最相关和最有价值的信息
3. 采用符合用户偏好的表达风格和信息组织方式
4. 提供与用户背景和需求相匹配的建议和补充信息
"""
        return description

    def call_local_llm(
        self, prompt: str, system_prompt: str = None, max_new_tokens: int = 2000
    ) -> Dict:
        """调用本地 LLM 模型进行推理"""
        if not TRANSFORMERS_AVAILABLE:
            return {
                "content": "transformers 未安装，无法使用本地 LLM 模型",
                "thinking": "",
            }

        try:
            model, tokenizer = local_llm_models.get_local_llm()

            # 构建消息
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # 应用 chat template
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Tokenize
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            # 生成
            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.8,
                    repetition_penalty=1.05
                )

            # 解码
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            return {"content": response.strip(), "thinking": ""}
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"content": f"本地模型调用失败：{str(e)}", "thinking": ""}

    def call_deepseek_api(
        self, prompt: str, system_prompt: str = None, enable_thinking: bool = True
    ) -> Dict:
        """调用 DeepSeek API，支持 thinking 过程"""
        if not self.client:
            return {
                "content": "DeepSeek API 未配置，请设置 DASHSCOPE_API_KEY 环境变量",
                "thinking": "",
            }

        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            extra_body = {}
            if enable_thinking:
                extra_body["extend_fields"] = {
                    "chat_template_kwargs": {"enable_thinking": True}
                }

            response = self.client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=4000,
                extra_body=extra_body if extra_body else None,
            )

            message = response.choices[0].message
            content = message.content.strip() if message.content else ""

            # 提取 thinking 内容（deepseek-v3的thinking可能在多个位置）
            thinking = ""
            # 方法1：检查message对象是否有thinking属性
            if hasattr(message, "thinking") and message.thinking:
                thinking = message.thinking.strip()
            # 方法2：检查是否有reasoning_content
            elif hasattr(message, "reasoning_content") and message.reasoning_content:
                thinking = message.reasoning_content.strip()
            # 方法3：检查response对象本身
            elif hasattr(response, "thinking") and response.thinking:
                thinking = response.thinking.strip()
            # 方法4：检查choices[0]是否有thinking
            elif (
                hasattr(response.choices[0], "thinking")
                and response.choices[0].thinking
            ):
                thinking = response.choices[0].thinking.strip()
            # 方法5：尝试从content中提取（如果格式是<think>...</think>）
            if not thinking and "<think>" in content.lower():
                import re

                think_match = re.search(
                    r"<think>(.*?)</think>", content, re.DOTALL | re.IGNORECASE
                )
                if think_match:
                    thinking = think_match.group(1).strip()
                    # 从content中移除thinking部分
                    content = re.sub(
                        r"<think>.*?</think>",
                        "",
                        content,
                        flags=re.DOTALL | re.IGNORECASE,
                    ).strip()

            return {"content": content, "thinking": thinking}
        except Exception as e:
            return {"content": f"API 调用失败：{str(e)}", "thinking": ""}

    def call_multimodal_api(
        self, content: List[Dict], system_prompt: str = None, stream: bool = False
    ) -> Dict:
        """调用多模态API（qwen3-omni-flash），支持视频、图片等文件"""
        if not self.client:
            return {
                "content": "多模态 API 未配置，请设置 DASHSCOPE_API_KEY 环境变量",
                "thinking": "",
            }

        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": content})

            response = self.client.chat.completions.create(
                model=MULTIMODAL_MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=4000,
                stream=stream,
            )

            if stream:
                # 流式响应处理
                full_content = ""
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        full_content += chunk.choices[0].delta.content
                return {"content": full_content.strip(), "thinking": ""}
            else:
                message = response.choices[0].message
                content_text = message.content.strip() if message.content else ""
                return {"content": content_text, "thinking": ""}
        except Exception as e:
            return {"content": f"多模态 API 调用失败：{str(e)}", "thinking": ""}


# 初始化 PersonalProxyWeb
proxy = PersonalProxyWeb()


def get_active_user_id(allow_default: bool = False) -> Optional[str]:
    """获取当前会话中的用户 ID，必要时回退到默认用户"""
    user_id = session.get("user_id")
    if user_id:
        return user_id
    if allow_default:
        return DEFAULT_USER_ID
    return None


def load_user_accounts() -> Dict[str, Dict]:
    """从磁盘加载账号信息"""
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and "users" in data:
                return data["users"]
            if isinstance(data, dict):
                return data
    except json.JSONDecodeError:
        return {}
    return {}


def save_user_accounts(users: Dict[str, Dict]):
    """保存账号信息"""
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump({"users": users}, f, ensure_ascii=False, indent=2)


def ensure_user_storage(user_id: str):
    """初始化用户数据目录和默认文件"""
    proxy._resolve_user_dir(user_id)

    facts_file = proxy._user_file("personal_facts.json", user_id)
    if not os.path.exists(facts_file):
        proxy.save_personal_facts(
            {
                "age": "",
                "gender": "",
                "employment_status": "",
                "education_level": "",
                "residence_area": "",
            },
            user_id,
        )

    prefs_file = proxy._user_file("personal_preferences.json", user_id)
    if not os.path.exists(prefs_file):
        proxy.save_personal_preferences([], user_id)

    history_file = proxy._user_file("intent_history.json", user_id)
    if not os.path.exists(history_file):
        proxy.save_intent_history([], user_id)

    # 确保意图类别文件存在且包含8个默认类别
    # load_intent_categories 会自动处理：如果文件不存在或类别数量不足8个，会使用默认的8个类别
    proxy.load_intent_categories(user_id)


@app.context_processor
def inject_current_user():
    """模板中注入当前登录用户"""
    return {"current_user": session.get("user_id")}


def login_required_view(func):
    """页面视图登录校验"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login", next=request.path))
        return func(*args, **kwargs)

    return wrapper


def login_required_api(func):
    """API 登录校验"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"success": False, "message": "请先登录"}), 401
        return func(*args, **kwargs)

    return wrapper


@app.route("/login", methods=["GET", "POST"])
def login():
    """登录页面"""
    if session.get("user_id"):
        return redirect(url_for("index"))

    login_error = None
    raw_next = request.args.get("next")
    next_url = raw_next if raw_next and raw_next.startswith("/") else url_for("index")
    username_value = ""

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        username_value = username
        password = request.form.get("password", "").strip()

        if not username or not password:
            login_error = "请输入账号和密码"
        else:
            users = load_user_accounts()
            user = users.get(username)
            if not user or not check_password_hash(
                user.get("password_hash", ""), password
            ):
                login_error = "账号或密码错误"
            else:
                session["user_id"] = username
                ensure_user_storage(username)
                return redirect(next_url or url_for("index"))

    return render_template(
        "login.html",
        login_error=login_error,
        next_url=next_url,
        login_username=username_value,
    )


@app.route("/register", methods=["GET", "POST"])
def register():
    """注册账号"""
    if session.get("user_id"):
        return redirect(url_for("index"))

    register_error = None
    username_value = ""

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        username_value = username
        password = request.form.get("password", "").strip()
        confirm_password = request.form.get("confirm_password", "").strip()

        if not username or not password or not confirm_password:
            register_error = "请完整填写注册信息"
        elif not USERNAME_PATTERN.match(username):
            register_error = "账号需为3-32位字母、数字、下划线或短横线"
        elif password != confirm_password:
            register_error = "两次输入的密码不一致"
        else:
            users = load_user_accounts()
            if username in users:
                register_error = "该账号已存在"
            else:
                users[username] = {
                    "password_hash": generate_password_hash(password),
                    "created_at": datetime.now().isoformat(),
                }
                save_user_accounts(users)
                ensure_user_storage(username)
                session["user_id"] = username
                return redirect(url_for("index"))

    return render_template(
        "register.html",
        register_error=register_error,
        register_username=username_value,
    )


@app.route("/logout")
def logout():
    """退出登录"""
    session.pop("user_id", None)
    return redirect(url_for("login"))


@app.route("/")
@login_required_view
def index():
    """主页（个人资料管理）"""
    return render_template("index.html")


@app.route("/api/personal-facts", methods=["GET"])
@login_required_api
def get_personal_facts():
    """获取个人事实"""
    user_id = get_active_user_id()
    facts = proxy.load_personal_facts(user_id)
    return jsonify({"success": True, "data": facts})


@app.route("/api/personal-facts", methods=["POST"])
@login_required_api
def update_personal_facts():
    """更新个人事实"""
    data = request.json
    user_id = get_active_user_id()
    proxy.save_personal_facts(data, user_id)
    return jsonify({"success": True, "message": "个人事实已更新"})


@app.route("/api/personal-preferences", methods=["GET"])
@login_required_api
def get_personal_preferences():
    """获取个人偏好"""
    user_id = get_active_user_id()
    preferences = proxy.load_personal_preferences(user_id)
    return jsonify({"success": True, "data": preferences})


@app.route("/api/personal-preferences", methods=["POST"])
@login_required_api
def update_personal_preferences():
    """更新个人偏好（短期）"""
    data = request.json
    user_id = get_active_user_id()
    if not isinstance(data, list):
        return jsonify({"success": False, "message": "数据格式错误"}), 400
    proxy.save_personal_preferences(data, user_id)
    return jsonify({"success": True, "message": "个人偏好已更新"})


@app.route("/api/intent-history", methods=["GET"])
@login_required_api
def get_intent_history():
    """获取意图判断历史"""
    user_id = get_active_user_id()
    history = proxy.load_intent_history(user_id)
    return jsonify({"success": True, "data": history})


@app.route("/api/intent-history", methods=["POST"])
@login_required_api
def add_intent_history():
    """添加意图判断历史"""
    data = request.json
    user_id = get_active_user_id()
    history = proxy.load_intent_history(user_id)

    new_item = {
        "id": f"intent_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "context": data.get("context", ""),
        "user": data.get("user", "当前用户"),
        "intent": data.get("intent", ""),
        "intent_explanation": data.get("intent_explanation", ""),
        "user_feedback": data.get("user_feedback", ""),
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }

    history.append(new_item)
    proxy.save_intent_history(history, user_id)
    return jsonify({"success": True, "data": new_item})


@app.route("/api/intent-history/<path:intent_id>", methods=["PUT"])
@login_required_api
def update_intent_history(intent_id):
    """更新意图判断历史"""
    data = request.json
    user_id = get_active_user_id()
    history = proxy.load_intent_history(user_id)

    for i, item in enumerate(history):
        if str(item.get("id")) == str(intent_id):
            history[i].update(
                {
                    "context": data.get("context", item.get("context")),
                    "user": data.get("user", item.get("user")),
                    "intent": data.get("intent", item.get("intent")),
                    "intent_explanation": data.get(
                        "intent_explanation", item.get("intent_explanation")
                    ),
                    "user_feedback": data.get(
                        "user_feedback", item.get("user_feedback")
                    ),
                    "updated_at": datetime.now().isoformat(),
                }
            )
            proxy.save_intent_history(history, user_id)
            return jsonify({"success": True, "data": history[i]})

    return jsonify({"success": False, "message": "未找到对应的意图历史"}), 404


@app.route("/api/intent-history/<path:intent_id>", methods=["DELETE"])
@login_required_api
def delete_intent_history(intent_id):
    """删除意图判断历史"""
    user_id = get_active_user_id()
    history = proxy.load_intent_history(user_id)
    history = [h for h in history if str(h.get("id")) != str(intent_id)]
    proxy.save_intent_history(history, user_id)
    return jsonify({"success": True, "message": "已删除"})


@app.route("/api/intent-categories", methods=["GET"])
@login_required_api
def get_intent_categories():
    user_id = get_active_user_id()
    categories = proxy.load_intent_categories(user_id)
    return jsonify({"success": True, "data": categories})


@app.route("/api/intent-categories", methods=["POST"])
@login_required_api
def add_intent_category():
    data = request.json
    name = (data.get("name") or "").strip()
    generic_explanation = (data.get("generic_explanation") or "").strip()
    if not name or not generic_explanation:
        return (
            jsonify({"success": False, "message": "意图名称和通用解释不能为空"}),
            400,
        )
    user_id = get_active_user_id()
    categories = proxy.load_intent_categories(user_id)
    for c in categories:
        if c.get("name") == name:
            return jsonify({"success": False, "message": "该意图已存在"}), 400
    new_id = f"intent_type_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    max_order = 0
    for c in categories:
        order = c.get("order") or 0
        if order > max_order:
            max_order = order
    new_category = {
        "id": new_id,
        "name": name,
        "generic_explanation": generic_explanation,
        "order": max_order + 1,
    }
    categories.append(new_category)
    proxy.save_intent_categories(categories, user_id)
    return jsonify({"success": True, "data": new_category})


@app.route("/api/preferences/upload-file", methods=["POST"])
@login_required_api
def upload_preference_file():
    """上传个人信息文件（支持视频、图片、文本等），使用多模态大模型提取个人偏好"""
    print(f"[DEBUG] 文件上传请求 - 用户ID: {session.get('user_id')}")
    print(f"[DEBUG] 请求文件: {request.files}")

    user_id = get_active_user_id()

    if "file" not in request.files:
        return jsonify({"success": False, "message": "未找到上传文件"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "message": "未选择文件"}), 400

    # 验证文件类型 - 支持更多格式
    allowed_extensions = {
        # 文本文件
        "txt",
        "md",
        "doc",
        "docx",
        # 图片文件
        "jpg",
        "jpeg",
        "png",
        "gif",
        "bmp",
        "webp",
        # 视频文件
        "mp4",
        "avi",
        "mov",
        "wmv",
        "flv",
        "mkv",
        "webm",
        # PDF文件
        "pdf",
    }
    file_ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if file_ext not in allowed_extensions:
        return jsonify(
            {
                "success": False,
                "message": "不支持的文件格式。支持格式：文本(.txt, .md, .doc, .docx)、图片(.jpg, .png, .gif等)、视频(.mp4, .avi, .mov等)、PDF(.pdf)",
            }
        ), 400

    # 获取文件类别
    category = request.form.get("category", "未知来源")

    try:
        # 判断文件类型并处理
        content_items = []

        if file_ext in {"txt", "md"}:
            # 文本文件：直接读取
            file_content = file.read().decode("utf-8")
            content_items.append({"type": "text", "text": file_content})
        elif file_ext in {"jpg", "jpeg", "png", "gif", "bmp", "webp"}:
            # 图片文件：转换为base64
            file_data = file.read()
            base64_data = base64.b64encode(file_data).decode("utf-8")
            # 根据文件扩展名确定MIME类型
            mime_types = {
                "jpg": "image/jpeg",
                "jpeg": "image/jpeg",
                "png": "image/png",
                "gif": "image/gif",
                "bmp": "image/bmp",
                "webp": "image/webp",
            }
            mime_type = mime_types.get(file_ext, "image/jpeg")
            content_items.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{base64_data}"},
                }
            )
        elif file_ext in {"mp4", "avi", "mov", "wmv", "flv", "mkv", "webm"}:
            # 视频文件：转换为base64
            file_data = file.read()
            base64_data = base64.b64encode(file_data).decode("utf-8")
            # 根据文件扩展名确定MIME类型
            mime_types = {
                "mp4": "video/mp4",
                "avi": "video/x-msvideo",
                "mov": "video/quicktime",
                "wmv": "video/x-ms-wmv",
                "flv": "video/x-flv",
                "mkv": "video/x-matroska",
                "webm": "video/webm",
            }
            mime_type = mime_types.get(file_ext, "video/mp4")
            content_items.append(
                {
                    "type": "video_url",
                    "video_url": {"url": f"data:{mime_type};base64,{base64_data}"},
                }
            )
        elif file_ext == "pdf":
            # PDF文件：直接转换成 txt 并按文本处理
            file_data = file.read()
            extracted_text = proxy.extract_pdf_text(file_data)
            if not extracted_text:
                print("[PDF解析] 未从该 PDF 中提取到可用文本，返回失败提示")
                return (
                    jsonify(
                        {
                            "success": False,
                            "message": "无法从该 PDF 中提取文字，请先转为可编辑的 txt/md 再上传",
                        }
                    ),
                    400,
                )

            txt_path = proxy.save_pdf_text_file(extracted_text, file.filename, user_id)
            if txt_path:
                print(f"[PDF解析] 已转换为 txt 文件: {txt_path}")
            content_items.append({"type": "text", "text": extracted_text})
            print(
                f"[PDF解析] 抽取成功，字符数 {len(extracted_text)}，按 txt 文件处理完成"
            )
        else:
            # 其他文件类型（doc, docx等）暂时不支持，提示用户
            return jsonify(
                {
                    "success": False,
                    "message": f"暂不支持 .{file_ext} 格式，请转换为文本、图片或视频格式",
                }
            ), 400

        # 添加提示词文本
        extract_prompt_text = f"""这是用户上传的【{category}】。请仔细分析这些内容，总结出用户的个人偏好（短期偏好）。

个人偏好应该是关于用户的行为习惯、阅读偏好、学习方式、创作风格、兴趣倾向等短期可变的特征。

请提取出所有可能的个人偏好，每个偏好用一句话简洁描述。注意：
1. 只提取偏好类信息，不要提取事实类信息（如年龄、性别、职业等）
2. 每个偏好应该独立且明确
3. 使用第三人称描述，例如"喜欢阅读长文本"而不是"我喜欢阅读长文本"
4. 如果内容中没有明确的偏好信息，请返回"无明确偏好"
5. 对于视频和图片，关注用户的观看习惯、创作风格、内容偏好等
6. 对于文本，关注用户的阅读习惯、写作风格、话题偏好等
7. 结合内容的来源类别【{category}】进行分析。例如：
   - 如果是"个人原创内容"，重点关注用户的创作风格、擅长领域、表达习惯。
   - 如果是"已阅读内容"，重点关注用户的阅读兴趣、关注话题、信息获取偏好。
   - 如果是"个人整理内容"，重点关注用户的知识组织方式、归纳总结习惯。
   - 如果是"他人创作内容"，重点关注用户对他人的关注点、审美偏好。
   - 如果是"人工智能生成内容"，重点关注用户对AI工具的使用偏好、指令风格。

请按以下格式输出，每行一个偏好：
偏好1: [偏好描述]
偏好2: [偏好描述]
...

如果没有找到偏好，只输出：
无明确偏好"""

        content_items.append({"type": "text", "text": extract_prompt_text})

        # 使用多模态大模型提取个人偏好
        system_prompt = "你是一个擅长分析用户内容并提取个人偏好的专家。你需要从用户看过的或创造的内容中，识别出能够反映用户行为习惯、阅读偏好、学习方式等短期可变的特征。"

        result = proxy.call_multimodal_api(
            content=content_items, system_prompt=system_prompt, stream=False
        )

        content = result["content"].strip()

        # 解析提取的偏好
        preferences = []
        if "无明确偏好" not in content:
            lines = content.split("\n")
            for line in lines:
                line = line.strip()
                if line and (
                    "偏好" in line or line.startswith("-") or line.startswith("•")
                ):
                    # 提取偏好文本
                    if ":" in line or "：" in line:
                        pref_text = line.split(":", 1)[-1].split("：", 1)[-1].strip()
                    else:
                        pref_text = line.lstrip("-•").strip()

                    if pref_text and len(pref_text) > 2:
                        preferences.append(
                            {
                                "id": f"pref_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
                                "text": pref_text,
                                "selected": False,  # 默认不选中，由用户自己选择
                                "preference_type": "like",  # 默认为"like"，用户可以在前端选择
                            }
                        )

        return jsonify(
            {
                "success": True,
                "data": preferences,
                "message": f"成功提取 {len(preferences)} 个个人偏好",
            }
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"success": False, "message": f"文件处理失败：{str(e)}"}), 500


@app.route("/api/preferences/add", methods=["POST"])
@login_required_api
def add_preference():
    """添加新的个人偏好"""
    data = request.json
    preference_text = data.get("text", "").strip()

    if not preference_text:
        return jsonify({"success": False, "message": "偏好内容不能为空"}), 400

    user_id = get_active_user_id()
    preferences = proxy.load_personal_preferences(user_id)

    # 检查是否已存在相同偏好
    for pref in preferences:
        if pref.get("text") == preference_text:
            return jsonify({"success": False, "message": "该偏好已存在"}), 400

    # 添加新偏好
    preference_type = data.get(
        "preference_type", "like"
    )  # 默认为"like"（喜欢），也可以是"dislike"（不喜欢）
    new_preference = {
        "id": f"pref_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
        "text": preference_text,
        "selected": data.get("selected", True),  # 默认选中
        "preference_type": preference_type,  # "like" 或 "dislike"
    }
    preferences.append(new_preference)
    proxy.save_personal_preferences(preferences, user_id)

    return jsonify({"success": True, "data": new_preference, "message": "偏好已添加"})


@app.route("/api/preferences/<preference_id>", methods=["DELETE"])
@login_required_api
def delete_preference(preference_id):
    """删除个人偏好"""
    user_id = get_active_user_id()
    preferences = proxy.load_personal_preferences(user_id)

    # 过滤掉要删除的偏好
    new_preferences = [p for p in preferences if p.get("id") != preference_id]

    if len(new_preferences) == len(preferences):
        return jsonify({"success": False, "message": "未找到该偏好"}), 404

    proxy.save_personal_preferences(new_preferences, user_id)
    return jsonify({"success": True, "message": "偏好已删除"})


@app.route("/api/preferences/<preference_id>/toggle", methods=["PUT"])
@login_required_api
def toggle_preference(preference_id):
    """切换个人偏好的选中状态"""
    user_id = get_active_user_id()
    preferences = proxy.load_personal_preferences(user_id)

    found = False
    for pref in preferences:
        if pref.get("id") == preference_id:
            pref["selected"] = not pref.get("selected", False)
            found = True
            break

    if not found:
        return jsonify({"success": False, "message": "未找到该偏好"}), 404

    proxy.save_personal_preferences(preferences, user_id)
    return jsonify({"success": True, "data": preferences, "message": "偏好状态已更新"})


@app.route("/api/preferences/<preference_id>/toggle-type", methods=["PUT"])
@login_required_api
def toggle_preference_type(preference_id):
    """切换个人偏好的类型（喜欢/不喜欢）"""
    user_id = get_active_user_id()
    preferences = proxy.load_personal_preferences(user_id)

    found = False
    for pref in preferences:
        if pref.get("id") == preference_id:
            current_type = pref.get("preference_type", "like")
            # 切换类型
            pref["preference_type"] = "dislike" if current_type == "like" else "like"
            found = True
            break

    if not found:
        return jsonify({"success": False, "message": "未找到该偏好"}), 404

    proxy.save_personal_preferences(preferences, user_id)
    return jsonify({"success": True, "data": preferences, "message": "偏好类型已更新"})


@app.route("/api/generate-description", methods=["POST"])
@login_required_api
def generate_description():
    """生成个性化描述（情景匹配检索）"""
    data = request.json
    context = data.get("context", "")
    user_id = get_active_user_id()

    if not context:
        return jsonify({"success": False, "message": "输入场景不能为空"}), 400

    # ===== 1. 使用情景 embedding 检索相似历史（按类别最多3条）=====
    try:
        similar_histories, similarity_rankings = (
            proxy.find_similar_intent_history_with_embedding(
                context=context,
                user_id=user_id,
                recall_k=20,
                top_k=None,
            )
        )
        print(
            f"找到 {len(similar_histories)} 条相似历史记录 "
            "(按意图类别筛选，每类最多3条)"
        )
    except Exception as e:
        print(f"检索相似历史失败: {str(e)}")
        similar_histories = []
        similarity_rankings = []

    # ===== 2. 加载允许的意图类别 =====
    intent_categories = proxy.load_intent_categories(user_id)

    # 格式化为文本
    categories_text = "\n".join(
        [
            f"- {cat['name']}: {cat.get('generic_explanation', '')}"
            for cat in intent_categories
        ]
    )

    # 格式化相似历史
    similar_histories_text = ""
    if similar_histories:
        for idx, h in enumerate(similar_histories, 1):
            similarity_pct = h.get("similarity_score", 0) * 100
            similar_histories_text += f"""
历史记录 {idx}:
  - 情景: {h.get("context", "")}
  - 意图: {h.get("intent", "")}
  - 意图解释: {h.get("intent_explanation", "")}
  - 相似度: {similarity_pct:.1f}%
"""
    else:
        similar_histories_text = "暂无相似历史记录"

    # 格式化候选列表（用于给大模型的概率可视化）
    candidate_histories_text = ""
    if similarity_rankings:
        for idx, h in enumerate(similarity_rankings[:5], 1):
            similarity_pct = h.get("similarity_score", 0) * 100
            candidate_histories_text += f"""
候选 {idx}:
  - 情景: {h.get("context", "")}
  - 意图: {h.get("intent", "")}
  - 意图解释: {h.get("intent_explanation", "")}
  - 相似度概率: {similarity_pct:.1f}%
"""
    else:
        candidate_histories_text = "暂无候选"

    # ===== 3. 构建 prompt，生成意图和意图解释（使用本地 LLM 模型）=====
    intent_prompt = f"""请基于用户输入的搜索场景，结合相似历史记录，准确判断用户的搜索意图并给出相应的解释。

【用户输入的搜索场景】
{context}

【相似的历史记录（共 {len(similar_histories)} 条，每类最多3条）】
{similar_histories_text}

【历史匹配候选及概率（按相似度从高到低列出前5条）】
{candidate_histories_text}

【允许的意图类别】
以下是用户预定义的意图类别及其通用解释。你生成的意图必须从中选择一个（不能创造新类别）：
{categories_text}

【任务要求】
1. 分析用户的搜索场景，理解其背后的真实需求和动机
2. 综合"历史匹配候选及概率"，重点关注高概率候选的情景与意图解释，并说明它们如何影响你的判断
3. 从"允许的意图类别"中选择一个最匹配的类别
4. 给出精炼的意图解释，简要说明用户的核心需求和期望获得的信息类型；如果引用了某个候选，请在解释中指出

【输出格式】
意图类别: [从允许列表中选择的类别名称]
意图解释: [用一段话精炼说明用户的核心意图和需求]
"""

    # 使用本地 LLM 模型进行意图判断
    result = proxy.call_local_llm(
        intent_prompt,
        "你是一个专业的用户意图分析专家，擅长从搜索场景中洞察用户的真实需求，结合历史行为模式提供准确的意图判断和解释。",
        max_new_tokens=2000,
    )

    # ===== 4. 解析结果 =====
    intent_match = re.search(r"意图类别[：:]\s*(.+)", result["content"])
    explanation_match = re.search(r"意图解释[：:]\s*(.+)", result["content"], re.DOTALL)

    llm_intent_raw = intent_match.group(1).strip() if intent_match else "未识别"
    intent_explanation = (
        explanation_match.group(1).strip() if explanation_match else "未能生成解释"
    )

    print(f"[意图识别] LLM 判定原始意图: '{llm_intent_raw}'")
    intent = proxy.identify_intent_category(llm_intent_raw, intent_categories)
    print(f"[意图识别] 最终使用的意图类别: '{intent}'")

    # ===== 5. 加载个人记忆 =====
    personal_facts = proxy.load_personal_facts(user_id)
    personal_preferences = proxy.load_personal_preferences(user_id)

    # 检索相关个人记忆（使用context和intent）
    memory_result = proxy.retrieve_personal_memory(
        context, intent, personal_facts, personal_preferences
    )
    personal_memory = memory_result["memory"]

    # ===== 6. 生成个性化描述 =====
    personal_description = proxy.generate_personal_description(
        context, intent, intent_explanation, personal_memory
    )

    # ===== 7. 返回结果 =====
    return jsonify(
        {
            "success": True,
            "data": {
                "context": context,
                "similar_histories": similar_histories,  # top 匹配
                "similarity_rankings": similarity_rankings,  # 全量可视化
                "intent": intent,
                "intent_explanation": intent_explanation,
                "personal_memory": personal_memory,
                "personal_description": personal_description,
                "thinking": result.get("thinking", ""),
            },
        }
    )


@app.route("/api/generate-description", methods=["POST"])
@login_required_api
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
    user_id = get_active_user_id()
    personal_facts = proxy.load_personal_facts(user_id)
    personal_preferences = proxy.load_personal_preferences(user_id)
    intent_history = proxy.load_intent_history(user_id)

    # 提取意图类别列表
    intent_categories = []
    for h in intent_history:
        intent_cat = (
            h.get("intent", "").split("：")[0]
            if "：" in h.get("intent", "")
            else h.get("intent", "")
        )
        if intent_cat and intent_cat not in intent_categories:
            intent_categories.append(intent_cat)

    intent_categories_text = (
        "\n".join([f"- {cat}" for cat in intent_categories])
        if intent_categories
        else "暂无历史意图类别"
    )

    # ========== 第一轮：推理意图类别并进行历史检索（使用本地 LLM 模型）==========
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

    # 使用本地 LLM 模型进行第一轮推理
    round1_result = proxy.call_local_llm(
        round1_prompt,
        "你是一个意图理解专家，擅长分析用户的搜索意图并进行历史检索。",
        max_new_tokens=1500,
    )
    round1_thinking = round1_result["thinking"]
    round1_content = round1_result["content"]

    # 解析第一轮结果
    import re

    category_match = re.search(r"意图类别[：:]\s*(.+)", round1_content)
    similar_category_match = re.search(r"相似历史类别[：:]\s*(.+)", round1_content)
    similar_history_match = re.search(
        r"相似历史记录[：:]\s*(.+)", round1_content, re.DOTALL
    )

    intent_category = category_match.group(1).strip() if category_match else "信息查询"
    similar_category = (
        similar_category_match.group(1).strip() if similar_category_match else "无"
    )
    similar_history = (
        similar_history_match.group(1).strip() if similar_history_match else "无"
    )

    # ========== 第二轮：给出意图类别判断和意图解释（使用本地 LLM 模型）==========
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

    # 使用本地 LLM 模型进行第二轮推理
    round2_result = proxy.call_local_llm(
        round2_prompt,
        "你是一个意图分析专家，能够基于历史信息给出准确的意图判断和解释。",
        max_new_tokens=1500,
    )
    round2_thinking = round2_result["thinking"]
    round2_content = round2_result["content"]

    # 解析第二轮结果
    final_category_match = re.search(r"最终意图类别[：:]\s*(.+)", round2_content)
    explanation_match = re.search(r"意图解释[：:]\s*(.+)", round2_content, re.DOTALL)

    final_intent_category = (
        final_category_match.group(1).strip()
        if final_category_match
        else intent_category
    )
    intent_explanation = (
        explanation_match.group(1).strip()
        if explanation_match
        else "用户希望通过搜索获得相关信息"
    )

    intent = f"{final_intent_category}：{query}"

    # 检索相关个人记忆（使用query作为context和intent）
    memory_result = proxy.retrieve_personal_memory(
        query, intent, personal_facts, personal_preferences
    )
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
        "updated_at": datetime.now().isoformat(),
    }
    intent_history.append(new_history)
    proxy.save_intent_history(intent_history, user_id)

    return jsonify(
        {
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
                "similar_history": similar_history,
            },
        }
    )


@app.route("/api/search-with-description", methods=["POST"])
@login_required_api
def search_with_description():
    """
    基于个性化描述进行搜索，调用大模型生成内容
    """
    data = request.json
    context = data.get("context", "")
    intent = data.get("intent", "")
    intent_explanation = data.get("intent_explanation", "")
    personal_memory = data.get("personal_memory", "")
    personal_description = data.get("personal_description", "")

    if not personal_description:
        return jsonify({"success": False, "message": "个性化描述不能为空"}), 400

    # 构建搜索prompt
    search_prompt = f"""你是一个专业的知识内容生成助手。用户提出了一个搜索需求，我们已经对用户的意图和个人背景进行了分析。请根据这些信息，生成一篇符合用户需求的专业内容。

【用户搜索场景】
{context}

【用户意图分析】
意图类别：{intent}
意图解释：{intent_explanation}

【用户个人背景】
{personal_memory}

【个性化指导】
{personal_description}

【任务要求】
1. 根据用户的搜索场景和意图，生成相关的专业内容
2. 内容应该符合用户意图类别的特点（如定义-论证式、故事-叙事式等）
3. 考虑用户的个人背景和偏好，调整内容的深度、风格和呈现方式
4. 内容要专业、准确、有深度，同时保持易读性
5. 适当使用结构化的组织方式（如标题、列表、段落等）

请直接输出内容，不需要额外的说明或前缀。"""

    try:
        result = proxy.call_deepseek_api(
            search_prompt,
            "你是一个专业的知识内容生成助手，擅长根据用户的个性化需求生成高质量、有针对性的内容。",
            enable_thinking=True,
        )

        return jsonify(
            {
                "success": True,
                "data": {
                    "content": result["content"],
                    "thinking": result["thinking"],
                },
            }
        )
    except Exception as e:
        return jsonify({"success": False, "message": f"生成失败：{str(e)}"}), 500


if __name__ == "__main__":
    # 初始化一些随机数据（如果数据文件不存在）
    default_user = DEFAULT_USER_ID
    default_facts_file = proxy._user_file("personal_facts.json", default_user)
    default_prefs_file = proxy._user_file("personal_preferences.json", default_user)
    default_history_file = proxy._user_file("intent_history.json", default_user)

    if not os.path.exists(default_facts_file):
        proxy.save_personal_facts(proxy.generate_random_personal_facts(), default_user)
    if not os.path.exists(default_prefs_file):
        proxy.save_personal_preferences(
            proxy.generate_random_personal_preferences(), default_user
        )
    if not os.path.exists(default_history_file):
        proxy.save_intent_history(proxy.generate_random_intent_history(3), default_user)

    print("=" * 60)
    print("Personal Proxy Web Demo 启动中...")
    print("=" * 60)
    print(
        f"DeepSeek API Key: {'已配置' if DASHSCOPE_API_KEY else '未配置（请设置环境变量 DASHSCOPE_API_KEY）'}"
    )
    
    # 预加载本地 LLM 模型
    if TRANSFORMERS_AVAILABLE:
        print("\n正在预加载本地 LLM 模型...")
        print(f"模型路径: {LOCAL_LLM_MODEL_PATH}")
        try:
            model, tokenizer = local_llm_models.get_local_llm()
            print("✓ 本地 LLM 模型预加载成功")
        except Exception as e:
            print(f"✗ 本地 LLM 模型预加载失败: {str(e)}")
            print("  应用将继续启动，但意图判断功能可能无法正常工作")
    else:
        print("\n警告: transformers 未安装，无法加载本地 LLM 模型")
    print("=" * 60)
    print("应用已启动，可以通过以下地址访问：")
    print(f"  - http://localhost:{SERVER_PORT}")
    print(f"  - http://127.0.0.1:{SERVER_PORT}")
    print("=" * 60)
    print("如果无法访问，请检查：")
    print(f"  1. 防火墙是否允许 {SERVER_PORT} 端口")
    print(f"  2. 是否有其他程序占用 {SERVER_PORT} 端口")
    print(f"  3. 尝试使用 127.0.0.1:{SERVER_PORT} 而不是 localhost:{SERVER_PORT}")
    print("=" * 60)
    try:
        app.run(debug=True, host="0.0.0.0", port=SERVER_PORT, use_reloader=False)
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"\n错误：端口 {SERVER_PORT} 已被占用")
            print("请尝试：")
            print("  1. 关闭占用端口的程序")
            print("  2. 或修改 app.py 中的端口号")
        else:
            print(f"\n启动失败：{e}")
        raise

