import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from gem import Env
from openai import OpenAI
from roll.pipeline.agentic.env.parse_action_utils import default_parser_action_func
from roll.utils.logging import get_logger

logger = get_logger()


@dataclass
class IntentHistory:
    """意图判断历史记录"""

    context: str  # 情景context
    person: str  # 人
    intent: str  # 意图
    intent_reason: str  # 意图理由
    feedback: Optional[str] = None  # 用户反馈（修改后的意图或理由）
    feedback_reason: Optional[str] = None  # 反馈理由


class PersonalProxyEnv(Env):
    """
    Personal Proxy 环境：用于训练代理模型理解用户意图并生成个性化描述。

    工作流程：
    1. 使用 qwen api 模拟用户（系统提示词随机拟人）
    2. 预置个人事实（长期）和个人偏好（短期）属性
    3. 维护不同情景下的意图判断历史
    4. Proxy模型通过检索意图判断历史理解当前情景的意图
    5. 根据意图检索个人事实和偏好，汇总为情景相关的个人记忆
    6. 合并意图和个人记忆为用户情景感知的个人描述
    7. 使用 qwen api 对个人描述进行打分
    """

    def __init__(
        self,
        user_llm_config: Dict[str, Any],
        judge_llm_config: Dict[str, Any],
        personal_facts: Optional[List[str]] = None,
        personal_preferences: Optional[List[str]] = None,
        intent_history: Optional[List[IntentHistory]] = None,
        max_steps: int = 20,
        format_penalty: float = -0.1,
        action_pattern: str = r"<answer>(.*?)</answer>",
        special_token_list: Tuple[str, ...] = ("<|im_start|>", "<|im_end|>"),
        seed: Optional[int] = None,
        enable_user_feedback: bool = True,
        **kwargs,
    ):
        """
        初始化 Personal Proxy 环境

        Args:
            user_llm_config: 用户模拟LLM配置（qwen api）
                - api_key: API密钥
                - base_url: API基础URL
                - model_name: 模型名称
                - timeout: 超时时间
            judge_llm_config: 打分LLM配置（qwen api）
                - api_key: API密钥
                - base_url: API基础URL
                - model_name: 模型名称
                - timeout: 超时时间
            personal_facts: 个人事实列表（长期属性）
            personal_preferences: 个人偏好列表（短期属性）
            intent_history: 意图判断历史列表
            max_steps: 最大步数
            format_penalty: 格式错误惩罚
            action_pattern: 动作解析模式
            special_token_list: 特殊token列表
            seed: 随机种子
            enable_user_feedback: 是否启用用户反馈功能
        """
        Env.__init__(self)

        self.max_steps = max_steps
        self.format_penalty = format_penalty
        self.action_pattern = action_pattern
        self.special_token_list = special_token_list
        self.enable_user_feedback = enable_user_feedback

        # 初始化用户LLM（用于模拟用户）
        self.user_llm_config = user_llm_config
        self.user_client = None
        self.user_model_name = user_llm_config.get("model_name", "qwen-plus")
        self._init_user_llm()

        # 初始化打分LLM（用于对个人描述打分）
        self.judge_llm_config = judge_llm_config
        self.judge_client = None
        self.judge_model_name = judge_llm_config.get("model_name", "qwen-plus")
        self._init_judge_llm()

        # 个人属性
        self.personal_facts = personal_facts or []
        self.personal_preferences = personal_preferences or []

        # 意图判断历史
        self.intent_history: List[IntentHistory] = intent_history or []

        # 当前状态
        self.current_context = None
        self.current_person = None
        self.current_intent = None
        self.current_intent_reason = None
        self.current_personal_memory = None
        self.current_personal_description = None
        self.num_steps = 0

        # 用户系统提示词（随机拟人）
        self.user_system_prompts = [
            "你是一个忙碌的上班族，喜欢简洁高效的沟通方式。",
            "你是一个热情开朗的人，喜欢分享和讨论。",
            "你是一个内向安静的人，表达比较含蓄。",
            "你是一个理性分析型的人，注重逻辑和细节。",
            "你是一个感性的人，更关注情感和体验。",
            "你是一个幽默风趣的人，喜欢用轻松的方式交流。",
            "你是一个严谨认真的人，对事情要求较高。",
            "你是一个随和的人，容易接受建议。",
        ]
        self.current_user_persona = None

    def _init_user_llm(self):
        """初始化用户模拟LLM"""
        try:
            api_key = self.user_llm_config.get("api_key", None)
            base_url = self.user_llm_config.get(
                "base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
            timeout = self.user_llm_config.get("timeout", 60)

            if not api_key:
                logger.warning("用户LLM api_key未提供，将无法模拟用户")
                return

            self.user_client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
            )
            logger.info(
                f"用户LLM初始化成功: base_url={base_url}, model={self.user_model_name}"
            )
        except Exception as e:
            logger.warning(f"用户LLM初始化失败: {e}")

    def _init_judge_llm(self):
        """初始化打分LLM"""
        try:
            api_key = self.judge_llm_config.get("api_key", None)
            base_url = self.judge_llm_config.get(
                "base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
            timeout = self.judge_llm_config.get("timeout", 60)

            if not api_key:
                logger.warning("打分LLM api_key未提供，将无法进行打分")
                return

            self.judge_client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
            )
            logger.info(
                f"打分LLM初始化成功: base_url={base_url}, model={self.judge_model_name}"
            )
        except Exception as e:
            logger.warning(f"打分LLM初始化失败: {e}")

    def _generate_user_persona(self) -> str:
        """随机生成用户拟人系统提示词"""
        # 改为确定性选择，避免随机行为
        self.current_user_persona = (
            self.user_system_prompts[0] if self.user_system_prompts else ""
        )
        return self.current_user_persona

    def _simulate_user_response(self, context: str, person: str) -> Tuple[str, str]:
        """
        使用qwen api模拟用户在当前情景下的反应

        Returns:
            (用户反应文本, 用户意图)
        """
        if not self.user_client:
            # 如果没有LLM，返回默认反应
            return "我需要帮助", "寻求帮助"

        persona = self.current_user_persona or self._generate_user_persona()

        prompt = f"""你是一个真实用户，具有以下性格特点：
{persona}

当前情景：{context}
涉及的人：{person}

请以这个用户的身份，自然地表达你在当前情景下的反应和意图。
直接给出你的反应，不需要解释。"""

        try:
            response = self.user_client.chat.completions.create(
                model=self.user_model_name,
                messages=[
                    {"role": "system", "content": persona},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=200,
            )
            user_response = response.choices[0].message.content.strip()

            # 提取用户意图（简化版，实际可以用另一个LLM调用）
            intent_prompt = f"""基于以下用户反应，提取用户的意图（用一句话概括）：
用户反应：{user_response}
情景：{context}
涉及的人：{person}

只返回意图，不要其他内容。"""

            intent_response = self.user_client.chat.completions.create(
                model=self.user_model_name,
                messages=[
                    {"role": "system", "content": "你是一个意图提取专家。"},
                    {"role": "user", "content": intent_prompt},
                ],
                temperature=0.0,
                max_tokens=50,
            )
            intent = intent_response.choices[0].message.content.strip()

            return user_response, intent
        except Exception as e:
            logger.warning(f"用户模拟失败: {e}")
            return "我需要帮助", "寻求帮助"

    def _retrieve_intent_from_history(
        self, context: str, person: str
    ) -> Optional[IntentHistory]:
        """
        从意图判断历史中检索相似情景的意图

        Returns:
            最相似的意图历史记录，如果没有则返回None
        """
        if not self.intent_history:
            return None

        # 简单的相似度匹配（实际可以使用embedding相似度）
        best_match = None
        best_score = 0.0

        context_lower = context.lower()
        person_lower = person.lower()

        for history in self.intent_history:
            score = 0.0
            # 检查情景相似度
            if (
                context_lower in history.context.lower()
                or history.context.lower() in context_lower
            ):
                score += 0.5
            # 检查人物相似度
            if (
                person_lower in history.person.lower()
                or history.person.lower() in person_lower
            ):
                score += 0.3
            # 如果用户提供了反馈，说明这个历史记录更准确
            if history.feedback:
                score += 0.2

            if score > best_score:
                best_score = score
                best_match = history

        return best_match if best_score > 0.3 else None

    def _retrieve_personal_memory(self, intent: str) -> str:
        """
        根据意图检索相关的个人事实和偏好，汇总为情景相关的个人记忆

        Args:
            intent: 用户意图

        Returns:
            汇总后的个人记忆文本
        """
        relevant_facts = []
        relevant_preferences = []

        intent_lower = intent.lower()

        # 检索相关的个人事实
        for fact in self.personal_facts:
            # 简单的关键词匹配（实际可以使用更复杂的语义匹配）
            fact_lower = fact.lower()
            if any(keyword in fact_lower for keyword in intent_lower.split()):
                relevant_facts.append(fact)

        # 检索相关的个人偏好
        for pref in self.personal_preferences:
            pref_lower = pref.lower()
            if any(keyword in pref_lower for keyword in intent_lower.split()):
                relevant_preferences.append(pref)

        # 汇总个人记忆
        memory_parts = []
        if relevant_facts:
            memory_parts.append(
                "个人事实（长期）：\n" + "\n".join(f"- {f}" for f in relevant_facts)
            )
        if relevant_preferences:
            memory_parts.append(
                "个人偏好（短期）：\n"
                + "\n".join(f"- {p}" for p in relevant_preferences)
            )

        if not memory_parts:
            # 如果没有找到相关的，返回通用的
            if self.personal_facts:
                memory_parts.append(
                    "个人事实（长期）：\n"
                    + "\n".join(f"- {f}" for f in self.personal_facts[:3])
                )
            if self.personal_preferences:
                memory_parts.append(
                    "个人偏好（短期）：\n"
                    + "\n".join(f"- {p}" for p in self.personal_preferences[:3])
                )

        return "\n\n".join(memory_parts) if memory_parts else "暂无相关个人记忆"

    def _generate_personal_description(
        self, intent: str, intent_reason: str, personal_memory: str
    ) -> str:
        """
        合并意图和个人记忆为用户情景感知的个人描述

        Args:
            intent: 用户意图
            intent_reason: 意图理由
            personal_memory: 个人记忆

        Returns:
            用户情景感知的个人描述
        """
        description = f"""用户意图：{intent}

意图理由：{intent_reason}

{personal_memory}

基于以上信息，这是一个关于用户在当前情景下的个性化描述。"""

        return description

    def _judge_personal_description(
        self, personal_description: str, context: str, person: str
    ) -> Tuple[float, str]:
        """
        使用qwen api对个人描述进行打分

        Args:
            personal_description: 个人描述
            context: 当前情景
            person: 涉及的人

        Returns:
            (分数 0-1, 评分理由)
        """
        if not self.judge_client:
            # 如果没有LLM，返回默认分数
            return 0.5, "无法进行LLM评分"

        prompt = f"""请评估以下个人描述的质量，从0到1打分（1为最高分）。

当前情景：{context}
涉及的人：{person}

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

        try:
            response = self.judge_client.chat.completions.create(
                model=self.judge_model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个专业的评估专家，负责评估个人描述的质量。",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=300,
            )

            judge_response = response.choices[0].message.content.strip()

            # 提取分数
            score_match = re.search(r"分数[：:]\s*([0-9.]+)", judge_response)
            if score_match:
                score = float(score_match.group(1))
                score = max(0.0, min(1.0, score))  # 限制在0-1之间
            else:
                score = 0.5
                logger.warning("无法从评分响应中提取分数")

            # 提取理由
            reason_match = re.search(r"理由[：:]\s*(.+)", judge_response, re.DOTALL)
            if reason_match:
                reason = reason_match.group(1).strip()
            else:
                reason = judge_response

            return score, reason
        except Exception as e:
            logger.warning(f"LLM评分失败: {e}")
            return 0.5, f"评分失败: {str(e)}"

    def get_instructions(self) -> str:
        """获取环境指令"""
        instruction = """你是一个Personal Proxy模型，需要理解用户在当前情景下的意图，并生成个性化的用户描述。

你的任务流程：
1. 根据当前情景和涉及的人，理解用户的意图
2. 从意图判断历史中检索相似情景的意图（如果存在）
3. 根据意图检索相关的个人事实（长期）和个人偏好（短期）
4. 汇总为情景相关的个人记忆
5. 合并意图和个人记忆，生成用户情景感知的个人描述

当前状态：
"""
        if self.current_context:
            instruction += f"情景：{self.current_context}\n"
        if self.current_person:
            instruction += f"涉及的人：{self.current_person}\n"
        if self.current_intent:
            instruction += f"当前意图：{self.current_intent}\n"
        if self.current_personal_memory:
            instruction += f"\n个人记忆：\n{self.current_personal_memory}\n"

        instruction += "\n请生成用户情景感知的个人描述，格式：<answer>你的描述</answer>"

        return instruction

    def reset(self, seed: Optional[int] = None):
        """重置环境"""
        Env.reset(self, seed)

        try:
            # 生成固定情景和人物
            contexts = [
                "在办公室开会讨论项目进度",
                "在家准备晚餐",
                "在健身房锻炼",
                "在咖啡店和朋友聊天",
                "在商场购物",
                "在图书馆学习",
                "在医院看病",
                "在餐厅点餐",
            ]
            persons = [
                "同事",
                "家人",
                "朋友",
                "陌生人",
                "服务人员",
                "医生",
                "老师",
            ]

            # 使用固定选择，避免随机行为
            self.current_context = contexts[0] if contexts else None
            self.current_person = persons[0] if persons else None

            # 生成用户拟人
            self._generate_user_persona()

            # 模拟用户反应和意图
            user_response, user_intent = self._simulate_user_response(
                self.current_context, self.current_person
            )

            # 从历史中检索相似意图
            history_match = self._retrieve_intent_from_history(
                self.current_context, self.current_person
            )

            if history_match:
                # 使用历史记录中的意图（如果用户有反馈，使用反馈）
                self.current_intent = history_match.feedback or history_match.intent
                self.current_intent_reason = (
                    history_match.feedback_reason or history_match.intent_reason
                )
            else:
                # 使用新生成的意图
                self.current_intent = user_intent
                self.current_intent_reason = f"基于用户反应'{user_response}'推断的意图"

            # 检索个人记忆
            self.current_personal_memory = self._retrieve_personal_memory(
                self.current_intent
            )

            # 重置状态
            self.current_personal_description = None
            self.num_steps = 0

            observation = self.get_instructions()
            info = {
                "env_instruction": observation,
                "context": self.current_context,
                "person": self.current_person,
                "user_response": user_response,
                "intent": self.current_intent,
                "intent_reason": self.current_intent_reason,
                "personal_memory": self.current_personal_memory,
            }

            return observation, info

        except (RuntimeError, RuntimeWarning, ValueError) as e:
            logger.warning(f"重置失败: {e}")
            # 使用简单的加一策略生成下一个种子，避免使用具有随机性的 hash 行为
            next_seed = ((seed or 0) + 1) % (2**32)
            return self.reset(next_seed)

    def step(self, action: str):
        """执行一步"""
        metrics_agg_mode = {
            "action_is_valid": "mean",
            "format_penalty": "mean",
            "description_score": "mean",
            "intent_accuracy": "mean",
        }

        self.num_steps += 1

        # 解析动作
        action_info = self.parse_action(action)

        if action_info["action"] is None:
            # 无效动作
            reward = self.format_penalty
            metrics = {
                "action_is_valid": False,
                "format_penalty": self.format_penalty,
                "description_score": 0.0,
                "intent_accuracy": 0.0,
            }
            info = {
                "metrics": metrics,
                "metrics_agg_mode": metrics_agg_mode,
                "action_desc": f"第{self.num_steps}步，未提供有效的个人描述",
            }
            info.update(action_info)

            observation = self.get_instructions()
            terminated = self.num_steps >= self.max_steps
            truncated = False

            return observation, reward, terminated, truncated, info

        # 提取个人描述
        personal_description = action_info["action_content"] or action_info["action"]
        self.current_personal_description = personal_description

        # 使用LLM对描述进行打分
        score, reason = self._judge_personal_description(
            personal_description, self.current_context, self.current_person
        )

        # 计算奖励（分数作为主要奖励）
        reward = score

        # 意图准确性（简化版，检查描述中是否包含意图关键词）
        intent_accuracy = 0.0
        if self.current_intent:
            intent_keywords = set(self.current_intent.lower().split())
            desc_lower = personal_description.lower()
            matched = sum(1 for kw in intent_keywords if kw in desc_lower)
            intent_accuracy = matched / max(1, len(intent_keywords))

        # 保存意图判断历史（如果这是新的）
        if not self._retrieve_intent_from_history(
            self.current_context, self.current_person
        ):
            new_history = IntentHistory(
                context=self.current_context,
                person=self.current_person,
                intent=self.current_intent,
                intent_reason=self.current_intent_reason,
            )
            self.intent_history.append(new_history)

        metrics = {
            "action_is_valid": True,
            "format_penalty": 0.0,
            "description_score": score,
            "intent_accuracy": intent_accuracy,
        }

        action_desc = f"第{self.num_steps}步，生成个人描述，得分：{score:.2f}"

        info = {
            "metrics": metrics,
            "metrics_agg_mode": metrics_agg_mode,
            "action_desc": action_desc,
            "personal_description": personal_description,
            "description_score": score,
            "score_reason": reason,
            "intent_accuracy": intent_accuracy,
        }
        info.update(action_info)

        observation = self.get_instructions()
        terminated = self.num_steps >= self.max_steps
        truncated = False

        return observation, reward, terminated, truncated, info

    def parse_action(self, text: str):
        """解析动作"""
        return default_parser_action_func(
            text, self.action_pattern, {}, self.special_token_list
        )

    def provide_user_feedback(
        self,
        intent: Optional[str] = None,
        intent_reason: Optional[str] = None,
        feedback_reason: Optional[str] = None,
    ):
        """
        提供用户反馈，修改意图判断结果和理由

        Args:
            intent: 修改后的意图（如果为None则不修改）
            intent_reason: 修改后的意图理由（如果为None则不修改）
            feedback_reason: 反馈理由
        """
        if not self.enable_user_feedback:
            logger.warning("用户反馈功能未启用")
            return

        # 找到当前情景对应的历史记录
        history_match = self._retrieve_intent_from_history(
            self.current_context, self.current_person
        )

        if history_match:
            if intent is not None:
                history_match.feedback = intent
            if intent_reason is not None:
                history_match.feedback_reason = intent_reason
            if feedback_reason is not None:
                history_match.feedback_reason = feedback_reason
        else:
            # 创建新的历史记录
            new_history = IntentHistory(
                context=self.current_context,
                person=self.current_person,
                intent=intent or self.current_intent,
                intent_reason=intent_reason or self.current_intent_reason,
                feedback=intent,
                feedback_reason=feedback_reason,
            )
            self.intent_history.append(new_history)

        logger.info(f"用户反馈已记录：意图={intent}, 理由={intent_reason}")

    def add_personal_fact(self, fact: str):
        """添加个人事实（长期）"""
        if fact not in self.personal_facts:
            self.personal_facts.append(fact)
            logger.info(f"已添加个人事实：{fact}")

    def add_personal_preference(self, preference: str):
        """添加个人偏好（短期）"""
        if preference not in self.personal_preferences:
            self.personal_preferences.append(preference)
            logger.info(f"已添加个人偏好：{preference}")

    def get_intent_history(self) -> List[Dict]:
        """获取意图判断历史（转换为字典格式）"""
        return [asdict(h) for h in self.intent_history]

    def sample_random_action(self):
        """采样随机动作（用于测试）"""
        return "<answer>这是一个测试的个人描述，包含用户意图和个人记忆信息。</answer>"

    def render(self, mode: str = "text"):
        """渲染当前状态"""
        if mode == "text":
            lines = [
                f"情景：{self.current_context}",
                f"涉及的人：{self.current_person}",
                f"用户意图：{self.current_intent}",
                f"意图理由：{self.current_intent_reason}",
                f"个人记忆：{self.current_personal_memory}",
            ]
            if self.current_personal_description:
                lines.append(f"个人描述：{self.current_personal_description}")
            return "\n".join(lines)
        else:
            raise ValueError(f"不支持的渲染模式: {mode}")

    def close(self):
        """清理资源"""
        self.intent_history = []
        self.current_context = None
        self.current_person = None
        self.current_intent = None
        self.current_personal_description = None
