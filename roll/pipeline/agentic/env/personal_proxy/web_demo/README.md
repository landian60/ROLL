# Personal Proxy Web Demo

基于 Web 的个性化代理系统演示，接入 DeepSeek API，支持中文提示。

## 功能特性

1. **个人事实管理（长期）**
   - 年龄
   - 性别
   - 就业状态
   - 教育水平
   - 居住区域
   - 支持随机生成和手动编辑

2. **个人偏好管理（短期）**
   - 是否喜欢阅读长文本
   - 是否喜欢大纲导览
   - 支持随机生成和手动编辑

3. **意图判断历史管理**
   - 当前情景
   - 用户
   - 意图
   - 意图解释
   - 用户反馈
   - 支持添加、编辑、删除和随机生成

4. **知识搜索场景**
   - 介绍YOLO结构
   - 介绍伊斯坦布尔历史
   - 支持自定义场景

5. **个性化描述生成**
   - 基于当前情景和意图判断历史生成个性化描述
   - 使用 DeepSeek API 进行质量评分
   - 显示个人记忆和意图解释

## 安装和运行

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 设置环境变量

设置阿里云 DashScope API Key（用于 deepseek-v3.2-exp）：

```bash
export DASHSCOPE_API_KEY="your-dashscope-api-key"
```

或者在运行前设置：

```bash
DASHSCOPE_API_KEY="your-dashscope-api-key" python app.py
```

### 3. 运行应用

```bash
python app.py
```

应用将在 `http://localhost:5000` 启动。

## 使用说明

### 页面说明

系统包含两个页面：

1. **个人资料管理页面** (`/`)：配置个人事实、偏好和意图判断历史
2. **知识搜索页面** (`/search`)：进行个性化知识搜索，查看思考过程和最终回答

### 1. 配置个人事实和偏好（个人资料管理页面）

- 在左侧面板填写或编辑个人事实（年龄、性别、就业状态、教育水平、居住区域）
- 配置个人偏好（是否喜欢阅读长文本、是否喜欢大纲导览）
- 可以点击"随机生成"按钮快速生成示例数据
- 点击"保存"按钮保存配置

### 2. 管理意图判断历史（个人资料管理页面）

- 在右侧面板查看所有意图判断历史
- 点击"添加新意图"按钮添加新的意图判断记录
- 点击"编辑"按钮修改已有记录
- 点击"删除"按钮删除记录
- 点击"随机生成"按钮生成示例历史记录

### 3. 个性化知识搜索（知识搜索页面）

- 点击主页右上角的"🔍 知识搜索"按钮或访问 `/search` 页面
- 在搜索框输入查询（例如："介绍YOLO结构"、"介绍伊斯坦布尔历史"）
- 点击"搜索"按钮
- 系统会使用两个模型：
  1. **Proxy模型**：根据意图判断历史生成当前查询的意图，检索相关的个人事实和偏好，生成个性化描述
  2. **LLM模型**：接受原始查询和个性化描述，生成最终回答（包含思考过程）
- 查看结果：
  - **个性化信息**：显示意图、意图解释、个人记忆和个性化描述
  - **思考过程**：显示LLM模型的推理过程（如果启用）
  - **回答**：显示最终的回答内容

## 数据存储

所有数据保存在 `data/` 目录下：

- `personal_facts.json`: 个人事实
- `personal_preferences.json`: 个人偏好
- `intent_history.json`: 意图判断历史

数据以 JSON 格式存储，可以直接编辑。

## API 接口

### 个人事实

- `GET /api/personal-facts`: 获取个人事实
- `POST /api/personal-facts`: 更新个人事实
- `POST /api/personal-facts/random`: 生成随机个人事实

### 个人偏好

- `GET /api/personal-preferences`: 获取个人偏好
- `POST /api/personal-preferences`: 更新个人偏好
- `POST /api/personal-preferences/random`: 生成随机个人偏好

### 意图判断历史

- `GET /api/intent-history`: 获取所有意图判断历史
- `POST /api/intent-history`: 添加新的意图判断历史
- `PUT /api/intent-history/<intent_id>`: 更新指定的意图判断历史
- `DELETE /api/intent-history/<intent_id>`: 删除指定的意图判断历史
- `POST /api/intent-history/random`: 生成随机意图判断历史

### 场景和描述生成

- `GET /api/scenarios`: 获取所有知识搜索场景
- `POST /api/generate-description`: 生成个性化描述

## 技术栈

- **后端**: Flask
- **前端**: HTML + CSS + JavaScript
- **AI API**: DeepSeek API
- **数据存储**: JSON 文件

## 注意事项

1. 确保已正确配置 DeepSeek API Key
2. 首次运行会自动创建 `data/` 目录并初始化示例数据
3. 所有数据都保存在本地 JSON 文件中
4. 支持中文输入和显示

## 扩展建议

1. 使用数据库替代 JSON 文件存储
2. 添加用户认证和会话管理
3. 实现更高级的意图检索（使用 embedding 相似度）
4. 添加更多知识搜索场景
5. 支持批量导入/导出数据
6. 添加数据可视化功能
