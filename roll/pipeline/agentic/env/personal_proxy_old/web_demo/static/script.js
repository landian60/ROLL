// API 基础 URL
const API_BASE = '';

// 当前编辑的意图ID
let currentEditIntentId = null;

// 页面加载时初始化
document.addEventListener('DOMContentLoaded', function() {
    // 只在index.html页面加载个人事实和偏好
    if (document.getElementById('age')) {
        loadPersonalFacts();
    }
    if (document.getElementById('like_long_text')) {
        loadPersonalPreferences();
    }
    // 只在index.html页面加载意图历史
    if (document.getElementById('intent_history_list')) {
        loadIntentHistory();
    }
    // 只在index.html页面加载场景
    if (document.getElementById('scenario_select')) {
        loadScenarios();
    }
});

// ========== 个人事实相关 ==========
async function loadPersonalFacts() {
    try {
        const ageEl = document.getElementById('age');
        if (!ageEl) return; // 如果元素不存在，直接返回
        
        const response = await fetch(`${API_BASE}/api/personal-facts`);
        const result = await response.json();
        if (result.success) {
            const data = result.data;
            if (ageEl) ageEl.value = data.age || '';
            const genderEl = document.getElementById('gender');
            if (genderEl) genderEl.value = data.gender || '';
            const employmentEl = document.getElementById('employment_status');
            if (employmentEl) employmentEl.value = data.employment_status || '';
            const educationEl = document.getElementById('education_level');
            if (educationEl) educationEl.value = data.education_level || '';
            const residenceEl = document.getElementById('residence_area');
            if (residenceEl) residenceEl.value = data.residence_area || '';
        }
    } catch (error) {
        console.error('加载个人事实失败:', error);
        // 只在index.html页面显示错误
        if (document.getElementById('age')) {
            alert('加载个人事实失败: ' + error.message);
        }
    }
}

async function savePersonalFacts() {
    const facts = {
        age: document.getElementById('age').value,
        gender: document.getElementById('gender').value,
        employment_status: document.getElementById('employment_status').value,
        education_level: document.getElementById('education_level').value,
        residence_area: document.getElementById('residence_area').value
    };
    
    try {
        const response = await fetch(`${API_BASE}/api/personal-facts`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(facts)
        });
        const result = await response.json();
        if (result.success) {
            alert('个人事实已保存！');
        } else {
            alert('保存失败: ' + result.message);
        }
    } catch (error) {
        console.error('保存个人事实失败:', error);
        alert('保存失败: ' + error.message);
    }
}

async function generateRandomFacts() {
    try {
        const response = await fetch(`${API_BASE}/api/personal-facts/random`, {
            method: 'POST'
        });
        const result = await response.json();
        if (result.success) {
            const data = result.data;
            document.getElementById('age').value = data.age || '';
            document.getElementById('gender').value = data.gender || '';
            document.getElementById('employment_status').value = data.employment_status || '';
            document.getElementById('education_level').value = data.education_level || '';
            document.getElementById('residence_area').value = data.residence_area || '';
            alert('已生成随机个人事实！');
        }
    } catch (error) {
        console.error('生成随机个人事实失败:', error);
        alert('生成失败: ' + error.message);
    }
}

// ========== 个人偏好相关 ==========
async function loadPersonalPreferences() {
    try {
        const likeLongTextEl = document.getElementById('like_long_text');
        if (!likeLongTextEl) return; // 如果元素不存在，直接返回
        
        const response = await fetch(`${API_BASE}/api/personal-preferences`);
        const result = await response.json();
        if (result.success) {
            const data = result.data;
            if (likeLongTextEl) likeLongTextEl.checked = data.like_long_text || false;
            const likeOutlineEl = document.getElementById('like_outline_navigation');
            if (likeOutlineEl) likeOutlineEl.checked = data.like_outline_navigation || false;
        }
    } catch (error) {
        console.error('加载个人偏好失败:', error);
        // 只在index.html页面显示错误
        if (document.getElementById('like_long_text')) {
            alert('加载个人偏好失败: ' + error.message);
        }
    }
}

async function savePersonalPreferences() {
    const preferences = {
        like_long_text: document.getElementById('like_long_text').checked,
        like_outline_navigation: document.getElementById('like_outline_navigation').checked
    };
    
    try {
        const response = await fetch(`${API_BASE}/api/personal-preferences`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(preferences)
        });
        const result = await response.json();
        if (result.success) {
            alert('个人偏好已保存！');
        } else {
            alert('保存失败: ' + result.message);
        }
    } catch (error) {
        console.error('保存个人偏好失败:', error);
        alert('保存失败: ' + error.message);
    }
}

async function generateRandomPreferences() {
    try {
        const response = await fetch(`${API_BASE}/api/personal-preferences/random`, {
            method: 'POST'
        });
        const result = await response.json();
        if (result.success) {
            const data = result.data;
            document.getElementById('like_long_text').checked = data.like_long_text || false;
            document.getElementById('like_outline_navigation').checked = data.like_outline_navigation || false;
            alert('已生成随机个人偏好！');
        }
    } catch (error) {
        console.error('生成随机个人偏好失败:', error);
        alert('生成失败: ' + error.message);
    }
}

// ========== 意图判断历史相关 ==========
async function loadIntentHistory() {
    try {
        const response = await fetch(`${API_BASE}/api/intent-history`);
        const result = await response.json();
        if (result.success) {
            renderIntentHistory(result.data);
        }
    } catch (error) {
        console.error('加载意图历史失败:', error);
        alert('加载意图历史失败: ' + error.message);
    }
}

function renderIntentHistory(history) {
    const container = document.getElementById('intent_history_list');
    if (!container) {
        console.warn('intent_history_list 元素不存在，可能不在当前页面');
        return;
    }
    if (history.length === 0) {
        container.innerHTML = '<p style="color: #999; text-align: center; padding: 20px;">暂无意图判断历史</p>';
        return;
    }
    
    container.innerHTML = history.map(item => `
        <div class="intent-item">
            <div class="intent-item-header">
                <div class="intent-item-title">${item.context || '未命名'}</div>
                <div class="intent-item-actions">
                    <button class="btn btn-small" onclick="editIntent('${item.id}')">编辑</button>
                    <button class="btn btn-small btn-secondary" onclick="deleteIntent('${item.id}')">删除</button>
                </div>
            </div>
            <div class="intent-item-content">
                <div><strong>用户：</strong>${item.user || '当前用户'}</div>
                <div><strong>意图：</strong>${item.intent || ''}</div>
                <div><strong>意图解释：</strong>${item.intent_explanation || ''}</div>
                ${item.user_feedback ? `<div><strong>用户反馈：</strong>${item.user_feedback}</div>` : ''}
            </div>
        </div>
    `).join('');
}

function showAddIntentForm() {
    currentEditIntentId = null;
    document.getElementById('modal_title').textContent = '添加意图判断';
    document.getElementById('modal_context').value = '';
    document.getElementById('modal_user').value = '当前用户';
    document.getElementById('modal_intent').value = '';
    document.getElementById('modal_intent_explanation').value = '';
    document.getElementById('modal_user_feedback').value = '';
    document.getElementById('intent_modal').style.display = 'block';
}

function editIntent(intentId) {
    currentEditIntentId = intentId;
    fetch(`${API_BASE}/api/intent-history`)
        .then(res => res.json())
        .then(result => {
            if (result.success) {
                const item = result.data.find(h => h.id === intentId);
                if (item) {
                    document.getElementById('modal_title').textContent = '编辑意图判断';
                    document.getElementById('modal_context').value = item.context || '';
                    document.getElementById('modal_user').value = item.user || '当前用户';
                    document.getElementById('modal_intent').value = item.intent || '';
                    document.getElementById('modal_intent_explanation').value = item.intent_explanation || '';
                    document.getElementById('modal_user_feedback').value = item.user_feedback || '';
                    document.getElementById('intent_modal').style.display = 'block';
                }
            }
        })
        .catch(error => {
            console.error('加载意图详情失败:', error);
            alert('加载失败: ' + error.message);
        });
}

function closeIntentModal() {
    document.getElementById('intent_modal').style.display = 'none';
    currentEditIntentId = null;
}

async function saveIntent() {
    const data = {
        context: document.getElementById('modal_context').value,
        user: document.getElementById('modal_user').value,
        intent: document.getElementById('modal_intent').value,
        intent_explanation: document.getElementById('modal_intent_explanation').value,
        user_feedback: document.getElementById('modal_user_feedback').value
    };
    
    if (!data.context || !data.intent) {
        alert('请填写当前情景和意图');
        return;
    }
    
    try {
        let response;
        if (currentEditIntentId) {
            // 更新
            response = await fetch(`${API_BASE}/api/intent-history/${currentEditIntentId}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });
        } else {
            // 新增
            response = await fetch(`${API_BASE}/api/intent-history`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });
        }
        
        const result = await response.json();
        if (result.success) {
            alert('保存成功！');
            closeIntentModal();
            loadIntentHistory();
        } else {
            alert('保存失败: ' + result.message);
        }
    } catch (error) {
        console.error('保存意图失败:', error);
        alert('保存失败: ' + error.message);
    }
}

async function deleteIntent(intentId) {
    if (!confirm('确定要删除这条意图判断历史吗？')) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/api/intent-history/${intentId}`, {
            method: 'DELETE'
        });
        const result = await response.json();
        if (result.success) {
            alert('已删除！');
            loadIntentHistory();
        } else {
            alert('删除失败: ' + result.message);
        }
    } catch (error) {
        console.error('删除意图失败:', error);
        alert('删除失败: ' + error.message);
    }
}

async function generateRandomHistory() {
    if (!confirm('确定要生成随机意图判断历史吗？这将添加3条新的历史记录。')) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/api/intent-history/random`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ count: 3 })
        });
        const result = await response.json();
        if (result.success) {
            alert('已生成随机意图判断历史！');
            loadIntentHistory();
        }
    } catch (error) {
        console.error('生成随机历史失败:', error);
        alert('生成失败: ' + error.message);
    }
}

// ========== 场景相关 ==========
async function loadScenarios() {
    try {
        const response = await fetch(`${API_BASE}/api/scenarios`);
        const result = await response.json();
        if (result.success) {
            const select = document.getElementById('scenario_select');
            result.data.forEach(scenario => {
                const option = document.createElement('option');
                option.value = scenario.context;
                option.textContent = scenario.context;
                select.appendChild(option);
            });
        }
    } catch (error) {
        console.error('加载场景失败:', error);
    }
}

function selectScenario() {
    const select = document.getElementById('scenario_select');
    document.getElementById('custom_scenario').value = select.value;
}

// ========== 生成个性化描述 ==========
async function generateDescription() {
    const scenarioSelect = document.getElementById('scenario_select').value;
    const customScenario = document.getElementById('custom_scenario').value.trim();
    
    const context = customScenario || scenarioSelect;
    
    if (!context) {
        alert('请选择或输入一个搜索场景');
        return;
    }
    
    // 显示加载状态
    const resultBox = document.getElementById('description_result');
    if (!resultBox) {
        alert('无法找到结果显示区域');
        return;
    }
    resultBox.style.display = 'block';
    resultBox.innerHTML = '<p style="text-align: center; padding: 20px;">正在生成，请稍候...</p>';
    
    try {
        const response = await fetch(`${API_BASE}/api/generate-description`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ context })
        });
        
        const result = await response.json();
        if (result.success) {
            const data = result.data;
            
            // 直接设置结果框的完整内容
            resultBox.innerHTML = `
                <h3>生成结果</h3>
                <div class="result-item">
                    <strong>当前情景：</strong>
                    <span id="result_context">${data.context}</span>
                </div>
                <div class="result-item">
                    <strong>用户意图：</strong>
                    <span id="result_intent">${data.intent}</span>
                </div>
                <div class="result-item">
                    <strong>意图解释：</strong>
                    <span id="result_intent_explanation">${data.intent_explanation}</span>
                </div>
                <div class="result-item">
                    <strong>个人记忆：</strong>
                    <pre id="result_personal_memory">${data.personal_memory}</pre>
                </div>
                <div class="result-item">
                    <strong>个性化描述：</strong>
                    <pre id="result_personal_description">${data.personal_description}</pre>
                </div>
                <div class="result-item score-item">
                    <strong>质量评分：</strong>
                    <span class="score" id="result_score">${(data.score * 100).toFixed(1)}%</span>
                </div>
                <div class="result-item">
                    <strong>评分理由：</strong>
                    <p id="result_score_reason">${data.score_reason}</p>
                </div>
                <div class="result-item" style="margin-top: 20px; text-align: center;">
                    <button class="btn btn-primary btn-large" onclick="window.goToSearchWithDescription('${(data.context || '').replace(/'/g, "\\'")}', ${JSON.stringify(data.personal_description || '')}, ${JSON.stringify(data.intent || '')}, ${JSON.stringify(data.intent_explanation || '')}, ${JSON.stringify(data.personal_memory || '')})">
                        使用此个性化描述进行知识搜索
                    </button>
                </div>
            `;
        } else {
            // 显示错误信息
            resultBox.innerHTML = `
                <h3 style="color: #dc3545;">生成失败</h3>
                <p style="color: #dc3545;">${result.message || '未知错误'}</p>
            `;
        }
    } catch (error) {
        console.error('生成描述失败:', error);
        // 显示错误信息
        const resultBox = document.getElementById('description_result');
        if (resultBox) {
            resultBox.innerHTML = `
                <h3 style="color: #dc3545;">生成失败</h3>
                <p style="color: #dc3545;">${error.message || '网络错误或服务器异常'}</p>
            `;
            resultBox.style.display = 'block';
        } else {
            alert('生成失败: ' + error.message);
        }
    }
}

// 点击模态框外部关闭
window.onclick = function(event) {
    const modal = document.getElementById('intent_modal');
    if (modal && event.target == modal) {
        closeIntentModal();
    }
}

// ========== 跳转到搜索页面 ==========
// 确保函数是全局的
window.goToSearchWithDescription = function(query, personalDescription, intent, intentExplanation, personalMemory) {
    try {
        // 将数据存储到sessionStorage，跳转到搜索页面
        sessionStorage.setItem('pendingSearch', JSON.stringify({
            query: query,
            personal_description: personalDescription,
            intent: intent,
            intent_explanation: intentExplanation,
            personal_memory: personalMemory
        }));
        
        // 跳转到搜索页面
        window.location.href = '/search';
    } catch (e) {
        console.error('跳转失败:', e);
        alert('跳转失败: ' + e.message);
    }
}

