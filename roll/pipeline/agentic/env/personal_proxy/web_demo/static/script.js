// API 基础 URL
const API_BASE = '';

// 当前编辑的意图ID
let currentEditIntentId = null;
let isRedirectingToLogin = false;

let intentCategories = [];
let intentHistory = [];
let currentPersonalDescription = null; // 存储当前的个性化描述数据
let lastSavedIntentId = null; // 存储最后自动保存的意图ID

function redirectToLogin() {
    if (isRedirectingToLogin) {
        return;
    }
    isRedirectingToLogin = true;
    alert('登录状态已过期，请重新登录。');
    const next = encodeURIComponent(window.location.pathname + window.location.search);
    window.location.href = `/login?next=${next}`;
}

async function fetchWithAuth(url, options = {}) {
    // 确保包含凭证（用于Flask session）
    options.credentials = options.credentials || 'same-origin';
    const response = await fetch(url, options);
    if (response.status === 401) {
        redirectToLogin();
        return null;
    }
    return response;
}

async function fetchJson(url, options = {}) {
    const response = await fetchWithAuth(url, options);
    if (!response) {
        return null;
    }
    return response.json();
}

// 页面加载时初始化
document.addEventListener('DOMContentLoaded', function() {
    // 只在index.html页面加载个人事实和偏好
    if (document.getElementById('age')) {
        loadPersonalFacts();
    }
    if (document.getElementById('preferences_list')) {
        loadPersonalPreferences();
    }
    // 只在index.html页面加载意图历史
    if (document.getElementById('intent_history_list')) {
        loadIntentHistory();
    }
});

// ========== 个人事实相关 ==========
async function loadPersonalFacts() {
    try {
        const ageEl = document.getElementById('age');
        if (!ageEl) return; // 如果元素不存在，直接返回
        
        const result = await fetchJson(`${API_BASE}/api/personal-facts`);
        if (!result) return;
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
        const result = await fetchJson(`${API_BASE}/api/personal-facts`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(facts)
        });
        if (!result) return;
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

// ========== 个人偏好相关 ==========
let currentPreferences = [];
let parsedPreferencesCache = [];

async function loadPersonalPreferences() {
    try {
        const preferencesListEl = document.getElementById('preferences_list');
        if (!preferencesListEl) return; // 如果元素不存在，直接返回
        
        const result = await fetchJson(`${API_BASE}/api/personal-preferences`);
        if (!result) return;
        if (result.success) {
            currentPreferences = result.data || [];
            renderPreferencesList();
        }
    } catch (error) {
        console.error('加载个人偏好失败:', error);
        if (document.getElementById('preferences_list')) {
            alert('加载个人偏好失败: ' + error.message);
        }
    }
}

function renderPreferencesList() {
    const container = document.getElementById('preferences_list');
    if (!container) return;
    
    if (currentPreferences.length === 0) {
        container.innerHTML = '<p style="color: #999; text-align: center;">暂无偏好，请添加或上传文件</p>';
        return;
    }
    
    container.innerHTML = currentPreferences.map(pref => {
        const prefType = pref.preference_type || 'like';
        const isLike = prefType === 'like';
        const typeLabel = isLike ? '<span style="color: #4CAF50; font-weight: bold; margin-right: 8px;">✓ 喜欢</span>' : '<span style="color: #f44336; font-weight: bold; margin-right: 8px;">✗ 不喜欢</span>';
        const borderColor = isLike ? '#4CAF50' : '#f44336';
        const bgColor = pref.selected ? (isLike ? '#f0f8ff' : '#fff5f5') : '#fff';
        const toggleTypeText = isLike ? '改为不喜欢' : '改为喜欢';
        const toggleTypeColor = isLike ? '#f44336' : '#4CAF50';
        
        return `
        <div class="preference-item" style="display: flex; align-items: center; padding: 8px; border: 1px solid ${borderColor}; border-radius: 4px; margin-bottom: 8px; background: ${bgColor};">
            <input type="checkbox" 
                   id="pref_${pref.id}" 
                   ${pref.selected ? 'checked' : ''} 
                   onchange="togglePreference('${pref.id}')"
                   style="margin-right: 10px;">
            ${typeLabel}
            <label for="pref_${pref.id}" style="flex: 1; cursor: pointer; margin: 0;">${pref.text}</label>
            <button class="btn btn-small" onclick="togglePreferenceType('${pref.id}')" style="padding: 4px 8px; font-size: 12px; margin-right: 5px; background: ${toggleTypeColor}; color: white; border: none; border-radius: 3px; cursor: pointer;">${toggleTypeText}</button>
            <button class="btn btn-small btn-secondary" onclick="deletePreference('${pref.id}')" style="padding: 4px 8px; font-size: 12px;">删除</button>
        </div>
        `;
    }).join('');
}


async function uploadPreferenceFile() {
    const fileInput = document.getElementById('preference_file');
    if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
        alert('请先选择文件');
        return;
    }
    
    // 显示文件类别选择模态框
    const modal = document.getElementById('file_category_modal');
    if (modal) {
        modal.style.display = 'block';
        // 重置选择
        const radios = document.getElementsByName('file_category');
        if (radios.length > 0) radios[0].checked = true;
        document.getElementById('other_source_input').style.display = 'none';
        document.getElementById('other_source_detail').value = '';
    }
}

function closeFileCategoryModal() {
    const modal = document.getElementById('file_category_modal');
    if (modal) modal.style.display = 'none';
}

async function confirmUploadWithCategory() {
    // 获取选择的类别
    const selectedRadio = document.querySelector('input[name="file_category"]:checked');
    if (!selectedRadio) {
        alert('请选择文件内容类别');
        return;
    }
    
    let category = selectedRadio.value;
    if (category === '其他来源') {
        const detail = document.getElementById('other_source_detail').value.trim();
        if (!detail) {
            alert('请输入具体的来源说明');
            return;
        }
        category = `其他来源：${detail}`;
    }
    
    // 关闭模态框
    closeFileCategoryModal();
    
    // 开始上传和解析
    const fileInput = document.getElementById('preference_file');
    if (!fileInput || !fileInput.files || fileInput.files.length === 0) return;
    
    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);
    formData.append('category', category); // 添加类别信息
    
    try {
        // 显示加载状态（这里可以优化加一个全局loading）
        const btn = document.querySelector('button[onclick="uploadPreferenceFile()"]');
        const originalText = btn ? btn.innerText : '解析文件';
        if (btn) {
            btn.innerText = '正在解析...';
            btn.disabled = true;
        }
        
        const response = await fetchWithAuth(`${API_BASE}/api/preferences/upload-file`, {
            method: 'POST',
            body: formData
        });
        
        if (!response) {
            if (btn) {
                btn.innerText = originalText;
                btn.disabled = false;
            }
            return;
        }
        
        // 检查响应类型
        const contentType = response.headers.get('content-type');
        if (!contentType || !contentType.includes('application/json')) {
            const text = await response.text();
            console.error('服务器返回非JSON响应:', text.substring(0, 200));
            alert('服务器返回错误，请检查您是否已登录，或联系管理员');
            if (btn) {
                btn.innerText = originalText;
                btn.disabled = false;
            }
            return;
        }
        
        const result = await response.json();
        
        if (result.success) {
            parsedPreferencesCache = result.data || [];
            if (parsedPreferencesCache.length === 0) {
                alert('未从文件中提取到个人偏好');
                return;
            }
            // 显示模态框让用户选择
            showPreferenceSelectionModal(parsedPreferencesCache);
        } else {
            alert('解析失败: ' + result.message);
        }
    } catch (error) {
        console.error('上传文件失败:', error);
        alert('上传失败: ' + error.message);
    } finally {
        fileInput.value = ''; // 清空文件选择
        const btn = document.querySelector('button[onclick="uploadPreferenceFile()"]');
        if (btn) {
            btn.innerText = '解析文件';
            btn.disabled = false;
        }
    }
}

// 监听其他来源选项的变化
document.addEventListener('DOMContentLoaded', function() {
    const radios = document.getElementsByName('file_category');
    for (let i = 0; i < radios.length; i++) {
        radios[i].addEventListener('change', function() {
            const otherInput = document.getElementById('other_source_input');
            if (this.value === '其他来源') {
                otherInput.style.display = 'block';
            } else {
                otherInput.style.display = 'none';
            }
        });
    }
});

function showPreferenceSelectionModal(preferences) {
    const modal = document.getElementById('preference_modal');
    const listContainer = document.getElementById('parsed_preferences_list');
    
    if (!modal || !listContainer) return;
    
    listContainer.innerHTML = preferences.map((pref, index) => `
        <div class="preference-item" style="padding: 10px; border: 1px solid #e0e0e0; border-radius: 4px; margin-bottom: 8px;">
            <label style="display: flex; align-items: center; cursor: pointer; margin-bottom: 8px;">
                <input type="checkbox" 
                       id="parsed_pref_${index}" 
                       value="${index}"
                       style="margin-right: 10px;">
                <span style="flex: 1;">${pref.text}</span>
            </label>
            <div style="display: flex; gap: 15px; margin-left: 25px; margin-top: 8px;">
                <label style="display: flex; align-items: center; cursor: pointer; font-size: 14px;">
                    <input type="radio" 
                           name="pref_type_${index}" 
                           value="like"
                           id="pref_like_${index}"
                           checked
                           style="margin-right: 5px;">
                    <span style="color: #4CAF50;">✓ 喜欢</span>
                </label>
                <label style="display: flex; align-items: center; cursor: pointer; font-size: 14px;">
                    <input type="radio" 
                           name="pref_type_${index}" 
                           value="dislike"
                           id="pref_dislike_${index}"
                           style="margin-right: 5px;">
                    <span style="color: #f44336;">✗ 不喜欢</span>
                </label>
            </div>
        </div>
    `).join('');
    
    modal.style.display = 'block';
}

function closePreferenceModal() {
    const modal = document.getElementById('preference_modal');
    if (modal) modal.style.display = 'none';
}

async function addSelectedPreferences() {
    const checkboxes = document.querySelectorAll('#parsed_preferences_list input[type="checkbox"]:checked');
    const selectedIndices = Array.from(checkboxes).map(cb => parseInt(cb.value));
    
    if (selectedIndices.length === 0) {
        alert('请至少选择一个偏好');
        return;
    }
    
    try {
        for (const index of selectedIndices) {
            const pref = parsedPreferencesCache[index];
            // 获取用户选择的偏好类型（喜欢或不喜欢）
            const typeRadio = document.querySelector(`input[name="pref_type_${index}"]:checked`);
            const preferenceType = typeRadio ? typeRadio.value : 'like';
            
            const result = await fetchJson(`${API_BASE}/api/preferences/add`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ 
                    text: pref.text, 
                    selected: true,
                    preference_type: preferenceType
                })
            });
            
            if (result && result.success) {
                currentPreferences.push(result.data);
            }
        }
        
        renderPreferencesList();
        closePreferenceModal();
        alert(`成功添加 ${selectedIndices.length} 个偏好`);
    } catch (error) {
        console.error('添加偏好失败:', error);
        alert('添加失败: ' + error.message);
    }
}

async function addCustomPreference() {
    const input = document.getElementById('custom_preference_input');
    if (!input) return;
    
    const text = input.value.trim();
    if (!text) {
        alert('请输入偏好内容');
        return;
    }
    
    // 获取用户选择的偏好类型
    const typeRadio = document.querySelector('input[name="custom_pref_type"]:checked');
    const preferenceType = typeRadio ? typeRadio.value : 'like';
    
    try {
        const result = await fetchJson(`${API_BASE}/api/preferences/add`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                text: text, 
                selected: true,
                preference_type: preferenceType
            })
        });
        
        if (!result) return;
        if (result.success) {
            currentPreferences.push(result.data);
            renderPreferencesList();
            input.value = ''; // 清空输入框
            // 重置为默认"喜欢"
            const likeRadio = document.getElementById('custom_pref_like');
            if (likeRadio) likeRadio.checked = true;
            alert('偏好已添加');
        } else {
            alert('添加失败: ' + result.message);
        }
    } catch (error) {
        console.error('添加偏好失败:', error);
        alert('添加失败: ' + error.message);
    }
}

async function togglePreference(prefId) {
    try {
        const result = await fetchJson(`${API_BASE}/api/preferences/${prefId}/toggle`, {
            method: 'PUT'
        });
        
        if (!result) return;
        if (result.success) {
            currentPreferences = result.data;
            renderPreferencesList();
        } else {
            alert('更新失败: ' + result.message);
        }
    } catch (error) {
        console.error('更新偏好失败:', error);
        alert('更新失败: ' + error.message);
    }
}

async function togglePreferenceType(prefId) {
    try {
        const result = await fetchJson(`${API_BASE}/api/preferences/${prefId}/toggle-type`, {
            method: 'PUT'
        });
        
        if (!result) return;
        if (result.success) {
            currentPreferences = result.data;
            renderPreferencesList();
        } else {
            alert('更新失败: ' + result.message);
        }
    } catch (error) {
        console.error('切换偏好类型失败:', error);
        alert('切换失败: ' + error.message);
    }
}

async function deletePreference(prefId) {
    if (!confirm('确定要删除这个个人偏好吗？')) {
        return;
    }
    
    try {
        const result = await fetchJson(`${API_BASE}/api/preferences/${prefId}`, {
            method: 'DELETE'
        });
        
        if (!result) return;
        if (result.success) {
            currentPreferences = currentPreferences.filter(p => p.id !== prefId);
            renderPreferencesList();
            alert('偏好已删除');
        } else {
            alert('删除失败: ' + result.message);
        }
    } catch (error) {
        console.error('删除偏好失败:', error);
        alert('删除失败: ' + error.message);
    }
}

// ========== 意图判断历史相关 ==========
async function loadIntentHistory() {
    try {
        const [categoriesResult, historyResult] = await Promise.all([
            fetchJson(`${API_BASE}/api/intent-categories`),
            fetchJson(`${API_BASE}/api/intent-history`)
        ]);

        if (!categoriesResult || !historyResult) return;
        if (categoriesResult.success && historyResult.success) {
            intentCategories = (categoriesResult.data || []).slice().sort((a, b) => (a.order || 0) - (b.order || 0));
            intentHistory = historyResult.data || [];
            renderIntentHistory(intentCategories, intentHistory);
            populateIntentSelect();
        }
    } catch (error) {
        console.error('加载意图历史失败:', error);
        alert('加载意图历史失败: ' + error.message);
    }
}

function renderIntentHistory(categories, history) {
    const container = document.getElementById('intent_history_list');
    if (!container) {
        console.warn('intent_history_list 元素不存在，可能不在当前页面');
        return;
    }
    if (!categories || categories.length === 0) {
        container.innerHTML = '<p style="color: #999; text-align: center; padding: 20px;">暂无意图类别配置</p>';
        return;
    }

    const historyByIntent = {};
    (history || []).forEach(item => {
        const key = item.intent || '未分类';
        if (!historyByIntent[key]) {
            historyByIntent[key] = [];
        }
        historyByIntent[key].push(item);
    });

    container.innerHTML = categories.map(category => {
        const items = historyByIntent[category.name] || [];

        const historyHtml = items.length === 0
            ? '<div style="color: #999;">当前还没有该意图的判断记录</div>'
            : `
                <div class="intent-history-list">
                    ${items.map(item => `
                        <div class="intent-history-item">
                            <div><strong>情景：</strong>${item.context || ''}</div>
                            <div><strong>意图解释：</strong>${item.intent_explanation || ''}</div>
                            <div class="intent-history-item-actions">
                                <button class="btn btn-small btn-primary" onclick="editIntent('${item.id}')">编辑</button>
                                <button class="btn btn-small btn-secondary" onclick="deleteIntent('${item.id}')">删除</button>
                            </div>
                        </div>
                    `).join('')}
                </div>
            `;

        return `
        <div class="intent-item">
            <div class="intent-item-header" onclick="toggleIntentCategory('${category.id}')">
                <div class="intent-item-title">${category.name}</div>
                <div id="intent_toggle_${category.id}" class="intent-item-toggle-indicator">▼</div>
            </div>
            <div class="intent-item-content">
                <div style="margin-bottom: 8px;"><strong>通用解释：</strong>${category.generic_explanation || ''}</div>
                <div id="intent_history_panel_${category.id}" class="intent-item-history-panel" style="display: none;">
                    ${historyHtml}
                </div>
            </div>
        </div>
        `;
    }).join('');
}

function toggleIntentCategory(categoryId) {
    const panelId = `intent_history_panel_${categoryId}`;
    const toggleId = `intent_toggle_${categoryId}`;
    const panel = document.getElementById(panelId);
    const toggle = document.getElementById(toggleId);
    if (!panel || !toggle) {
        return;
    }

    const isHidden = panel.style.display === 'none' || panel.style.display === '';
    panel.style.display = isHidden ? 'block' : 'none';
    toggle.textContent = isHidden ? '▲' : '▼';
}

function populateIntentSelect() {
    const select = document.getElementById('modal_intent_select');
    if (!select) return;

    const currentValue = select.value;
    select.innerHTML = '<option value="">请选择意图类别</option>';

    intentCategories.forEach(category => {
        const option = document.createElement('option');
        option.value = category.name;
        option.textContent = category.name;
        select.appendChild(option);
    });

    if (currentValue) {
        select.value = currentValue;
    }
}

function showAddIntentForm() {
    currentEditIntentId = null;
    document.getElementById('modal_title').textContent = '添加意图判断';
    const contextInput = document.getElementById('modal_context');
    if (contextInput) contextInput.value = '';
    const explanationInput = document.getElementById('modal_intent_explanation');
    if (explanationInput) explanationInput.value = '';
    const newIntentNameInput = document.getElementById('new_intent_name');
    if (newIntentNameInput) newIntentNameInput.value = '';
    const newIntentGenericInput = document.getElementById('new_intent_generic_explanation');
    if (newIntentGenericInput) newIntentGenericInput.value = '';
    populateIntentSelect();
    const intentSelect = document.getElementById('modal_intent_select');
    if (intentSelect) intentSelect.value = '';
    document.getElementById('intent_modal').style.display = 'block';
}

async function editIntent(intentId) {
    currentEditIntentId = intentId;

    let item = intentHistory.find(h => String(h.id) === String(intentId));
    if (!item) {
        try {
            const result = await fetchJson(`${API_BASE}/api/intent-history`);
            if (!result || !result.success) return;
            intentHistory = result.data || [];
            item = intentHistory.find(h => String(h.id) === String(intentId));
        } catch (error) {
            console.error('加载意图详情失败:', error);
            alert('加载失败: ' + error.message);
            return;
        }
    }

    if (!item) {
        alert('未找到对应的意图记录');
        return;
    }

    document.getElementById('modal_title').textContent = '编辑意图判断';
    const contextInput = document.getElementById('modal_context');
    if (contextInput) contextInput.value = item.context || '';
    const explanationInput = document.getElementById('modal_intent_explanation');
    if (explanationInput) explanationInput.value = item.intent_explanation || '';
    populateIntentSelect();
    const intentSelect = document.getElementById('modal_intent_select');
    if (intentSelect) intentSelect.value = item.intent || '';
    const newIntentNameInput = document.getElementById('new_intent_name');
    if (newIntentNameInput) newIntentNameInput.value = '';
    const newIntentGenericInput = document.getElementById('new_intent_generic_explanation');
    if (newIntentGenericInput) newIntentGenericInput.value = '';
    document.getElementById('intent_modal').style.display = 'block';
}

function closeIntentModal() {
    document.getElementById('intent_modal').style.display = 'none';
    currentEditIntentId = null;
}

async function saveIntent() {
    const contextInput = document.getElementById('modal_context');
    const intentSelect = document.getElementById('modal_intent_select');
    const explanationInput = document.getElementById('modal_intent_explanation');

    const data = {
        context: contextInput ? contextInput.value.trim() : '',
        intent: intentSelect ? intentSelect.value : '',
        intent_explanation: explanationInput ? explanationInput.value.trim() : ''
    };

    if (!data.context || !data.intent) {
        alert('请填写当前情景并选择意图类别');
        return;
    }
    
    try {
        let result;
        if (currentEditIntentId) {
            // 更新
            result = await fetchJson(`${API_BASE}/api/intent-history/${currentEditIntentId}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });
        } else {
            // 新增
            result = await fetchJson(`${API_BASE}/api/intent-history`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });
        }
        
        if (!result) return;
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
    if (!confirm('删除后将无法恢复，确认继续删除？')) {
        return;
    }
    
    try {
        const result = await fetchJson(`${API_BASE}/api/intent-history/${intentId}`, {
            method: 'DELETE'
        });
        if (!result) return;
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

// ========== 保存意图对话框相关 ==========
function showSaveIntentModal(context, intent, intentExplanation) {
    const modal = document.getElementById('save_intent_modal');
    const contextTextarea = document.getElementById('save_intent_context');
    const categorySelect = document.getElementById('save_intent_category');
    const explanationTextarea = document.getElementById('save_intent_explanation');
    
    if (!modal || !contextTextarea || !categorySelect || !explanationTextarea) {
        console.error('保存意图模态框元素未找到');
        return;
    }
    
    // 填充表单
    contextTextarea.value = context || '';
    explanationTextarea.value = intentExplanation || '';
    
    // 填充意图类别下拉框
    populateSaveIntentCategories();
    
    // 设置当前选中的意图
    if (intent) {
        categorySelect.value = intent;
    }
    
    // 显示模态框
    modal.style.display = 'block';
}

function populateSaveIntentCategories() {
    const select = document.getElementById('save_intent_category');
    if (!select) return;
    
    // 清空现有选项
    select.innerHTML = '<option value="">请选择意图类别</option>';
    
    // 添加所有意图类别
    intentCategories.forEach(category => {
        const option = document.createElement('option');
        option.value = category.name;
        option.textContent = category.name;
        select.appendChild(option);
    });
}

function closeSaveIntentModal() {
    const modal = document.getElementById('save_intent_modal');
    if (modal) modal.style.display = 'none';
}

async function confirmSaveIntent() {
    const contextTextarea = document.getElementById('save_intent_context');
    const categorySelect = document.getElementById('save_intent_category');
    const explanationTextarea = document.getElementById('save_intent_explanation');
    
    const context = contextTextarea ? contextTextarea.value.trim() : '';
    const intent = categorySelect ? categorySelect.value : '';
    const intentExplanation = explanationTextarea ? explanationTextarea.value.trim() : '';
    
    // 验证必填项
    if (!context || !intent) {
        alert('请填写当前情景并选择意图类别');
        return;
    }
    
    // 构建数据
    const data = {
        context: context,
        intent: intent,
        intent_explanation: intentExplanation,
        user: '当前用户'
    };
    
    try {
        const result = await fetchJson(`${API_BASE}/api/intent-history`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        if (!result) return;
        
        if (result.success) {
            alert('保存成功！');
            closeSaveIntentModal();
            // 刷新意图历史列表
            loadIntentHistory();
        } else {
            alert('保存失败: ' + result.message);
        }
    } catch (error) {
        console.error('保存意图失败:', error);
        alert('保存失败: ' + error.message);
    }
}

// ========== 自动保存意图相关 ==========
async function autoSaveIntent(context, intent, intentExplanation) {
    const saveStatus = document.getElementById('save_status');
    
    if (!context || !intent) {
        if (saveStatus) {
            saveStatus.textContent = '保存失败：数据不完整';
            saveStatus.style.color = '#dc3545';
        }
        return;
    }
    
    // 构建数据
    const data = {
        context: context,
        intent: intent,
        intent_explanation: intentExplanation || '',
        user: '当前用户'
    };
    
    try {
        const result = await fetchJson(`${API_BASE}/api/intent-history`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        if (!result) {
            if (saveStatus) {
                saveStatus.textContent = '保存失败';
                saveStatus.style.color = '#dc3545';
            }
            return;
        }
        
        if (result.success) {
            // 保存成功，记录ID
            lastSavedIntentId = result.data.id;
            if (saveStatus) {
                saveStatus.textContent = '✓ 已保存';
                saveStatus.style.color = '#28a745';
            }
            // 刷新意图历史列表
            loadIntentHistory();
        } else {
            if (saveStatus) {
                saveStatus.textContent = '保存失败: ' + result.message;
                saveStatus.style.color = '#dc3545';
            }
        }
    } catch (error) {
        console.error('自动保存意图失败:', error);
        if (saveStatus) {
            saveStatus.textContent = '保存失败';
            saveStatus.style.color = '#dc3545';
        }
    }
}

function showModifyIntentModal() {
    if (!lastSavedIntentId) {
        alert('没有找到已保存的意图记录');
        return;
    }
    
    // 使用已保存的意图ID进行编辑
    editIntent(lastSavedIntentId);
}

async function addNewIntentCategory() {
    const nameInput = document.getElementById('new_intent_name');
    const genericInput = document.getElementById('new_intent_generic_explanation');
    if (!nameInput || !genericInput) return;

    const name = nameInput.value.trim();
    const genericExplanation = genericInput.value.trim();

    if (!name || !genericExplanation) {
        alert('请填写意图名称和通用解释');
        return;
    }

    try {
        const result = await fetchJson(`${API_BASE}/api/intent-categories`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                name: name,
                generic_explanation: genericExplanation
            })
        });

        if (!result) return;
        if (result.success) {
            intentCategories.push(result.data);
            intentCategories.sort((a, b) => (a.order || 0) - (b.order || 0));
            populateIntentSelect();
            const intentSelect = document.getElementById('modal_intent_select');
            if (intentSelect) {
                intentSelect.value = result.data.name;
            }
            alert('新意图类别已添加');
        } else {
            alert('添加失败: ' + (result.message || '未知错误'));
        }
    } catch (error) {
        console.error('添加意图类别失败:', error);
        alert('添加意图类别失败: ' + error.message);
    }
}

// ========== 生成个性化描述 ==========
async function generateDescription() {
    const customScenario = document.getElementById('custom_scenario').value.trim();
    
    const context = customScenario;
    
    if (!context) {
        alert('输入一个搜索场景');
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
        const result = await fetchJson(`${API_BASE}/api/generate-description`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ context })
        });
        if (!result) return;
        if (result.success) {
            const data = result.data;
            
            // 构建相似历史记录的HTML（带相似度分数）
            let similarHistoriesHTML = '';
            if (data.similar_histories && data.similar_histories.length > 0) {
                similarHistoriesHTML = `
                    <div class="result-item" style="background: #f0f8ff; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                        <strong style="color: #667eea; font-size: 16px;">📚 最相似的历史记录（Top 1）</strong>
                        <div style="font-size: 12px; color: #666; margin: 5px 0 15px 0;">
                            通过 Sentence Embedding 情景匹配（对全部历史计算相似度 → 选出 Top1）
                        </div>
                        ${data.similar_histories.map((h, idx) => `
                            <div style="background: white; padding: 15px; margin: 10px 0; border-left: 4px solid #667eea; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                    <strong style="color: #667eea; font-size: 14px;">📝 历史记录 ${idx + 1}</strong>
                                    ${h.similarity_score !== undefined ? 
                                        `<span style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 4px 12px; border-radius: 16px; font-size: 12px; font-weight: 600;">
                                            相似度: ${(h.similarity_score * 100).toFixed(1)}%
                                        </span>` : ''}
                                </div>
                                <div style="margin: 8px 0; line-height: 1.6;">
                                    <strong style="color: #555;">情景：</strong>
                                    <span style="color: #333;">${h.context || '无'}</span>
                                </div>
                                <div style="margin: 8px 0; line-height: 1.6;">
                                    <strong style="color: #555;">意图：</strong>
                                    <span style="color: #333;">${h.intent || '无'}</span>
                                </div>
                                <div style="margin: 8px 0; line-height: 1.6;">
                                    <strong style="color: #555;">意图解释：</strong>
                                    <span style="color: #333;">${h.intent_explanation || '无'}</span>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                `;
            } else {
                similarHistoriesHTML = `
                    <div class="result-item" style="background: #fff3cd; padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #ffc107;">
                        <strong style="color: #856404;">ℹ️ 相似历史记录：</strong>
                        <div style="color: #856404; margin-top: 5px;">暂无相似的历史记录，系统将基于意图类别直接生成</div>
                    </div>
                `;
            }

            let rankingHTML = '';
            if (data.similarity_rankings && data.similarity_rankings.length > 0) {
                const topRankingItems = data.similarity_rankings.slice(0, 5);
                rankingHTML = `
                    <div class="result-item" style="background: #f8f9ff; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                        <strong style="color: #4c51bf; font-size: 16px;">📊 相似度可视化（前5条）</strong>
                        <div style="margin-top: 10px;">
                            ${topRankingItems
                                .map(
                                    (item, idx) => `
                                        <div style="margin: 12px 0;">
                                            <div style="display: flex; justify-content: space-between; font-size: 12px; color: #555;">
                                                <span>Top ${idx + 1} · ${item.context ? item.context.slice(0, 24) + (item.context.length > 24 ? '…' : '') : '无'}</span>
                                                <span>${(item.similarity_score * 100).toFixed(1)}%</span>
                                            </div>
                                            <div style="height: 8px; border-radius: 4px; background: #e2e8f0; margin-top: 6px;">
                                                <div style="height: 100%; border-radius: 4px; background: linear-gradient(90deg, #667eea, #764ba2); width: ${(item.similarity_score * 100).toFixed(1)}%;"></div>
                                            </div>
                                        </div>
                                    `
                                )
                                .join('')}
                        </div>
                    </div>
                `;
            }
            
            // 保存当前的个性化描述数据
            currentPersonalDescription = data;
            
            // 直接设置结果框的完整内容
            resultBox.innerHTML = `
                <h3>生成结果</h3>
                ${similarHistoriesHTML}
                ${rankingHTML}
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
                <div class="result-item" style="background: #fff3cd; padding: 15px; border-radius: 8px; border-left: 4px solid #ffc107;">
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px;">
                        <span style="color: #856404; font-weight: 600;">💾 自动保存提示</span>
                        <span id="save_status" style="color: #28a745; font-size: 14px;">正在保存...</span>
                    </div>
                    <p style="color: #856404; margin: 10px 0; font-size: 14px;">
                        该意图判断已自动保存到右侧意图判断历史中。如需修改，请点击下方按钮。
                    </p>
                    <div style="text-align: center; margin-top: 10px;">
                        <button class="btn btn-secondary" onclick="showModifyIntentModal()" style="padding: 8px 20px;">
                            ✏️ 需要修改
                        </button>
                    </div>
                </div>
                <div class="result-item" style="text-align: center; padding-top: 10px;">
                    <button class="btn btn-primary btn-large" onclick="searchWithPersonalDescription()" style="width: auto; padding: 12px 30px;">
                        🔍 依据个性化描述搜索
                    </button>
                </div>
                <div id="search_result_container" style="display: none; margin-top: 20px; padding: 20px; background: #f0f8ff; border-radius: 8px; border-left: 4px solid #667eea;">
                    <h4 style="color: #667eea; margin-bottom: 15px;">📝 搜索结果</h4>
                    <div id="search_result_content" style="line-height: 1.8; white-space: pre-wrap; word-wrap: break-word;"></div>
                </div>
            `;
            
            // 生成成功后，自动保存意图
            autoSaveIntent(data.context, data.intent, data.intent_explanation);
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

// ========== 依据个性化描述搜索 ==========
async function searchWithPersonalDescription() {
    if (!currentPersonalDescription) {
        alert('请先生成个性化描述');
        return;
    }
    
    const searchResultContainer = document.getElementById('search_result_container');
    const searchResultContent = document.getElementById('search_result_content');
    
    if (!searchResultContainer || !searchResultContent) {
        alert('无法找到搜索结果显示区域');
        return;
    }
    
    // 显示加载状态
    searchResultContainer.style.display = 'block';
    searchResultContent.innerHTML = '<p style="text-align: center; color: #667eea;">正在生成搜索结果，请稍候...</p>';
    
    try {
        const result = await fetchJson(`${API_BASE}/api/search-with-description`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                context: currentPersonalDescription.context,
                intent: currentPersonalDescription.intent,
                intent_explanation: currentPersonalDescription.intent_explanation,
                personal_memory: currentPersonalDescription.personal_memory,
                personal_description: currentPersonalDescription.personal_description
            })
        });
        
        if (!result) return;
        
        if (result.success) {
            const searchContent = result.data.content || '未生成内容';
            const thinking = result.data.thinking || '';
            
            // 显示搜索结果
            let htmlContent = `<div style="background: white; padding: 15px; border-radius: 6px; margin-bottom: 15px;">${searchContent.replace(/\n/g, '<br>')}</div>`;
            
            // 如果有思考过程，添加可折叠的思考过程展示
            if (thinking) {
                htmlContent += `
                    <div style="margin-top: 15px; padding: 10px; background: #f8f9fa; border-radius: 6px; border-left: 3px solid #6c757d;">
                        <div style="cursor: pointer; font-weight: 600; color: #555; margin-bottom: 5px;" onclick="toggleThinking()">
                            💭 思考过程 <span id="thinking_toggle">▼</span>
                        </div>
                        <div id="thinking_content" style="display: none; margin-top: 10px; padding: 10px; background: white; border-radius: 4px; white-space: pre-wrap; font-size: 13px; color: #666; line-height: 1.6;">${thinking}</div>
                    </div>
                `;
            }
            
            searchResultContent.innerHTML = htmlContent;
        } else {
            searchResultContent.innerHTML = `<p style="color: #dc3545;">搜索失败: ${result.message || '未知错误'}</p>`;
        }
    } catch (error) {
        console.error('搜索失败:', error);
        searchResultContent.innerHTML = `<p style="color: #dc3545;">搜索失败: ${error.message || '网络错误或服务器异常'}</p>`;
    }
}

function toggleThinking() {
    const thinkingContent = document.getElementById('thinking_content');
    const thinkingToggle = document.getElementById('thinking_toggle');
    
    if (!thinkingContent || !thinkingToggle) return;
    
    if (thinkingContent.style.display === 'none') {
        thinkingContent.style.display = 'block';
        thinkingToggle.textContent = '▲';
    } else {
        thinkingContent.style.display = 'none';
        thinkingToggle.textContent = '▼';
    }
}

// 点击模态框外部关闭
window.onclick = function(event) {
    const intentModal = document.getElementById('intent_modal');
    if (intentModal && event.target == intentModal) {
        closeIntentModal();
    }
    
    const preferenceModal = document.getElementById('preference_modal');
    if (preferenceModal && event.target == preferenceModal) {
        closePreferenceModal();
    }
    
    const saveIntentModal = document.getElementById('save_intent_modal');
    if (saveIntentModal && event.target == saveIntentModal) {
        closeSaveIntentModal();
    }

    const fileCategoryModal = document.getElementById('file_category_modal');
    if (fileCategoryModal && event.target == fileCategoryModal) {
        closeFileCategoryModal();
    }
}
