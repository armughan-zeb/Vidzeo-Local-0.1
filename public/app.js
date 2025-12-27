// =============================================================================
// VIDZEO LOCAL - Frontend Application
// =============================================================================

const API_BASE = '';

// =============================================================================
// STATE
// =============================================================================

const state = {
    // Images (Single)
    uploadedImages: [],
    generatedImages: [],
    generatedPrompts: [],
    selectedScene: 0,

    // Images (Bulk)
    bulkUploadedImages: [],
    bulkImageSource: 'custom',

    // Music
    musicPath: null,
    bulkMusicPath: null,

    // Queue
    queue: [],

    // Generation
    isGenerating: false,

    // Settings (loaded from localStorage)
    settings: {
        groqApiKey: '',
        togetherApiKey: '',
        pollinationsApiKey: ''
    }
};

// =============================================================================
// SETTINGS MANAGEMENT
// =============================================================================

function loadSettings() {
    try {
        const saved = localStorage.getItem('vidzeo_settings');
        if (saved) {
            state.settings = JSON.parse(saved);
            console.log('✅ Settings loaded from localStorage');
        }
    } catch (e) {
        console.error('Failed to load settings:', e);
    }
}

function saveSettings() {
    const groqKey = document.getElementById('settingsGroqKey').value;
    const togetherKey = document.getElementById('settingsTogetherKey').value;
    const pollinationsKey = document.getElementById('settingsPollinationsKey')?.value || '';

    state.settings.groqApiKey = groqKey;
    state.settings.togetherApiKey = togetherKey;
    state.settings.pollinationsApiKey = pollinationsKey;

    try {
        localStorage.setItem('vidzeo_settings', JSON.stringify(state.settings));
        showToast('Settings saved!', 'success');
        updateApiStatus();
        autoPopulateApiKeys();
    } catch (e) {
        showToast('Failed to save settings', 'error');
    }
}

function clearSettings() {
    if (!confirm('Are you sure you want to clear all saved API keys?')) return;

    state.settings = { groqApiKey: '', togetherApiKey: '', pollinationsApiKey: '' };
    localStorage.removeItem('vidzeo_settings');

    document.getElementById('settingsGroqKey').value = '';
    document.getElementById('settingsTogetherKey').value = '';
    const pollinationsField = document.getElementById('settingsPollinationsKey');
    if (pollinationsField) pollinationsField.value = '';

    updateApiStatus();
    autoPopulateApiKeys();
    showToast('Settings cleared', 'info');
}

function updateApiStatus() {
    const groqStatus = document.getElementById('groqStatus');
    const togetherStatus = document.getElementById('togetherStatus');

    if (state.settings.groqApiKey) {
        groqStatus.textContent = 'Set ✓';
        groqStatus.className = 'api-status set';
    } else {
        groqStatus.textContent = 'Not Set';
        groqStatus.className = 'api-status';
    }

    if (state.settings.togetherApiKey) {
        togetherStatus.textContent = 'Set ✓';
        togetherStatus.className = 'api-status set';
    } else {
        togetherStatus.textContent = 'Not Set';
        togetherStatus.className = 'api-status';
    }

    const pollinationsStatus = document.getElementById('pollinationsStatus');
    if (pollinationsStatus) {
        if (state.settings.pollinationsApiKey) {
            pollinationsStatus.textContent = 'Set ✓';
            pollinationsStatus.className = 'api-status set';
        } else {
            pollinationsStatus.textContent = 'Optional (Free)';
            pollinationsStatus.className = 'api-status';
        }
    }
}

function autoPopulateApiKeys() {
    // Populate Single Generation API key fields with saved values
    // (API keys are now only entered in Settings tab)
    const groqFields = ['#scriptGroqKey'];
    const togetherFields = [];

    groqFields.forEach(selector => {
        const el = document.querySelector(selector);
        if (el && !el.value && state.settings.groqApiKey) {
            el.value = state.settings.groqApiKey;
        }
    });

    togetherFields.forEach(selector => {
        const el = document.querySelector(selector);
        if (el && !el.value && state.settings.togetherApiKey) {
            el.value = state.settings.togetherApiKey;
        }
    });
}

window.togglePassword = function (inputId) {
    const input = document.getElementById(inputId);
    input.type = input.type === 'password' ? 'text' : 'password';
};

async function testConnections() {
    const btn = document.getElementById('testConnectionBtn');
    btn.disabled = true;
    btn.textContent = '⏳ Testing...';

    const groqKey = document.getElementById('settingsGroqKey').value;
    const togetherKey = document.getElementById('settingsTogetherKey').value;

    let groqOk = false;
    let togetherOk = false;

    // Test Groq
    if (groqKey) {
        try {
            const response = await fetch('https://api.groq.com/openai/v1/models', {
                headers: { 'Authorization': `Bearer ${groqKey}` }
            });
            groqOk = response.ok;
            document.getElementById('groqStatus').textContent = groqOk ? 'Connected ✓' : 'Error ✗';
            document.getElementById('groqStatus').className = `api-status ${groqOk ? 'set' : 'error'}`;
        } catch (e) {
            document.getElementById('groqStatus').textContent = 'Error ✗';
            document.getElementById('groqStatus').className = 'api-status error';
        }
    }

    // Test Together AI
    if (togetherKey) {
        try {
            const response = await fetch('https://api.together.xyz/v1/models', {
                headers: { 'Authorization': `Bearer ${togetherKey}` }
            });
            togetherOk = response.ok;
            document.getElementById('togetherStatus').textContent = togetherOk ? 'Connected ✓' : 'Error ✗';
            document.getElementById('togetherStatus').className = `api-status ${togetherOk ? 'set' : 'error'}`;
        } catch (e) {
            document.getElementById('togetherStatus').textContent = 'Error ✗';
            document.getElementById('togetherStatus').className = 'api-status error';
        }
    }

    if (!groqKey && !togetherKey) {
        showToast('Please enter at least one API key to test', 'error');
    } else {
        const results = [];
        if (groqKey) results.push(groqOk ? 'Groq ✓' : 'Groq ✗');
        if (togetherKey) results.push(togetherOk ? 'Together AI ✓' : 'Together AI ✗');
        showToast(`Connection test: ${results.join(', ')}`, groqOk || togetherOk ? 'success' : 'error');
    }

    btn.disabled = false;
    btn.textContent = '🔌 Test Connections';
}

// =============================================================================
// UTILITIES
// =============================================================================

function $(selector) {
    return document.querySelector(selector);
}

function $$(selector) {
    return document.querySelectorAll(selector);
}

function showToast(message, type = 'info') {
    const container = $('#toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
        toast.remove();
    }, 4000);
}

async function api(endpoint, options = {}) {
    try {
        const response = await fetch(`${API_BASE}${endpoint}`, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'API request failed');
        }

        return data;
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

function updateProgress(percent, text) {
    const fill = $('#progressFill');
    const textEl = $('#progressText');

    fill.style.width = `${percent}%`;
    textEl.textContent = text;
}

function showProgress() {
    $('#progressContainer').classList.remove('hidden');
}

function hideProgress() {
    $('#progressContainer').classList.add('hidden');
}

function updateStatus(message) {
    $('#statusBox').textContent = message;
}

// =============================================================================
// INITIALIZATION
// =============================================================================

async function init() {
    console.log('🚀 Vidzeo Local - Initializing...');

    // Load saved settings first
    loadSettings();

    // Populate settings UI with saved values
    if ($('#settingsGroqKey')) {
        $('#settingsGroqKey').value = state.settings.groqApiKey || '';
    }
    if ($('#settingsTogetherKey')) {
        $('#settingsTogetherKey').value = state.settings.togetherApiKey || '';
    }
    updateApiStatus();

    // Check API status
    try {
        const status = await api('/api/status');
        $('#statusTTS').textContent = `🎙️ TTS: ${status.tts ? '✅' : '❌'}`;
        $('#statusTTS').classList.toggle('ready', status.tts);
        $('#statusWhisper').textContent = `📝 Whisper: ${status.whisper ? '✅' : '❌'}`;
        $('#statusWhisper').classList.toggle('ready', status.whisper);
        $('#statusFFmpeg').textContent = `🎬 FFmpeg: ${status.ffmpeg ? '✅' : '❌'}`;
        $('#statusFFmpeg').classList.toggle('ready', status.ffmpeg);
    } catch (e) {
        showToast('Failed to connect to server', 'error');
    }

    // Load voices
    try {
        const voices = await api('/api/voices');
        populateSelect('#voice', voices);
        populateSelect('#bulkVoice', voices);
    } catch (e) {
        console.error('Failed to load voices');
    }

    // Load fonts
    try {
        const fonts = await api('/api/fonts');
        populateSelect('#font', fonts.flat);
        populateSelect('#bulkFont', fonts.flat);
    } catch (e) {
        console.error('Failed to load fonts');
    }

    // Load image styles
    try {
        const styles = await api('/api/styles');
        populateSelect('#imageStyle', styles);
        populateSelect('#bulkImageStyle', styles);
    } catch (e) {
        console.error('Failed to load styles');
    }

    // Setup event listeners
    setupEventListeners();

    // Auto-populate API keys in forms
    autoPopulateApiKeys();

    console.log('✅ Initialized');
}

function populateSelect(selector, options, defaultValue = null) {
    const select = $(selector);
    if (!select) return;

    select.innerHTML = '';
    options.forEach(opt => {
        const option = document.createElement('option');
        option.value = opt;
        option.textContent = opt;
        if (opt === defaultValue) option.selected = true;
        select.appendChild(option);
    });
}

// =============================================================================
// EVENT LISTENERS
// =============================================================================

function setupEventListeners() {
    // Tabs
    $$('.tab').forEach(tab => {
        tab.addEventListener('click', () => {
            $$('.tab').forEach(t => t.classList.remove('active'));
            $$('.tab-content').forEach(c => c.classList.remove('active'));
            tab.classList.add('active');
            $(`#tab-${tab.dataset.tab}`).classList.add('active');
        });
    });

    // Script Mode Toggle
    $$('#tab-single .toggle-btn[data-mode]').forEach(btn => {
        btn.addEventListener('click', () => {
            $$('#tab-single .toggle-btn[data-mode]').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            $$('.script-mode').forEach(m => m.classList.add('hidden'));
            $(`#mode-${btn.dataset.mode}`).classList.remove('hidden');
        });
    });

    // Image Source Toggle
    $$('#tab-single .toggle-btn[data-source]').forEach(btn => {
        btn.addEventListener('click', () => {
            $$('#tab-single .toggle-btn[data-source]').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            $$('.image-source').forEach(m => m.classList.add('hidden'));
            $(`#source-${btn.dataset.source}`).classList.remove('hidden');
        });
    });

    // Scene Mode Change
    $('#sceneMode').addEventListener('change', (e) => {
        $('#customSceneCount').classList.toggle('hidden', e.target.value !== 'Custom');
    });

    // Captions Toggle
    $('#captionsEnabled').addEventListener('change', (e) => {
        $('#captionSettings').classList.toggle('hidden', !e.target.checked);
    });

    // Range Sliders
    setupRangeSlider('#wordsPerGroup', '#wordsPerGroupValue');
    setupRangeSlider('#fontsize', '#fontsizeValue');
    setupRangeSlider('#outlineSize', '#outlineSizeValue');
    setupRangeSlider('#shadowDepth', '#shadowDepthValue');
    setupRangeSlider('#bgOpacity', '#bgOpacityValue');
    setupRangeSlider('#marginV', '#marginVValue');
    setupRangeSlider('#musicVolume', '#musicVolumeValue', '%');

    // Image Upload
    $('#imageFiles').addEventListener('change', handleImageUpload);

    // Music Upload
    $('#musicFile').addEventListener('change', handleMusicUpload);

    // Voice Preview
    $('#previewVoiceBtn').addEventListener('click', previewVoice);


    // Scene Preview
    $('#previewScenesBtn').addEventListener('click', previewScenes);
    $('#clearScenesBtn').addEventListener('click', clearScenes);
    $('#regenSceneBtn').addEventListener('click', regenerateScene);

    // Generate Video
    $('#generateBtn').addEventListener('click', generateVideo);

    // Bulk Queue
    $('#addToQueueBtn').addEventListener('click', addToQueue);
    $('#clearQueueBtn').addEventListener('click', clearQueue);
    $('#processQueueBtn').addEventListener('click', processQueue);
    $('#downloadAllBtn').addEventListener('click', downloadAll);

    // Bulk Script Mode Toggle (Write / Generate from Title)
    $$('#bulkScriptToggle .toggle-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            $$('#bulkScriptToggle .toggle-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            const mode = btn.dataset.mode;
            $('#bulk-mode-write').classList.toggle('hidden', mode !== 'write');
            $('#bulk-mode-generate').classList.toggle('hidden', mode !== 'generate');
        });
    });

    // Bulk Range Sliders
    setupRangeSlider('#bulkWordsPerGroup', '#bulkWordsPerGroupValue');
    setupRangeSlider('#bulkFontsize', '#bulkFontsizeValue');
    setupRangeSlider('#bulkOutlineSize', '#bulkOutlineSizeValue');
    setupRangeSlider('#bulkShadowDepth', '#bulkShadowDepthValue');
    setupRangeSlider('#bulkBgOpacity', '#bulkBgOpacityValue');
    setupRangeSlider('#bulkMarginV', '#bulkMarginVValue');
    setupRangeSlider('#bulkMusicVolume', '#bulkMusicVolumeValue', '%');

    // Bulk Music Upload
    if ($('#bulkMusicFile')) {
        $('#bulkMusicFile').addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file) return;

            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch('/api/upload/music', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                if (data.success) {
                    state.bulkMusicPath = data.path;
                    $('#bulkMusicFileName').textContent = `📁 ${file.name}`;
                    showToast('Bulk music uploaded', 'success');
                }
            } catch (error) {
                showToast('Failed to upload music', 'error');
            }
        });
    }

    // Bulk Image Source Toggle
    $$('#bulkImageToggle .toggle-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            $$('#bulkImageToggle .toggle-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            const source = btn.dataset.source;
            state.bulkImageSource = source;

            $('#bulk-source-custom').classList.toggle('hidden', source !== 'custom');
            $('#bulk-source-ai').classList.toggle('hidden', source !== 'ai');
        });
    });

    // Bulk Image Upload
    if ($('#bulkImageFiles')) {
        $('#bulkImageFiles').addEventListener('change', async (e) => {
            const files = e.target.files;
            if (!files.length) return;

            const formData = new FormData();
            for (const file of files) {
                formData.append('files', file);
            }

            try {
                const response = await fetch('/api/upload/images', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                if (data.success) {
                    state.bulkUploadedImages = data.paths;
                    // Render preview
                    const container = $('#bulkImagePreview');
                    container.innerHTML = '';
                    data.urls.forEach(url => {
                        const img = document.createElement('img');
                        img.src = url;
                        container.appendChild(img);
                    });
                    showToast(`Uploaded ${data.paths.length} images for bulk`, 'success');
                }
            } catch (error) {
                showToast('Failed to upload images', 'error');
            }
        });
    }

    // Settings
    $('#saveSettingsBtn').addEventListener('click', saveSettings);
    $('#testConnectionBtn').addEventListener('click', testConnections);
    $('#clearSettingsBtn').addEventListener('click', clearSettings);
}

function setupRangeSlider(inputSelector, valueSelector, suffix = '') {
    const input = $(inputSelector);
    const value = $(valueSelector);
    if (!input || !value) return;

    input.addEventListener('input', () => {
        value.textContent = input.value + suffix;
    });
}

// =============================================================================
// IMAGE HANDLING
// =============================================================================

async function handleImageUpload(e) {
    const files = e.target.files;
    if (!files.length) return;

    const formData = new FormData();
    for (const file of files) {
        formData.append('files', file);
    }

    try {
        const response = await fetch('/api/upload/images', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            state.uploadedImages = data.paths;
            renderImagePreview(data.urls);
            showToast(`Uploaded ${data.paths.length} images`, 'success');
        }
    } catch (error) {
        showToast('Failed to upload images', 'error');
    }
}

function renderImagePreview(urls) {
    const container = $('#imagePreview');
    container.innerHTML = '';

    urls.forEach(url => {
        const img = document.createElement('img');
        img.src = url;
        container.appendChild(img);
    });
}

// =============================================================================
// MUSIC HANDLING
// =============================================================================

async function handleMusicUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/api/upload/music', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            state.musicPath = data.path;
            $('#musicFileName').textContent = `📁 ${file.name}`;
            showToast('Music uploaded', 'success');
        }
    } catch (error) {
        showToast('Failed to upload music', 'error');
    }
}

// =============================================================================
// VOICE PREVIEW
// =============================================================================

async function previewVoice() {
    const voice = $('#voice').value;
    const btn = $('#previewVoiceBtn');

    btn.disabled = true;
    btn.textContent = '⏳';

    try {
        const data = await api('/api/preview-voice', {
            method: 'POST',
            body: JSON.stringify({ voice })
        });

        const audio = $('#voicePreview');
        audio.src = data.audio;
        audio.play();
    } catch (error) {
        showToast('Failed to preview voice', 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = '▶️';
    }
}

// =============================================================================
// SCRIPT GENERATION
// =============================================================================

async function generateScript() {
    const title = $('#scriptTitle').value;
    const duration = $('#scriptDuration').value;
    const style = $('#scriptStyle').value;
    const groqKey = $('#scriptGroqKey').value;

    if (!title) {
        showToast('Please enter a video title', 'error');
        return;
    }

    if (!groqKey) {
        showToast('Please enter your Groq API key', 'error');
        return;
    }

    const btn = $('#generateScriptBtn');
    btn.disabled = true;
    btn.textContent = '⏳ Generating...';

    try {
        const data = await api('/api/generate-script', {
            method: 'POST',
            body: JSON.stringify({
                title,
                duration,
                style,
                groq_api_key: groqKey
            })
        });

        $('#script').value = data.script;

        // Switch to write mode to show the script
        $$('#tab-single .toggle-btn[data-mode]').forEach(b => b.classList.remove('active'));
        $('[data-mode="write"]').classList.add('active');
        $$('.script-mode').forEach(m => m.classList.add('hidden'));
        $('#mode-write').classList.remove('hidden');

        showToast('Script generated!', 'success');
    } catch (error) {
        showToast(`Failed: ${error.message}`, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = '✨ Generate Script';
    }
}

async function bulkGenerateScript() {
    const title = $('#bulkScriptTitle').value;
    const duration = $('#bulkScriptDuration').value;
    const style = $('#bulkScriptStyle').value;

    if (!title) {
        showToast('Please enter a video title', 'error');
        return;
    }

    // Use API key from settings
    const groqKey = state.settings.groqApiKey;
    if (!groqKey) {
        showToast('Please set your Groq API key in Settings tab', 'error');
        return;
    }

    const btn = $('#bulkGenerateScriptBtn');
    btn.disabled = true;
    btn.textContent = '⏳ Generating...';

    try {
        const data = await api('/api/generate-script', {
            method: 'POST',
            body: JSON.stringify({
                title,
                duration,
                style,
                groq_api_key: groqKey
            })
        });

        $('#bulkScript').value = data.script;

        // Also set the name from title if empty
        if (!$('#bulkName').value) {
            $('#bulkName').value = title;
        }

        // Switch to write mode to show the script
        $$('#bulkScriptToggle .toggle-btn').forEach(b => b.classList.remove('active'));
        $('#bulkScriptToggle .toggle-btn[data-mode="write"]').classList.add('active');
        $('#bulk-mode-write').classList.remove('hidden');
        $('#bulk-mode-generate').classList.add('hidden');

        showToast('Script generated!', 'success');
    } catch (error) {
        showToast(`Failed: ${error.message}`, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = '✨ Generate Script';
    }
}

// =============================================================================
// SCENE PREVIEW
// =============================================================================

async function previewScenes() {
    const script = $('#script').value;
    const groqKey = $('#groqApiKey').value;
    const togetherKey = $('#togetherApiKey').value;
    const model = $('#imageModel').value;
    const resolution = $('#imageResolution').value;
    const sceneMode = $('#sceneMode').value;
    const customCount = $('#customSceneCount').value;
    const imageStyle = $('#imageStyle').value;

    if (!script) {
        showToast('Please enter a script first', 'error');
        return;
    }

    if (!groqKey || !togetherKey) {
        showToast('Please enter both API keys', 'error');
        return;
    }

    const btn = $('#previewScenesBtn');
    btn.disabled = true;
    btn.textContent = '⏳ Generating...';

    try {
        const data = await api('/api/generate-images', {
            method: 'POST',
            body: JSON.stringify({
                script,
                groq_api_key: groqKey,
                together_api_key: togetherKey,
                model,
                resolution,
                scene_mode: sceneMode,
                custom_count: parseInt(customCount),
                image_style: imageStyle
            })
        });

        state.generatedPrompts = data.prompts;
        state.generatedImages = data.images;

        renderSceneGallery(data.images);
        $('#sceneEditor').classList.remove('hidden');

        showToast(`Generated ${data.images.filter(i => i).length} scenes`, 'success');
    } catch (error) {
        showToast(`Failed: ${error.message}`, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = '🎨 Preview Scenes';
    }
}

function renderSceneGallery(images) {
    const gallery = $('#sceneGallery');
    gallery.innerHTML = '';

    images.forEach((url, index) => {
        if (!url) return;

        const item = document.createElement('div');
        item.className = 'scene-item';
        item.innerHTML = `
            <img src="${url}" alt="Scene ${index + 1}">
            <span class="scene-num">${index + 1}</span>
        `;
        item.addEventListener('click', () => selectScene(index));
        gallery.appendChild(item);
    });
}

function selectScene(index) {
    state.selectedScene = index;

    $$('.scene-item').forEach((item, i) => {
        item.classList.toggle('selected', i === index);
    });

    $('#editSceneNum').value = index + 1;
    $('#editScenePrompt').value = state.generatedPrompts[index] || '';
}

async function regenerateScene() {
    const sceneNum = parseInt($('#editSceneNum').value);
    const prompt = $('#editScenePrompt').value;
    const togetherKey = $('#togetherApiKey').value;
    const model = $('#imageModel').value;
    const resolution = $('#imageResolution').value;
    const imageStyle = $('#imageStyle').value;

    if (!prompt) {
        showToast('Please enter a prompt', 'error');
        return;
    }

    const btn = $('#regenSceneBtn');
    btn.disabled = true;
    btn.textContent = '⏳...';

    try {
        const data = await api('/api/regenerate-image', {
            method: 'POST',
            body: JSON.stringify({
                prompt,
                together_api_key: togetherKey,
                model,
                resolution,
                image_style: imageStyle
            })
        });

        // Update state
        state.generatedImages[sceneNum - 1] = data.image;
        state.generatedPrompts[sceneNum - 1] = prompt;

        renderSceneGallery(state.generatedImages);
        showToast(`Scene ${sceneNum} regenerated`, 'success');
    } catch (error) {
        showToast(`Failed: ${error.message}`, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = '🔄 Regenerate';
    }
}

function clearScenes() {
    state.generatedImages = [];
    state.generatedPrompts = [];
    $('#sceneGallery').innerHTML = '';
    $('#sceneEditor').classList.add('hidden');
}

// =============================================================================
// VIDEO GENERATION
// =============================================================================

async function generateVideo() {
    if (state.isGenerating) return;

    // Check if we're in "Generate from Title" mode
    const isGenerateMode = $('#mode-generate') && !$('#mode-generate').classList.contains('hidden');
    let script = $('#script').value;

    if (isGenerateMode) {
        // Auto-generate script from title
        const title = $('#scriptTitle').value;
        if (!title) {
            showToast('Please enter a video title', 'error');
            return;
        }

        const groqKey = state.settings.groqApiKey;
        if (!groqKey) {
            showToast('Please set your Groq API key in Settings tab', 'error');
            return;
        }

        showToast('Generating script...', 'info');
        try {
            const data = await api('/api/generate-script', {
                method: 'POST',
                body: JSON.stringify({
                    title,
                    duration: $('#scriptDuration').value,
                    style: $('#scriptStyle').value,
                    groq_api_key: groqKey
                })
            });
            script = data.script;
            $('#script').value = script; // Store for display
        } catch (error) {
            showToast(`Script generation failed: ${error.message}`, 'error');
            return;
        }
    }

    if (!script || script.length < 10) {
        showToast('Script is too short (min 10 characters)', 'error');
        return;
    }

    // Determine image source
    const imageSource = $('.toggle-btn[data-source].active').dataset.source;
    let images = [];

    if (imageSource === 'custom') {
        if (!state.uploadedImages.length) {
            showToast('Please upload images', 'error');
            return;
        }
        images = state.uploadedImages;
    } else {
        // AI images - check if already generated
        if (state.generatedImages.length > 0) {
            // Use pre-generated images (convert URLs to paths)
            images = state.generatedImages.map(url => {
                // Server will handle URL-to-path conversion
                return url;
            });
        }
    }

    state.isGenerating = true;
    const btn = $('#generateBtn');
    btn.disabled = true;
    btn.textContent = '⏳ Generating...';

    showProgress();
    updateProgress(5, 'Starting...');
    $('#videoPlaceholder').classList.remove('hidden');

    try {
        // Build request
        const requestData = {
            script,
            voice: $('#voice').value,
            name: $('#videoName').value,
            effect: $('#effect').value,

            // Image settings
            image_source: imageSource === 'ai' ? 'AI Generated' : 'Custom Images',
            images: imageSource === 'custom' ? state.uploadedImages : [],

            // AI image settings (from Settings tab)
            groq_api_key: state.settings.groqApiKey,
            together_api_key: state.settings.togetherApiKey,
            image_provider: $('#imageProvider')?.value || 'Pollinations AI (Free)',
            pollinations_api_key: state.settings.pollinationsApiKey || '',
            image_model: $('#imageModel').value,
            image_resolution: $('#imageResolution').value,
            scene_mode: $('#sceneMode').value,
            custom_count: parseInt($('#customSceneCount').value),
            image_style: $('#imageStyle').value,

            // Caption settings
            captions_enabled: $('#captionsEnabled').checked,
            caption_mode: $('input[name="captionMode"]:checked').value,
            words_per_group: parseInt($('#wordsPerGroup').value),
            font: $('#font').value,
            fontsize: parseInt($('#fontsize').value),
            bold: $('#bold').checked,
            uppercase: $('#uppercase').checked,
            text_color: $('#textColor').value,
            highlight_color: $('#highlightColor').value,
            outline_color: $('#outlineColor').value,
            outline_size: parseInt($('#outlineSize').value),
            shadow_on: $('#shadowOn').checked,
            shadow_color: $('#shadowColor').value,
            shadow_depth: parseInt($('#shadowDepth').value),
            bg_on: $('#bgOn').checked,
            bg_color: $('#bgColor').value,
            bg_opacity: parseInt($('#bgOpacity').value),
            position: $('#position').value,
            margin_v: parseInt($('#marginV').value),
            animation: $('#animation').value,

            // Music
            music_path: state.musicPath,
            music_volume: parseInt($('#musicVolume').value)
        };

        updateProgress(10, 'Sending request...');

        const data = await api('/api/generate', {
            method: 'POST',
            body: JSON.stringify(requestData)
        });

        updateProgress(100, 'Complete!');

        if (data.success) {
            // Show video
            const video = $('#videoPreview');
            video.src = data.video;
            $('#videoPlaceholder').classList.add('hidden');

            updateStatus(`✅ Video generated!\n📁 ${data.video}\n⏱️ ${data.duration.toFixed(1)}s`);
            showToast('Video generated successfully!', 'success');
        }
    } catch (error) {
        updateStatus(`❌ Error: ${error.message}`);
        showToast(`Generation failed: ${error.message}`, 'error');
    } finally {
        state.isGenerating = false;
        btn.disabled = false;
        btn.textContent = '🚀 Generate Video';
        hideProgress();
    }
}

// =============================================================================
// BULK QUEUE
// =============================================================================

function addToQueue() {
    let name = $('#bulkName').value || `Video_${state.queue.length + 1}`;
    let script = $('#bulkScript').value;
    const voice = $('#bulkVoice').value;

    // Check if we're in "Generate from Title" mode
    const isGenerateMode = $('#bulk-mode-generate') && !$('#bulk-mode-generate').classList.contains('hidden');

    // For storing title info for deferred generation
    let generateFromTitle = null;

    if (isGenerateMode) {
        // Store title info for later generation during processQueue
        const title = $('#bulkScriptTitle').value;
        if (!title) {
            showToast('Please enter a video title', 'error');
            return;
        }

        // Store title/duration/style - script will be generated during processing
        generateFromTitle = {
            title,
            duration: $('#bulkScriptDuration').value,
            style: $('#bulkScriptStyle').value
        };
        name = title;
        script = null; // Mark as needing generation
    } else {
        // Write Script mode - require actual script
        if (!script || script.length < 10) {
            showToast('Script is too short', 'error');
            return;
        }
    }

    // Snapshot current BULK settings
    const settings = {
        // AI Image Settings (from Settings tab)
        groq_api_key: state.settings.groqApiKey,
        together_api_key: state.settings.togetherApiKey,
        image_model: $('#bulkImageModel').value,
        image_resolution: $('#bulkImageResolution').value,
        scene_mode: $('#bulkSceneMode').value,
        custom_count: 30,
        image_style: $('#bulkImageStyle').value,

        // Caption Settings
        captions_enabled: $('#bulkCaptionsEnabled').checked,
        caption_mode: $('input[name="bulkCaptionMode"]:checked')?.value || 'single',
        words_per_group: parseInt($('#bulkWordsPerGroup').value),
        font: $('#bulkFont').value,
        fontsize: parseInt($('#bulkFontsize').value),
        bold: $('#bulkBold').checked,
        uppercase: $('#bulkUppercase').checked,
        text_color: $('#bulkTextColor').value,
        highlight_color: $('#bulkHighlightColor').value,
        outline_color: $('#bulkOutlineColor').value,
        outline_size: parseInt($('#bulkOutlineSize').value),
        shadow_on: $('#bulkShadowOn').checked,
        shadow_color: $('#bulkShadowColor').value,
        shadow_depth: parseInt($('#bulkShadowDepth').value),
        bg_on: $('#bulkBgOn').checked,
        bg_color: $('#bulkBgColor').value,
        bg_opacity: parseInt($('#bulkBgOpacity').value),
        position: $('#bulkPosition').value,
        margin_v: parseInt($('#bulkMarginV').value),
        animation: $('#bulkAnimation').value,

        // Effect
        effect: $('#bulkEffect').value,

        // Music
        music_path: state.bulkMusicPath || null,
        music_volume: parseInt($('#bulkMusicVolume').value),

        // Image Source (capture bulk selection)
        image_source: state.bulkImageSource === 'ai' ? 'AI Generated' : 'Custom Images',
        images: state.bulkImageSource === 'custom' ? [...state.bulkUploadedImages] : []
    };

    state.queue.push({
        id: Date.now(),
        name,
        script,
        voice,
        settings, // Store snapshot
        generateFromTitle, // Store title info for deferred generation (null if writing script)
        status: 'queued'
    });

    renderQueue();
    showToast(`Added "${name}" to queue`, 'success');

    // Clear inputs
    $('#bulkName').value = '';
    $('#bulkScript').value = '';
    $('#bulkScriptTitle').value = '';
    $('#bulkScriptStyle').value = '';
}

function clearQueue() {
    state.queue = [];
    renderQueue();
    showToast('Queue cleared', 'info');
}

function renderQueue() {
    const container = $('#queueList');
    const count = $('#queueCount');

    count.textContent = state.queue.length;

    if (state.queue.length === 0) {
        container.innerHTML = '<div class="queue-empty">Queue is empty</div>';
        return;
    }

    container.innerHTML = state.queue.map((item, index) => `
        <div class="queue-item">
            <div class="queue-item-info">
                <div class="queue-item-name">${index + 1}. ${item.name}</div>
                <div class="queue-item-status ${item.status}">${getStatusText(item.status)}</div>
            </div>
            <button class="queue-item-delete" onclick="removeFromQueue(${item.id})">🗑️</button>
        </div>
    `).join('');
}

function getStatusText(status) {
    const texts = {
        queued: '⏳ Queued',
        processing: '🔄 Processing...',
        done: '✅ Complete',
        error: '❌ Error'
    };
    return texts[status] || status;
}

window.removeFromQueue = function (id) {
    state.queue = state.queue.filter(item => item.id !== id);
    renderQueue();
};

async function processQueue() {
    if (state.queue.length === 0) {
        showToast('Queue is empty', 'error');
        return;
    }

    // Check if Groq API key is set (needed for script generation)
    const groqKey = state.settings.groqApiKey;

    // This would need more implementation for proper bulk processing
    showToast('Processing queue...', 'info');

    for (const item of state.queue) {
        if (item.status === 'queued') {
            item.status = 'processing';
            renderQueue();

            try {
                let script = item.script;

                // If this item needs script generation from title
                if (item.generateFromTitle && !script) {
                    if (!groqKey) {
                        throw new Error('Groq API key not set in Settings');
                    }

                    showToast(`Generating script for "${item.name}"...`, 'info');

                    const scriptData = await api('/api/generate-script', {
                        method: 'POST',
                        body: JSON.stringify({
                            title: item.generateFromTitle.title,
                            duration: item.generateFromTitle.duration,
                            style: item.generateFromTitle.style,
                            groq_api_key: groqKey
                        })
                    });

                    script = scriptData.script;
                    item.script = script; // Store generated script
                }

                if (!script || script.length < 10) {
                    throw new Error('Script is too short or missing');
                }

                // Construct request data from saved settings
                const requestData = {
                    script,
                    voice: item.voice,
                    name: item.name,
                    ...item.settings
                };

                const data = await api('/api/generate', {
                    method: 'POST',
                    body: JSON.stringify(requestData)
                });

                if (data.success) {
                    item.status = 'done';
                    item.result = data.video;

                    // Auto-download if enabled
                    if ($('#autoDownloadBulk').checked) {
                        try {
                            const a = document.createElement('a');
                            a.href = data.video;
                            a.download = `${item.name.replace(/[^a-z0-9]/gi, '_')}.mp4`;
                            document.body.appendChild(a);
                            a.click();
                            document.body.removeChild(a);
                            showToast(`Downloading ${item.name}...`, 'success');
                        } catch (e) {
                            console.error('Auto-download failed:', e);
                        }
                    }

                    // Add to gallery
                    const gallery = $('#bulkVideoGallery');
                    const vidDiv = document.createElement('div');
                    vidDiv.className = 'bulk-video-item';
                    vidDiv.innerHTML = `
                        <h4>${item.name}</h4>
                        <video src="${data.video}" controls width="200"></video>
                        <a href="${data.video}" download class="btn btn-sm">⬇️ Download</a>
                    `;
                    gallery.prepend(vidDiv);
                } else {
                    throw new Error(data.error || 'Unknown error');
                }

            } catch (e) {
                console.error('Bulk generation error:', e);
                item.status = 'error';
                showToast(`Failed: ${item.name}`, 'error');
            }

            renderQueue();
        }
    }
}

async function downloadAll() {
    try {
        window.location.href = '/api/queue/download-all';
    } catch (error) {
        showToast('Download failed', 'error');
    }
}

// =============================================================================
// INIT
// =============================================================================

document.addEventListener('DOMContentLoaded', init);
