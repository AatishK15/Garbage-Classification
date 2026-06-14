/* ============================================================
   Garbage Classification System — App Logic
   TensorFlow.js + MobileNet for in-browser AI classification
   ============================================================ */

// ─── Waste Category Data ───────────────────────────────────────
const CATEGORIES = {
    cardboard: {
        icon: '📦',
        color: '#c0853b',
        bin: '♻️ Blue Recycling Bin',
        instruction: 'Flatten all cardboard boxes before recycling. Remove any tape, staples, or plastic liners. Keep dry — wet cardboard goes in compost or trash.',
        keywords: ['carton', 'cardboard', 'packet', 'crate', 'box', 'packaging', 'envelope', 'mail']
    },
    glass: {
        icon: '🫙',
        color: '#27ae60',
        bin: '♻️ Green Glass Bin',
        instruction: 'Rinse glass containers before recycling. Remove lids and caps. Do not include mirrors, ceramics, window glass, or light bulbs — only bottles and jars.',
        keywords: ['bottle', 'wine bottle', 'beer bottle', 'pop bottle', 'water bottle', 'vase', 'goblet', 'glass', 'jar', 'pitcher', 'cocktail shaker', 'beer glass', 'cup']
    },
    metal: {
        icon: '🥫',
        color: '#7f8c8d',
        bin: '♻️ Blue Recycling Bin',
        instruction: 'Rinse food cans and aluminum containers. Crush cans to save space. Remove paper labels if possible. Includes tin cans, aluminum foil, and metal lids.',
        keywords: ['can', 'tin can', 'pot', 'pan', 'wok', 'caldron', 'frying pan', 'steel drum', 'safe', 'lock', 'chain', 'nail', 'screw', 'hook', 'thimble', 'spatula', 'corkscrew', 'aluminum', 'metal', 'iron', 'canister']
    },
    organic: {
        icon: '🍂',
        color: '#8b6914',
        bin: '🟤 Brown Compost Bin',
        instruction: 'Compost food scraps, fruit peels, coffee grounds, and yard waste. Avoid meat, dairy, and oils in home composting. Use biodegradable bags if required.',
        keywords: ['banana', 'orange', 'lemon', 'apple', 'pineapple', 'strawberry', 'fig', 'pomegranate', 'broccoli', 'cauliflower', 'mushroom', 'cucumber', 'bell pepper', 'corn', 'head cabbage', 'meat loaf', 'pizza', 'burrito', 'hotdog', 'cheeseburger', 'bagel', 'pretzel', 'dough', 'food', 'fruit', 'vegetable', 'leaf', 'hay', 'straw', 'flower', 'daisy', 'rose', 'acorn', 'ear']
    },
    paper: {
        icon: '📄',
        color: '#3498db',
        bin: '♻️ Blue Recycling Bin',
        instruction: 'Recycle clean, dry paper products: newspapers, magazines, office paper, and junk mail. Shred sensitive documents first. No waxed or food-soiled paper.',
        keywords: ['envelope', 'book jacket', 'comic book', 'notebook', 'menu', 'paper towel', 'toilet tissue', 'tissue', 'newspaper', 'letter opener', 'binder', 'page', 'book', 'document', 'magazine', 'notepad']
    },
    plastic: {
        icon: '🧴',
        color: '#e67e22',
        bin: '♻️ Blue / Yellow Recycling Bin',
        instruction: 'Check the recycling number on the bottom. Rinse containers and replace caps. Avoid plastic bags in curbside recycling — return them to store drop-offs.',
        keywords: ['water bottle', 'plastic bag', 'bucket', 'pill bottle', 'soap dispenser', 'nipple', 'rubber eraser', 'band aid', 'shower cap', 'plunger', 'ladle', 'mixing bowl', 'measuring cup', 'strainer', 'container', 'jug', 'tub', 'pail']
    },
    trash: {
        icon: '🗑️',
        color: '#636e72',
        bin: '⬛ Black Landfill Bin',
        instruction: 'Non-recyclable items go here: chip bags, styrofoam, broken ceramics, diapers, and contaminated materials. When in doubt, check local guidelines.',
        keywords: ['trash', 'diaper', 'muzzle', 'mask', 'bib', 'shoe', 'sandal', 'clog', 'running shoe', 'sock', 'mitten', 'glove']
    }
};

// Extended keyword → category lookup (built once)
const KEYWORD_MAP = {};
for (const [category, data] of Object.entries(CATEGORIES)) {
    for (const kw of data.keywords) {
        KEYWORD_MAP[kw.toLowerCase()] = category;
    }
}

// ─── DOM Elements ──────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const uploadZone      = $('#uploadZone');
const fileInput       = $('#fileInput');
const uploadContent   = $('#uploadContent');
const previewWrapper  = $('#previewWrapper');
const previewImage    = $('#previewImage');
const removeBtn       = $('#removeBtn');
const fileInfo        = $('#fileInfo');
const fileName        = $('#fileName');
const fileSize        = $('#fileSize');
const classifyBtn     = $('#classifyBtn');
const resultsEmpty    = $('#resultsEmpty');
const resultsContent  = $('#resultsContent');
const resultCard      = $('#resultCard');
const resultIcon      = $('#resultIcon');
const resultClass     = $('#resultClass');
const resultConfidence = $('#resultConfidence');
const resultTime      = $('#resultTime');
const disposalText    = $('#disposalText');
const disposalBin     = $('#disposalBin');
const breakdownList   = $('#breakdownList');
const modelStatusEl   = $('#modelStatus');
const predictionCount = $('#predictionCount');
const categoriesGrid  = $('#categoriesGrid');
const toastContainer  = $('#toastContainer');
const navbar          = $('#navbar');

let currentFile = null;
let model = null;
let totalPredictions = 0;

// ─── Initialize ────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    renderCategories();
    setupEventListeners();
    loadModel();
    setupScrollEffects();
});

// ─── Model Loading ─────────────────────────────────────────────
async function loadModel() {
    updateModelStatus('loading', 'Loading AI model…');
    try {
        model = await mobilenet.load({ version: 2, alpha: 1.0 });
        updateModelStatus('ready', 'AI Ready');
        showToast('🧠 AI model loaded successfully!', 'success');
    } catch (err) {
        console.error('Model load error:', err);
        updateModelStatus('error', 'Load Failed');
        showToast('⚠️ Failed to load AI model. Check your internet connection and refresh.', 'error');
    }
}

function updateModelStatus(state, text) {
    const dot = modelStatusEl.querySelector('.status-dot');
    const label = modelStatusEl.querySelector('.status-text');
    dot.className = 'status-dot';
    if (state === 'loading') dot.classList.add('loading');
    else if (state === 'error') dot.classList.add('error');
    label.textContent = text;
}

// ─── Event Listeners ───────────────────────────────────────────
function setupEventListeners() {
    // Click to upload
    uploadZone.addEventListener('click', (e) => {
        if (e.target === removeBtn || removeBtn.contains(e.target)) return;
        fileInput.click();
    });

    // Keyboard accessibility
    uploadZone.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            fileInput.click();
        }
    });

    // File selected
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleFile(e.target.files[0]);
    });

    // Drag & drop
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('drag-over');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('drag-over');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
    });

    // Remove image
    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        clearUpload();
    });

    // Classify button
    classifyBtn.addEventListener('click', classify);
}

// ─── Scroll Effects ────────────────────────────────────────────
function setupScrollEffects() {
    window.addEventListener('scroll', () => {
        if (window.scrollY > 40) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
    });
}

// ─── File Handling ─────────────────────────────────────────────
function handleFile(file) {
    // Validate type
    const validTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp', 'image/gif'];
    if (!validTypes.includes(file.type)) {
        showToast('❌ Invalid file type. Please upload JPG, PNG, WebP, or BMP.', 'error');
        return;
    }

    // Validate size (10MB max)
    if (file.size > 10 * 1024 * 1024) {
        showToast('❌ File too large. Maximum size is 10MB.', 'error');
        return;
    }

    currentFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        uploadContent.style.display = 'none';
        previewWrapper.style.display = 'flex';
    };
    reader.readAsDataURL(file);

    // Show file info
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);
    fileInfo.style.display = 'flex';

    // Enable classify button
    classifyBtn.disabled = false;
}

function clearUpload() {
    currentFile = null;
    fileInput.value = '';
    previewImage.src = '';
    uploadContent.style.display = 'flex';
    previewWrapper.style.display = 'none';
    fileInfo.style.display = 'none';
    classifyBtn.disabled = true;

    // Reset results
    resultsContent.style.display = 'none';
    resultsEmpty.style.display = 'flex';
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
}

// ─── Classification ────────────────────────────────────────────
async function classify() {
    if (!currentFile || !model) {
        if (!model) showToast('⚠️ AI model is still loading. Please wait…', 'info');
        return;
    }

    // UI: show loading state
    const btnText = classifyBtn.querySelector('.btn-text');
    const btnLoader = classifyBtn.querySelector('.btn-loader');
    btnText.style.display = 'none';
    btnLoader.style.display = 'inline-flex';
    classifyBtn.disabled = true;

    try {
        const startTime = performance.now();

        // Create image element for TF.js
        const imgEl = new Image();
        imgEl.crossOrigin = 'anonymous';
        await new Promise((resolve, reject) => {
            imgEl.onload = resolve;
            imgEl.onerror = reject;
            imgEl.src = previewImage.src;
        });

        // Run MobileNet prediction
        const predictions = await model.classify(imgEl, 15);
        const elapsed = Math.round(performance.now() - startTime);

        // Map to garbage categories
        const result = mapToGarbageCategory(predictions);
        result.inferenceTime = elapsed;

        // Display results
        displayResults(result);

        // Update counter
        totalPredictions++;
        predictionCount.textContent = totalPredictions;

        showToast(`✅ Classified as ${result.topCategory.name.toUpperCase()}`, 'success');

    } catch (err) {
        console.error('Classification error:', err);
        showToast('❌ Classification failed. Please try a different image.', 'error');
    } finally {
        btnText.style.display = 'inline';
        btnLoader.style.display = 'none';
        classifyBtn.disabled = false;
    }
}

// ─── Map MobileNet → Garbage Categories ────────────────────────
function mapToGarbageCategory(predictions) {
    // Score each garbage category based on MobileNet predictions
    const scores = {};
    for (const cat of Object.keys(CATEGORIES)) {
        scores[cat] = 0;
    }

    for (const pred of predictions) {
        const className = pred.className.toLowerCase();
        const prob = pred.probability;

        // Check each keyword against the prediction class name
        let matched = false;
        for (const [keyword, category] of Object.entries(KEYWORD_MAP)) {
            if (className.includes(keyword) || keyword.includes(className)) {
                scores[category] += prob;
                matched = true;
                break;
            }
        }

        // Heuristic fallback for common items
        if (!matched) {
            // Try partial matching of individual words
            const words = className.split(/[,\s]+/);
            for (const word of words) {
                if (word.length < 3) continue;
                for (const [keyword, category] of Object.entries(KEYWORD_MAP)) {
                    if (keyword.includes(word) || word.includes(keyword)) {
                        scores[category] += prob * 0.5;
                        matched = true;
                        break;
                    }
                }
                if (matched) break;
            }
        }

        // If still no match, add small score to trash
        if (!matched) {
            scores['trash'] += prob * 0.15;
        }
    }

    // Normalize scores
    const totalScore = Object.values(scores).reduce((a, b) => a + b, 0);
    const normalized = {};
    for (const [cat, score] of Object.entries(scores)) {
        normalized[cat] = totalScore > 0 ? score / totalScore : 1 / 7;
    }

    // Sort by confidence
    const sorted = Object.entries(normalized)
        .sort((a, b) => b[1] - a[1]);

    const topCat = sorted[0][0];
    const topConf = sorted[0][1];

    return {
        topCategory: {
            name: topCat,
            confidence: topConf,
            ...CATEGORIES[topCat]
        },
        allCategories: sorted.map(([name, conf]) => ({
            name,
            confidence: conf,
            ...CATEGORIES[name]
        })),
        mobilenetPredictions: predictions
    };
}

// ─── Display Results ───────────────────────────────────────────
function displayResults(result) {
    const top = result.topCategory;
    const confPct = (top.confidence * 100).toFixed(1);

    // Primary result
    resultIcon.textContent = top.icon;
    resultClass.textContent = top.name;
    resultClass.style.color = top.color;

    resultConfidence.textContent = `Confidence: ${confPct}%`;
    resultConfidence.className = 'result-confidence';
    if (top.confidence >= 0.5) resultConfidence.classList.add('high');
    else if (top.confidence >= 0.25) resultConfidence.classList.add('medium');
    else resultConfidence.classList.add('low');

    resultTime.textContent = `⚡ Inference time: ${result.inferenceTime}ms`;

    // Disposal info
    disposalText.textContent = top.instruction;
    disposalBin.textContent = top.bin;

    // Confidence breakdown
    breakdownList.innerHTML = '';
    for (const cat of result.allCategories) {
        const pct = (cat.confidence * 100).toFixed(1);
        const item = document.createElement('div');
        item.className = 'breakdown-item';
        item.innerHTML = `
            <span class="breakdown-icon">${cat.icon}</span>
            <span class="breakdown-name">${cat.name}</span>
            <div class="breakdown-bar-bg">
                <div class="breakdown-bar-fill" style="background: linear-gradient(90deg, ${cat.color}, ${cat.color}88);"></div>
            </div>
            <span class="breakdown-pct">${pct}%</span>
        `;
        breakdownList.appendChild(item);
    }

    // Show results, hide empty state
    resultsEmpty.style.display = 'none';
    resultsContent.style.display = 'block';

    // Animate bars after render
    requestAnimationFrame(() => {
        setTimeout(() => {
            const bars = breakdownList.querySelectorAll('.breakdown-bar-fill');
            result.allCategories.forEach((cat, i) => {
                if (bars[i]) bars[i].style.width = `${cat.confidence * 100}%`;
            });
        }, 50);
    });

    // Re-trigger card animations
    resultCard.style.animation = 'none';
    void resultCard.offsetHeight;
    resultCard.style.animation = '';
}

// ─── Render Category Cards ─────────────────────────────────────
function renderCategories() {
    categoriesGrid.innerHTML = '';
    for (const [name, data] of Object.entries(CATEGORIES)) {
        const card = document.createElement('div');
        card.className = 'category-card';
        card.style.setProperty('--cat-color', data.color);
        card.innerHTML = `
            <div class="category-icon">${data.icon}</div>
            <div class="category-name" style="color: ${data.color};">${name}</div>
            <div class="category-instruction">${data.instruction}</div>
            <div class="category-bin">${data.bin}</div>
        `;
        categoriesGrid.appendChild(card);
    }
}

// ─── Toast Notifications ───────────────────────────────────────
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    toastContainer.appendChild(toast);

    setTimeout(() => {
        toast.classList.add('exit');
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}
