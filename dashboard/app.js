let latencyChart = null;

async function fetchData(endpoint) {
    try {
        const response = await fetch(`/api/${endpoint}`);
        if (!response.ok) {
            console.error(`Failed to fetch ${endpoint}: ${response.status}`);
            return null;
        }
        return await response.json();
    } catch (error) {
        console.error(`Error fetching ${endpoint}:`, error);
        return null;
    }
}

function initializeChart() {
    const ctx = document.getElementById('latency-chart');
    if (!ctx) {
        console.error('Chart canvas not found');
        return;
    }
    
    latencyChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Query Latency (ms)',
                data: [],
                borderColor: '#3498db',
                backgroundColor: 'rgba(52, 152, 219, 0.1)',
                borderWidth: 2,
                tension: 0.4,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                }
            },
            scales: {
                x: {
                    display: false
                },
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Latency (ms)'
                    }
                }
            },
            animation: {
                duration: 0
            }
        }
    });
}

function updateChart(metrics) {
    if (!latencyChart || !metrics || metrics.length === 0) return;
    
    // Take most recent 100 points for performance
    const data = metrics.slice(0, 100).reverse();
    
    latencyChart.data.labels = data.map((_, i) => i);
    latencyChart.data.datasets[0].data = data.map(m => m.latency_ms || 0);
    latencyChart.update();
}

async function updateStatus() {
    const statusData = await fetchData('status');
    
    if (!statusData) {
        document.getElementById('status-dot').className = 'status-dot status-error';
        document.getElementById('status-text').textContent = 'Connection Error';
        return;
    }
    
    const statusDot = document.getElementById('status-dot');
    const statusText = document.getElementById('status-text');
    
    if (statusData.db_exists) {
        statusDot.className = 'status-dot status-success';
        statusText.textContent = 'Connected';
    } else {
        statusDot.className = 'status-dot status-warning';
        statusText.textContent = 'No Database';
    }
}

function updateOverview(phaseSummary) {
    if (!phaseSummary || Object.keys(phaseSummary).length === 0) {
        document.getElementById('total-queries').textContent = '0';
        document.getElementById('avg-latency').textContent = '0 ms';
        document.getElementById('success-rate').textContent = '0%';
        return;
    }
    
    let totalQueries = 0;
    let totalLatency = 0;
    let successfulQueries = 0;
    
    Object.values(phaseSummary).forEach(phase => {
        totalQueries += phase.query_count || 0;
        totalLatency += (phase.avg_latency || 0) * (phase.query_count || 0);
        successfulQueries += ((phase.success_rate || 0) / 100) * (phase.query_count || 0);
    });
    
    const avgLatency = totalQueries > 0 ? (totalLatency / totalQueries).toFixed(2) : 0;
    const successRate = totalQueries > 0 ? ((successfulQueries / totalQueries) * 100).toFixed(1) : 0;
    
    document.getElementById('total-queries').textContent = totalQueries.toLocaleString();
    document.getElementById('avg-latency').textContent = avgLatency + ' ms';
    document.getElementById('success-rate').textContent = successRate + '%';
}

function updatePhaseComparison(phaseSummary) {
    if (!phaseSummary) return;
    
    const container = document.getElementById('phase-comparison');
    container.innerHTML = '';
    
    if (Object.keys(phaseSummary).length === 0) {
        container.innerHTML = '<p style="text-align: center; color: #7f8c8d;">No phase data available</p>';
        return;
    }
    
    Object.entries(phaseSummary).forEach(([phase, stats]) => {
        const phaseDiv = document.createElement('div');
        phaseDiv.className = 'phase-item';
        phaseDiv.innerHTML = `
            <div class="phase-header">${escapeHtml(phase)}</div>
            <div class="phase-stats">
                <div class="phase-stat">
                    <span class="phase-stat-label">Queries</span>
                    <span class="phase-stat-value">${stats.query_count || 0}</span>
                </div>
                <div class="phase-stat">
                    <span class="phase-stat-label">Avg Latency</span>
                    <span class="phase-stat-value">${stats.avg_latency || 0} ms</span>
                </div>
                <div class="phase-stat">
                    <span class="phase-stat-label">Success Rate</span>
                    <span class="phase-stat-value">${stats.success_rate || 0}%</span>
                </div>
            </div>
        `;
        container.appendChild(phaseDiv);
    });
}

function updateLatencyDistribution(distribution) {
    if (!distribution) return;
    
    document.getElementById('p50').textContent = (distribution.p50 || 0) + ' ms';
    document.getElementById('p95').textContent = (distribution.p95 || 0) + ' ms';
    document.getElementById('p99').textContent = (distribution.p99 || 0) + ' ms';
    document.getElementById('p999').textContent = (distribution.p999 || 0) + ' ms';
}

function updateLearningStats(stats) {
    if (!stats) return;
    
    document.getElementById('learning-policy-updates').textContent = stats.policy_updates || 0;
    document.getElementById('policy-updates').textContent = stats.policy_updates || 0;
    document.getElementById('meta-runs').textContent = stats.meta_learning_runs || 0;
    document.getElementById('latest-loss').textContent = 
        stats.latest_loss !== null && stats.latest_loss !== undefined 
            ? stats.latest_loss.toFixed(6) 
            : 'N/A';
}

function updateRecentQueries(metrics) {
    if (!metrics || metrics.length === 0) {
        const tbody = document.getElementById('queries-tbody');
        tbody.innerHTML = '<tr><td colspan="4" style="text-align: center; color: #7f8c8d;">No query data available</td></tr>';
        return;
    }
    
    const tbody = document.getElementById('queries-tbody');
    tbody.innerHTML = '';
    
    metrics.slice(0, 20).forEach(query => {
        const row = document.createElement('tr');
        const timestamp = new Date(query.timestamp).toLocaleString();
        
        // Use CSS classes and HTML entities instead of Unicode characters for better Windows compatibility
        const statusClass = query.success ? 'status-success' : 'status-error';
        // Use HTML entities that work reliably on Windows: &#10003; (checkmark) and &#10007; (X)
        const statusText = query.success ? '&#10003; Success' : '&#10007; Failed';
        
        row.innerHTML = `
            <td>${escapeHtml(timestamp)}</td>
            <td>${escapeHtml(query.phase || 'unknown')}</td>
            <td>${(query.latency_ms || 0).toFixed(2)}</td>
            <td class="${statusClass}">${statusText}</td>
        `;
        tbody.appendChild(row);
    });
}

// Helper function to escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

async function updateDashboard() {
    try {
        const [metrics, phaseSummary, learningStats, distribution] = await Promise.all([
            fetchData('metrics'),
            fetchData('phase-summary'),
            fetchData('learning-stats'),
            fetchData('latency-distribution')
        ]);
        
        updateChart(metrics);
        updateOverview(phaseSummary);
        updatePhaseComparison(phaseSummary);
        updateLatencyDistribution(distribution);
        updateLearningStats(learningStats);
        updateRecentQueries(metrics);
    } catch (error) {
        console.error('Error updating dashboard:', error);
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('Dashboard initializing...');
    
    initializeChart();
    updateStatus();
    updateDashboard();
    
    // Update status more frequently than data
    setInterval(updateStatus, 2000);
    setInterval(updateDashboard, 3000);
    
    console.log('Dashboard initialized successfully');
});