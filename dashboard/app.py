let latencyChart = null;

async function fetchData(endpoint) {
    try {
        const response = await fetch(`/api/${endpoint}`);
        return await response.json();
    } catch (error) {
        console.error(`Error fetching ${endpoint}:`, error);
        return null;
    }
}

function updateStatus() {
    fetchData('status').then(data => {
        if (data && data.status === 'running') {
            document.getElementById('status-dot').classList.add('connected');
            document.getElementById('status-text').textContent = 'Connected';
        } else {
            document.getElementById('status-dot').classList.remove('connected');
            document.getElementById('status-text').textContent = 'Disconnected';
        }
    });
}

function initializeChart() {
    const ctx = document.getElementById('latency-chart').getContext('2d');
    latencyChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Latency (ms)',
                data: [],
                borderColor: '#3498db',
                backgroundColor: 'rgba(52, 152, 219, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Latency (ms)'
                    }
                },
                x: {
                    display: false
                }
            }
        }
    });
}

function updateChart(metrics) {
    if (!latencyChart || !metrics || metrics.length === 0) return;
    
    const recentMetrics = metrics.slice(0, 50).reverse();
    const labels = recentMetrics.map((_, i) => i);
    const data = recentMetrics.map(m => m.latency_ms);
    
    latencyChart.data.labels = labels;
    latencyChart.data.datasets[0].data = data;
    latencyChart.update('none');
}

function updateOverview(phaseSummary) {
    if (!phaseSummary) return;
    
    let totalQueries = 0;
    let totalLatency = 0;
    let successfulQueries = 0;
    
    Object.values(phaseSummary).forEach(phase => {
        totalQueries += phase.query_count;
        totalLatency += phase.avg_latency * phase.query_count;
        successfulQueries += (phase.query_count * phase.success_rate / 100);
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
    
    Object.entries(phaseSummary).forEach(([phase, stats]) => {
        const phaseDiv = document.createElement('div');
        phaseDiv.className = 'phase-item';
        phaseDiv.innerHTML = `
            <div class="phase-header">${phase}</div>
            <div class="phase-stats">
                <div class="phase-stat">
                    <span class="phase-stat-label">Queries</span>
                    <span class="phase-stat-value">${stats.query_count}</span>
                </div>
                <div class="phase-stat">
                    <span class="phase-stat-label">Avg Latency</span>
                    <span class="phase-stat-value">${stats.avg_latency} ms</span>
                </div>
                <div class="phase-stat">
                    <span class="phase-stat-label">Success Rate</span>
                    <span class="phase-stat-value">${stats.success_rate}%</span>
                </div>
            </div>
        `;
        container.appendChild(phaseDiv);
    });
}

function updateLatencyDistribution(distribution) {
    if (!distribution) return;
    
    document.getElementById('p50').textContent = distribution.p50 + ' ms';
    document.getElementById('p95').textContent = distribution.p95 + ' ms';
    document.getElementById('p99').textContent = distribution.p99 + ' ms';
    document.getElementById('p999').textContent = distribution.p999 + ' ms';
}

function updateLearningStats(stats) {
    if (!stats) return;
    
    document.getElementById('learning-policy-updates').textContent = stats.policy_updates;
    document.getElementById('policy-updates').textContent = stats.policy_updates;
    document.getElementById('meta-runs').textContent = stats.meta_learning_runs;
    document.getElementById('latest-loss').textContent = 
        stats.latest_loss !== null ? stats.latest_loss.toFixed(6) : 'N/A';
}

function updateRecentQueries(metrics) {
    if (!metrics || metrics.length === 0) return;
    
    const tbody = document.getElementById('queries-tbody');
    tbody.innerHTML = '';
    
    metrics.slice(0, 20).forEach(query => {
        const row = document.createElement('tr');
        const timestamp = new Date(query.timestamp).toLocaleString();
        const statusClass = query.success ? 'status-success' : 'status-error';
        const statusText = query.success ? '✓ Success' : '✗ Failed';
        
        row.innerHTML = `
            <td>${timestamp}</td>
            <td>${query.phase}</td>
            <td>${query.latency_ms.toFixed(2)}</td>
            <td class="${statusClass}">${statusText}</td>
        `;
        tbody.appendChild(row);
    });
}

async function updateDashboard() {
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
}

document.addEventListener('DOMContentLoaded', () => {
    initializeChart();
    updateStatus();
    updateDashboard();
    
    setInterval(updateStatus, 2000);
    setInterval(updateDashboard, 3000);
});