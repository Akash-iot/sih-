# ETHEREYE Platform Test Script (Fixed Version)

Write-Host "🚀 ETHEREYE Platform Test" -ForegroundColor Cyan
Write-Host "=========================" -ForegroundColor Cyan
Write-Host ""

# Test Backend Health
Write-Host "1. Testing Backend Server (Port 8000)..." -ForegroundColor Yellow
try {
    $backendResponse = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get -TimeoutSec 5
    Write-Host "   ✅ Backend Status: $($backendResponse.status)" -ForegroundColor Green
    Write-Host "   ✅ Backend Mode: $($backendResponse.mode)" -ForegroundColor Green
    Write-Host "   ✅ Scrapers: Etherscan ($($backendResponse.scrapers.etherscan)), CoinGecko ($($backendResponse.scrapers.coingecko))" -ForegroundColor Green
} catch {
    Write-Host "   ❌ Backend Error: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "   💡 Solution: Make sure backend is running" -ForegroundColor Yellow
}

Write-Host ""

# Test Frontend Server
Write-Host "2. Testing Frontend Server (Port 3000)..." -ForegroundColor Yellow
try {
    $frontendResponse = Invoke-WebRequest -Uri "http://localhost:3000" -Method Get -TimeoutSec 5 -UseBasicParsing
    if ($frontendResponse.StatusCode -eq 200) {
        Write-Host "   ✅ Frontend Status: Running (HTTP $($frontendResponse.StatusCode))" -ForegroundColor Green
    }
} catch {
    Write-Host "   ❌ Frontend Error: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "   💡 Solution: Make sure frontend is running" -ForegroundColor Yellow
}

Write-Host ""

# Test API Endpoints
Write-Host "3. Testing API Endpoints..." -ForegroundColor Yellow

# Test Gas Prices
try {
    $gasResponse = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/live/gas" -Method Get -TimeoutSec 5
    Write-Host "   ✅ Gas Prices API: Working" -ForegroundColor Green
    Write-Host "      Safe: $($gasResponse.safe_gas_price) gwei, Standard: $($gasResponse.standard_gas_price) gwei, Fast: $($gasResponse.fast_gas_price) gwei" -ForegroundColor Cyan
} catch {
    Write-Host "   ❌ Gas Prices API Error: $($_.Exception.Message)" -ForegroundColor Red
}

# Test Spider Map
try {
    $spiderResponse = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/spider-map/network/0x742d35cc6634c0532925a3b8d4ba26c0c8b0e76e" -Method Get -TimeoutSec 10
    Write-Host "   ✅ Spider Map API: Working" -ForegroundColor Green
    Write-Host "      Network: $($spiderResponse.network.nodes.Count) nodes, $($spiderResponse.network.links.Count) connections" -ForegroundColor Cyan
} catch {
    Write-Host "   ❌ Spider Map API Error: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""

# Final Status
Write-Host "🎯 ETHEREYE Platform Access Links:" -ForegroundColor Magenta
Write-Host "===================================" -ForegroundColor Magenta
Write-Host ""
Write-Host "📊 Your Platform URLs:" -ForegroundColor Yellow
Write-Host "   🏠 Homepage:          http://localhost:3000/index.html" -ForegroundColor White
Write-Host "   🕸️  Enhanced Traces:   http://localhost:3000/traces.html" -ForegroundColor White  
Write-Host "   🧪 API Test Page:     http://localhost:3000/test-api.html" -ForegroundColor White
Write-Host "   📡 Backend API:       http://localhost:8000" -ForegroundColor White
Write-Host "   📚 API Documentation: http://localhost:8000/docs" -ForegroundColor White
Write-Host ""
Write-Host "🎮 Quick Test Steps:" -ForegroundColor Yellow
Write-Host "   1. Open: http://localhost:3000/traces.html" -ForegroundColor Gray
Write-Host "   2. Enter: 0x742d35cc6634c0532925a3b8d4ba26c0c8b0e76e" -ForegroundColor Gray
Write-Host "   3. Click 'Trace' or 'Demo' button" -ForegroundColor Gray
Write-Host "   4. Explore the interactive network!" -ForegroundColor Gray
Write-Host ""
Write-Host "Press Enter to continue..."
Read-Host
