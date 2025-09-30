Write-Host "🚀 ETHEREYE Platform Test" -ForegroundColor Cyan
Write-Host "=========================" -ForegroundColor Cyan

Write-Host "Backend Health: " -NoNewline
try { 
    $h = Invoke-RestMethod "http://localhost:8000/health"
    Write-Host "✅ $($h.status)" -ForegroundColor Green 
} catch { 
    Write-Host "❌ Not running" -ForegroundColor Red 
}

Write-Host "Frontend Status: " -NoNewline
try { 
    $f = Invoke-WebRequest "http://localhost:3000" -UseBasicParsing
    Write-Host "✅ HTTP $($f.StatusCode)" -ForegroundColor Green 
} catch { 
    Write-Host "❌ Not running" -ForegroundColor Red 
}

Write-Host "Spider API: " -NoNewline
try { 
    $s = Invoke-RestMethod "http://localhost:8000/api/v1/spider-map/network/0x742d35cc6634c0532925a3b8d4ba26c0c8b0e76e"
    Write-Host "✅ $($s.network.nodes.Count) nodes" -ForegroundColor Green 
} catch { 
    Write-Host "❌ Failed" -ForegroundColor Red 
}

Write-Host ""
Write-Host "🌟 Access Links:" -ForegroundColor Yellow
Write-Host "   Transaction Traces: http://localhost:3000/traces.html"
Write-Host "   API Test Page:   http://localhost:3000/test-api.html"
Write-Host "   Backend API:     http://localhost:8000"
