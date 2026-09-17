# Yakutan 远程推理服务防火墙放行（需管理员权限运行；由「允许局域网访问(防火墙放行).bat」提权调用）
# 只需放行 WebSocket 服务端口 18775；18776 (llama-server) 仅绑定 127.0.0.1，无需也不应放行入站。

Write-Host 'Adding firewall rule for TCP 18775...' -ForegroundColor Cyan
try {
    Remove-NetFirewallRule -DisplayName 'Yakutan-Remote-Inference' -ErrorAction Stop
} catch {}
New-NetFirewallRule -DisplayName 'Yakutan-Remote-Inference' -Direction Inbound -LocalPort 18775 -Protocol TCP -Action Allow -Profile Any | Out-Null
Write-Host '[OK] Inbound rule for TCP 18775 created successfully!' -ForegroundColor Green
Write-Host ''
Read-Host 'Press Enter to close'
