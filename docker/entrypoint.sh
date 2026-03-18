#!/bin/bash
set -e

# 引数が渡された場合はそのまま実行（スクリプト実行 / シェル起動など）
if [ "$#" -gt 0 ]; then
    exec "$@"
fi

# 引数なし → VNC デスクトップ起動
# ── xstartup (desktop session) ───────────────────────────────────────────────
cat > /tmp/xstartup.sh << 'EOF'
#!/bin/sh
unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESS
exec startxfce4
EOF
chmod +x /tmp/xstartup.sh

# ── supervisor config ─────────────────────────────────────────────────────────
mkdir -p /etc/supervisor/conf.d
cat > /etc/supervisor/conf.d/vnc.conf << EOF
[supervisord]
nodaemon=true
user=root

[program:vnc]
command=tigervncserver :0 -fg \
    -xstartup /tmp/xstartup.sh \
    -geometry ${RESOLUTION:-1280x800} \
    -depth 24 \
    -localhost no \
    -SecurityTypes None \
    --I-KNOW-THIS-IS-INSECURE
autorestart=true
stdout_logfile=/var/log/vnc.log
stderr_logfile=/var/log/vnc.log

[program:novnc]
command=websockify --web=/usr/share/novnc 6080 localhost:5900
autorestart=true
stdout_logfile=/var/log/novnc.log
stderr_logfile=/var/log/novnc.log
EOF

echo "================================================================"
echo " noVNC: http://localhost:6080/vnc.html (no password)"
echo " VNC  : localhost:5900 (no password)"
echo "================================================================"

exec /usr/bin/supervisord -c /etc/supervisor/conf.d/vnc.conf
