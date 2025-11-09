@echo off
echo 正在启动前端服务...
start "web" cmd /k "cd /d C:\Users\GLY-PC\Documents\CODE\project\INFV3 && npm run dev"

echo 正在启动后端服务...
start "backserver" cmd /k "cd /d C:\Users\GLY-PC\Documents\CODE\project\INFV3\backend && "C:\ProgramData\anaconda3\python.exe" app.py"

echo END
    