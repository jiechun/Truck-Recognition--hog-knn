@echo off 
Setlocal Enabledelayedexpansion 
(for /r %%i in (*.png) do ( 
if not "%%~dpi"=="!var!" set n+=1
set "var=%%~dpi"  
echo.%%i !n!))>"new.txt"