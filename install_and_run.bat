@echo off
echo Creanting enviroment...
python -m venv venv

echo Activating enviroment...
call venv\Scripts\activate

echo Installing packages...
pip install -r requirements.txt

echo Installing finished.
pause