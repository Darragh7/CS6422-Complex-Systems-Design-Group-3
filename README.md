run setup.ps1 to create virtual env and install dependencies

then run 
     .\.venv\Scripts\Activate.ps1
        cd backend; python -m uvicorn app.main:app --reload
this activates the venv and starts fastAPI

NOTE: If you run into issue with permissions, try this:
        Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass0