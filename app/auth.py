import os
from functools import wraps
from flask import session, redirect, url_for, request, flash

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('authorized'):
            return redirect(url_for('routes.login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

def check_access_code(code):
    master_code = os.getenv('APP_ACCESS_CODE', 'alpha123')
    if code == master_code:
        session['authorized'] = True
        return True
    return False
