import os
from flask import Blueprint, render_template, request, jsonify, session, redirect, url_for, flash
from werkzeug.utils import secure_filename
from .auth import login_required, check_access_code
from .engine import get_rag_manager

bp = Blueprint('routes', __name__)

@bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        code = request.form.get('access_code')
        if check_access_code(code):
            return redirect(url_for('routes.dashboard'))
        else:
            flash('Invalid Access Code', 'error')
    return render_template('login.html')

@bp.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('routes.login'))

@bp.route('/')
@login_required
def dashboard():
    return render_template('dashboard.html')

@bp.route('/api/chat', methods=['POST'])
@login_required
def chat():
    data = request.json
    message = data.get('message')
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    
    rag = get_rag_manager()
    try:
        response = rag.query(message)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/upload', methods=['POST'])
@login_required
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        os.makedirs('data', exist_ok=True)
        file_path = os.path.join('data', filename)
        file.save(file_path)
        
        # Start background ingestion
        rag = get_rag_manager()
        rag.ingest_async(file_path)
        
        return jsonify({'message': f'File {filename} uploaded and ingestion started.'})

@bp.route('/api/status')
@login_required
def status():
    rag = get_rag_manager()
    counts = rag.vector_store._collection.count()
    return jsonify({
        'vector_count': counts,
        'status': 'Online'
    })
