from flask import Flask, render_template, request

app = Flask(__name__)

# Mock data to simulate Screenshot 1 (Populated Active State)
MOCK_DETECTIONS = [
    {"object": "person", "confidence": 90, "bbox": [477, 228, 560, 519]},
    {"object": "person", "confidence": 88, "bbox": [211, 241, 284, 507]},
    {"object": "person", "confidence": 88, "bbox": [110, 235, 224, 536]},
    {"object": "bus", "confidence": 80, "bbox": [95, 136, 551, 444]}
]

@app.route('/')
def index():
    """Renders the product landing capability page (Screenshot 2)."""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """
    Renders the Tactical Console.
    To view the IDLE state (Screenshot 3): clear the URL parameters -> /dashboard
    To view the ACTIVE state (Screenshot 1): add ?view=active -> /dashboard?view=active
    """
    view_state = request.args.get('view', 'idle')
    
    if view_state == 'active':
        return render_template(
            'dashboard.html',
            detections=MOCK_DETECTIONS,
            resolution="810×1080",
            original_image="https://images.unsplash.com/photo-1542282088-72c9c27ed0cd?w=500", # Temporary visual fallback
            processed_image="https://images.unsplash.com/photo-1542282088-72c9c27ed0cd?w=500",
            detection_image="https://images.unsplash.com/photo-1542282088-72c9c27ed0cd?w=500"
        )
    
    # Defaults to idle state if no query parameters are given
    return render_template(
        'dashboard.html',
        detections=[],
        resolution=None,
        original_image=None,
        processed_image=None,
        detection_image=None
    )

if __name__ == '__main__':
    # Runs a local development server on http://127.0.0.1:5000
    app.run(debug=True)