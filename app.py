from flask import Flask, render_template, request, redirect, session
import pickle, re, os, sqlite3, smtplib
from datetime import datetime
from textblob import TextBlob
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

app = Flask(__name__)
app.secret_key = "secretkey"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database.db")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VEC_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

# ================= DATABASE INIT =================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS complaints (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT,
        complaint TEXT,
        category TEXT,
        confidence REAL,
        sentiment TEXT,
        priority TEXT,
        status TEXT,
        response TEXT,
        timestamp TEXT)''')
    conn.commit()
    conn.close()

# ================= LOAD MODEL =================
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(VEC_PATH, 'rb') as f:
    vectorizer = pickle.load(f)

# ================= ROUTES =================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        complaint = request.form.get('complaint', '').strip()

        if not complaint:
            return render_template('index.html', message='Complaint is required')

        complaint_clean = clean_text(complaint)
        transformed = vectorizer.transform([complaint_clean])

        prediction = str(model.predict(transformed)[0])
        confidence = round(float(max(model.predict_proba(transformed)[0]) * 100), 2)

        sentiment = get_sentiment(complaint)
        priority = get_priority(sentiment)
        auto_response = generate_response(prediction, sentiment)

        # Save Complaint
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO complaints
            (name,email,complaint,category,confidence,sentiment,priority,status,response,timestamp)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        ''', (
            name, email, complaint, prediction, confidence,
            sentiment, priority, 'Pending', auto_response,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ))
        conn.commit()
        conn.close()

        # Send Email Notification
        send_email(email, name, prediction, priority, auto_response, complaint)

        return render_template(
            'index.html',
            result=prediction,
            confidence=confidence,
            response=auto_response,
            sentiment=sentiment,
            message='Complaint stored successfully'
        )

    except Exception as e:
        return render_template('index.html', message=f'Error: {str(e)}')

@app.route('/dashboard')
def dashboard():
    if 'admin' not in session:
        return redirect('/login')

    category = request.args.get('category', '').strip()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    if category:
        cursor.execute('SELECT * FROM complaints WHERE category=? ORDER BY id DESC', (category,))
    else:
        cursor.execute('SELECT * FROM complaints ORDER BY id DESC')

    data = cursor.fetchall()
    conn.close()

    return render_template('dashboard.html', data=data)

@app.route('/mark_solved/<int:id>')
def mark_solved(id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get complaint details first
    cursor.execute("SELECT name, email, category FROM complaints WHERE id=?", (id,))
    row = cursor.fetchone()

    # Update status
    cursor.execute("UPDATE complaints SET status='Solved' WHERE id=?", (id,))
    conn.commit()
    conn.close()

    # Send solved email
    if row:
        name = row[0]
        email = row[1]
        category = row[2]
        send_solved_email(email, name, category)

    return redirect('/dashboard')
def send_solved_email(to_email, name, category):
    sender_email = "karthiknakkawar7@gmail.com"
    sender_password = "ynzerqxrobqisxtn"

    subject = "Your Complaint Has Been Resolved"

    body = f"""
Hello {name},

We are happy to inform you that your complaint regarding {category} has been resolved successfully.

Thank you for your patience and support.

Regards,
Support Team
"""

    try:
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, to_email, msg.as_string())
        server.quit()

        print("Solved Email Sent")

    except Exception as e:
        print("Solved Email Error:", str(e))

@app.route('/delete/<int:id>')
def delete(id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM complaints WHERE id=?', (id,))
    conn.commit()
    conn.close()
    return redirect('/dashboard')

@app.route('/delete_all')
def delete_all():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM complaints')
    conn.commit()
    conn.close()
    return redirect('/dashboard')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form['username'] == 'karthikNakkawar' and request.form['password'] == 'Mynameis?':
            session['admin'] = True
            return redirect('/dashboard')
        return render_template('login.html', error='Invalid Credentials')

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('admin', None)
    return redirect('/')

@app.route('/confusion')
def confusion():
    if 'admin' not in session:
        return redirect('/login')
    return render_template('confusion.html')

# ================= HELPERS =================
def clean_text(text):
    text = text.lower()
    return re.sub(r'[^a-zA-Z\s]', '', text)

def get_priority(sentiment):
    return 'High' if sentiment == 'Negative' else 'Medium' if sentiment == 'Neutral' else 'Low'

def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    return 'Positive' if polarity > 0.2 else 'Negative' if polarity < 0 else 'Neutral'

def generate_response(category, sentiment):
    responses = {
        'Billing': {
            'Negative': 'We apologize for the billing inconvenience. Our billing team will review the charge urgently and contact you soon.',
            'Neutral': 'Your billing request has been received. Our billing team will verify and update you shortly.',
            'Positive': 'Thank you for contacting billing support. We will assist you shortly.'
        },

        'Delivery': {
            'Negative': 'We are sorry for the delivery delay. Your issue has been escalated and our delivery team is checking it now.',
            'Neutral': 'Your delivery concern has been received. Our team will track the shipment and update you soon.',
            'Positive': 'Thank you for reaching out. Our delivery team will assist you shortly.'
        },

        'Product': {
            'Negative': 'We regret the inconvenience caused by the product issue. Replacement or refund options will be reviewed immediately.',
            'Neutral': 'Your product complaint has been registered. Our support team will inspect the issue soon.',
            'Positive': 'Thank you for your feedback. Our product team will assist you shortly.'
        },

        'Service': {
            'Negative': 'We are sorry for your poor service experience. Your complaint has been marked as priority for quick resolution.',
            'Neutral': 'Your service feedback has been recorded. Our team will review and improve this issue.',
            'Positive': 'Thank you for your valuable feedback. We appreciate your support.'
        },

        'Technical': {
            'Negative': 'We apologize for the technical issue. Our technical team is working on it with high priority.',
            'Neutral': 'Your technical complaint has been received. Our engineers will investigate and update you soon.',
            'Positive': 'Thank you for informing us. Our technical team will assist you shortly.'
        }
    }

    return responses.get(category, {}).get(
        sentiment,
        'Thank you. Your complaint has been received and our team will contact you soon.'
    )

# ================= EMAIL FUNCTION =================
def send_email(to_email, name, category, priority, response, complaint):
    sender_email = "karthiknakkawar7@gmail.com"
    sender_password = "ynzerqxrobqisxtn"

    owner_email = "karthiknakkawar7@gmail.com"

    # ================= USER MAIL =================
    user_subject = "Complaint Received Successfully"

    user_body = f"""
Hello {name},

Your complaint has been received successfully.

Complaint:
{complaint}

Category: {category}
Priority: {priority}

Response:
{response}

Thank you,
Support Team
"""

    # ================= OWNER MAIL =================
    owner_subject = "New Complaint Submitted"

    owner_body = f"""
New complaint received.

Name: {name}
User Email: {to_email}

Complaint:
{complaint}

Category: {category}
Priority: {priority}

AI Response:
{response}
"""

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)

        # Send to User
        msg1 = MIMEMultipart()
        msg1["From"] = sender_email
        msg1["To"] = to_email
        msg1["Subject"] = user_subject
        msg1.attach(MIMEText(user_body, "plain"))
        server.sendmail(sender_email, to_email, msg1.as_string())

        # Send to Owner
        msg2 = MIMEMultipart()
        msg2["From"] = sender_email
        msg2["To"] = owner_email
        msg2["Subject"] = owner_subject
        msg2.attach(MIMEText(owner_body, "plain"))
        server.sendmail(sender_email, owner_email, msg2.as_string())

        server.quit()
        print("User + Owner Emails Sent Successfully")

    except Exception as e:
        print("Email Error:", str(e))

# ================= START =================
init_db()

if __name__ == '__main__':
    app.run(debug=True)