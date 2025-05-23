import streamlit as st

# ✅ Configure the page FIRST — before any other Streamlit commands or module imports
st.set_page_config(page_title="Foot Measurement System", layout="wide")

import sqlite3
from admin import show_admin_panel
from user import show_user_panel

# ---------- Database Initialization ----------
def init_db():
    conn = sqlite3.connect('foot_measurement.db')
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY,
                  name TEXT,
                  password TEXT,
                  email TEXT,
                  role TEXT)''')

    c.execute('''CREATE TABLE IF NOT EXISTS feedback
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT,
                  message TEXT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

    c.execute('''CREATE TABLE IF NOT EXISTS measurements
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT,
                  foot_length REAL,
                  foot_width REAL,
                  size_vn TEXT,
                  size_uk TEXT,
                  size_us TEXT,
                  size_eu TEXT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

    c.execute("SELECT * FROM users WHERE username = 'admin'")
    if not c.fetchone():
        c.execute("INSERT INTO users VALUES (?, ?, ?, ?, ?)",
                  ('admin', 'Admin', 'admin123', 'admin@example.com', 'admin'))

    conn.commit()
    conn.close()

# ---------- Helper Functions ----------
def get_users():
    conn = sqlite3.connect('foot_measurement.db')
    c = conn.cursor()
    c.execute("SELECT username, name, password, email, role FROM users")
    users = c.fetchall()
    conn.close()
    return users

def prepare_credentials():
    users = get_users()
    credentials = {'usernames': {}}
    for user in users:
        credentials['usernames'][user[0]] = {
            'name': user[1],
            'password': user[2],
            'email': user[3],
            'role': user[4]
        }
    return credentials

# ---------- Main App ----------
def main():
    init_db()
    credentials = prepare_credentials()

    # If logged in
    if 'username' in st.session_state:
        if st.session_state.get('role') == 'admin':
            show_admin_panel()
        else:
            show_user_panel()

    # Login page
    else:
        st.markdown("""
            <div style="text-align:center; padding-top: 2rem;">
                <h1 style="color: #3b82f6;">👣 Foot Measurement System</h1>
                <p style="font-size: 1.1rem; color: #666;">Welcome! Please log in to continue.</p>
            </div>
        """, unsafe_allow_html=True)

        with st.container():
            col1, col2, col3 = st.columns([2, 4, 2])
            with col2:
                with st.form("login_form", clear_on_submit=False):
                    st.subheader("🔐 Login")
                    username = st.text_input("Username", placeholder="Enter your username")
                    password = st.text_input("Password", type="password", placeholder="Enter your password")
                    submitted = st.form_submit_button("Login")

                    if submitted:
                        user = credentials['usernames'].get(username, None)
                        if user and user['password'] == password:
                            st.session_state['username'] = username
                            st.session_state['role'] = user['role']
                            st.success("Logged in successfully!")
                            st.rerun()
                        else:
                            st.error("Invalid username or password.")

        st.markdown("""
            <style>
                .stButton>button {
                    background-color: #3b82f6;
                    color: white;
                    border-radius: 8px;
                    padding: 0.5rem 1.2rem;
                    font-weight: 600;
                    margin-top: 1rem;
                }
                .stTextInput>div>input {
                    border-radius: 5px;
                }
            </style>
        """, unsafe_allow_html=True)

# ---------- Run ----------
if __name__ == "__main__":
    main()
