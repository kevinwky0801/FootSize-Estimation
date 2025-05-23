import streamlit as st
import pyqrcode  # type: ignore
import sqlite3
from io import BytesIO
from datetime import datetime

# Initialize database tables if they don't exist
def init_db():
    conn = sqlite3.connect('foot_measurement.db')
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT NOT NULL,
        password TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'user'
    )
    ''')
    
    # Create feedback table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        message TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        FOREIGN KEY (username) REFERENCES users(username)
    )
    ''')
    
    # Create feedback_responses table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS feedback_responses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        feedback_id INTEGER NOT NULL,
        admin_username TEXT NOT NULL,
        response TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        FOREIGN KEY (feedback_id) REFERENCES feedback(id),
        FOREIGN KEY (admin_username) REFERENCES users(username)
    )
    ''')
    
    # Create measurements table (if needed)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS measurements (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        foot_length REAL,
        foot_width REAL,
        timestamp TEXT NOT NULL,
        FOREIGN KEY (username) REFERENCES users(username)
    )
    ''')
    
    # Create admin user if not exists
    cursor.execute("SELECT * FROM users WHERE username = 'admin'")
    if not cursor.fetchone():
        cursor.execute(
            "INSERT INTO users (username, name, email, password, role) VALUES (?, ?, ?, ?, ?)",
            ('admin', 'Admin', 'admin@example.com', 'admin123', 'admin')
        )
    
    conn.commit()
    conn.close()

# Initialize the database
init_db()

# ---------- Database Connection ----------
def get_db_connection():
    try:
        conn = sqlite3.connect('foot_measurement.db')
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        st.error(f"Database connection error: {e}")
        return None

# ---------- Fetch Feedback ----------
def get_all_feedback():
    conn = get_db_connection()
    if not conn:
        return []
    try:
        c = conn.cursor()
        c.execute("SELECT id, username, message, timestamp FROM feedback ORDER BY timestamp DESC")
        return c.fetchall()
    except sqlite3.Error as e:
        st.error(f"Error fetching feedback: {e}")
        return []
    finally:
        conn.close()

# ---------- Fetch Users ----------
def get_all_users():
    conn = get_db_connection()
    if not conn:
        return []
    try:
        c = conn.cursor()
        c.execute("SELECT username, name, email, role FROM users ORDER BY username")
        return c.fetchall()
    except sqlite3.Error as e:
        st.error(f"Error fetching users: {e}")
        return []
    finally:
        conn.close()

# ---------- Generate QR Code ----------
def generate_qr(data):
    if not data:
        st.warning("Please enter data for QR code")
        return None
    try:
        qr = pyqrcode.create(data)
        buffer = BytesIO()
        qr.png(buffer, scale=6)
        return buffer.getvalue()
    except Exception as e:
        st.error(f"Error generating QR code: {e}")
        return None

# ---------- QR Code Page ----------
def show_qr_generator():
    st.title("QR Code Generator")
    my_link = "http://192.168.0.6:8501/"
    st.markdown(f"**Link:** [{my_link}]({my_link})")
    qr_image = generate_qr(my_link)
    if qr_image:
        st.image(qr_image, caption="QR Code", width=200)
        st.download_button("Download QR Code", data=qr_image, file_name="my_qr_code.png", mime="image/png")

# ---------- User Management ----------
def show_user_management():
    st.title("User Management")
    users = get_all_users()
    if not users:
        st.warning("No users found in database")
        return

    st.dataframe([dict(user) for user in users], use_container_width=True, hide_index=True)
    st.divider()
    st.subheader("Manage User")

    selected_user = st.selectbox("Select user to manage", [user['username'] for user in users], index=0)
    if not selected_user:
        return
    user_info = next(user for user in users if user['username'] == selected_user)

    if selected_user == 'admin':
        st.warning("Admin account cannot be modified")
        st.json(dict(user_info))
        return

    action = st.radio("Select action", ["View details", "Edit role", "Delete user"], horizontal=True)
    
    if action == "View details":
        st.json(dict(user_info))

    elif action == "Edit role":
        new_role = st.selectbox("Select new role", ["admin", "user"], index=0 if user_info['role'] == 'admin' else 1)
        if st.button("Update Role"):
            conn = get_db_connection()
            if not conn:
                return
            try:
                conn.execute("UPDATE users SET role = ? WHERE username = ?", (new_role, selected_user))
                conn.commit()
                st.success(f"Updated {selected_user}'s role to {new_role}")
                st.rerun()
            except sqlite3.Error as e:
                st.error(f"Error updating role: {e}")
            finally:
                conn.close()

    elif action == "Delete user":
        st.warning("This action cannot be undone!")
        if st.button("Confirm Delete", type="primary"):
            conn = get_db_connection()
            if not conn:
                return
            try:
                conn.execute("DELETE FROM measurements WHERE username = ?", (selected_user,))
                conn.execute("DELETE FROM feedback WHERE username = ?", (selected_user,))
                conn.execute("DELETE FROM users WHERE username = ?", (selected_user,))
                conn.commit()
                st.success(f"User {selected_user} deleted successfully")
                st.rerun()
            except sqlite3.Error as e:
                st.error(f"Error deleting user: {e}")
            finally:
                conn.close()

# ---------- Feedback Page ----------
def show_feedback():
    st.title("User Feedback")
    feedback = get_all_feedback()
    if not feedback:
        st.info("No feedback submitted yet")
        return

    tab1, tab2 = st.tabs(["View Feedback", "Respond to Feedback"])

    with tab1:
        for item in feedback:
            with st.expander(f"{item['username']} - {item['timestamp']}"):
                st.write(item['message'])
                # Show responses if any
                conn = get_db_connection()
                if conn:
                    try:
                        responses = conn.execute(
                            "SELECT admin_username, response, timestamp FROM feedback_responses WHERE feedback_id = ? ORDER BY timestamp",
                            (item['id'],)
                        ).fetchall()
                        if responses:
                            st.subheader("Responses")
                            for response in responses:
                                st.write(f"**{response['admin_username']}** ({response['timestamp']}):")
                                st.write(response['response'])
                                st.divider()
                    except sqlite3.Error as e:
                        st.error(f"Error loading responses: {e}")
                    finally:
                        conn.close()

    with tab2:
        selected_feedback = st.selectbox("Select feedback to respond to", 
                                       [f"{fb['username']} - {fb['timestamp']}" for fb in feedback])
        if selected_feedback:
            fb_index = [f"{fb['username']} - {fb['timestamp']}" for fb in feedback].index(selected_feedback)
            selected_fb = feedback[fb_index]
            
            st.write("**Original Message:**")
            st.write(selected_fb['message'])
            
            # Show existing responses
            conn = get_db_connection()
            if conn:
                try:
                    responses = conn.execute(
                        "SELECT admin_username, response, timestamp FROM feedback_responses WHERE feedback_id = ? ORDER BY timestamp",
                        (selected_fb['id'],)
                    ).fetchall()
                    
                    if responses:
                        st.subheader("Previous Responses")
                        for response in responses:
                            st.write(f"**{response['admin_username']}** ({response['timestamp']}):")
                            st.write(response['response'])
                            st.divider()
                    
                    # Response input
                    response = st.text_area("Your response", height=150, placeholder="Type your response here...")
                    if st.button("Submit Response"):
                        if not response.strip():
                            st.warning("Please enter a response")
                        else:
                            try:
                                conn.execute(
                                    "INSERT INTO feedback_responses (feedback_id, admin_username, response, timestamp) VALUES (?, ?, ?, ?)",
                                    (selected_fb['id'], st.session_state['username'], response, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                                )
                                conn.commit()
                                st.success("Response submitted successfully")
                                st.rerun()
                            except sqlite3.Error as e:
                                st.error(f"Error submitting response: {e}")
                finally:
                    conn.close()

# ---------- Dashboard ----------
def show_admin_dashboard():
    st.title("Admin Dashboard")
    conn = get_db_connection()
    if not conn:
        return
    try:
        stats = {
            "Total Users": conn.execute("SELECT COUNT(*) FROM users").fetchone()[0],
            "Total Feedback": conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0],
            "Admin Users": conn.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'").fetchone()[0],
            "Regular Users": conn.execute("SELECT COUNT(*) FROM users WHERE role = 'user'").fetchone()[0],
            "Recent Activity": conn.execute("SELECT COUNT(*) FROM measurements WHERE timestamp > datetime('now', '-7 days')").fetchone()[0]
        }
        cols = st.columns(3)
        for i, (name, value) in enumerate(stats.items()):
            cols[i % 3].metric(name, value)

        st.subheader("Recent Feedback")
        recent_feedback = conn.execute("""
            SELECT f.username, f.message, f.timestamp, 
                   (SELECT COUNT(*) FROM feedback_responses fr WHERE fr.feedback_id = f.id) as response_count
            FROM feedback f
            ORDER BY f.timestamp DESC LIMIT 3
        """).fetchall()
        
        if recent_feedback:
            for fb in recent_feedback:
                with st.expander(f"{fb['username']} - {fb['timestamp']} (Responses: {fb['response_count']})"):
                    st.write(fb['message'])
        else:
            st.info("No recent feedback")
    except sqlite3.Error as e:
        st.error(f"Error loading dashboard: {e}")
    finally:
        conn.close()

# ---------- Main Admin Panel ----------
def show_admin_panel():
    st.sidebar.title("Admin Menu")
    menu_options = {
        "📊 Dashboard": show_admin_dashboard,
        "🔳 QR Generator": show_qr_generator,
        "👥 User Management": show_user_management,
        "💬 Feedback": show_feedback
    }

    st.sidebar.divider()
    st.sidebar.markdown(f"**Logged in as:** {st.session_state['username']}")

    choice = st.sidebar.radio("Navigation", list(menu_options.keys()), label_visibility="collapsed")

    if st.sidebar.button("🚪 Logout", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    menu_options[choice]()

# ---------- Login Page ----------
def show_login():
    st.title("Admin Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            conn = get_db_connection()
            if not conn:
                return
            try:
                user = conn.execute(
                    "SELECT username, role FROM users WHERE username = ? AND password = ?",
                    (username, password)
                ).fetchone()
                
                if user:
                    st.session_state['username'] = user['username']
                    st.session_state['role'] = user['role']
                    st.rerun()
                else:
                    st.error("Invalid username or password")
            except sqlite3.Error as e:
                st.error(f"Login error: {e}")
            finally:
                conn.close()

# ---------- Main App ----------
def main():
    if 'username' not in st.session_state:
        show_login()
    elif st.session_state.get('role') == 'admin':
        show_admin_panel()
    else:
        st.error("You don't have admin privileges")
        if st.button("Logout"):
            st.session_state.clear()
            st.rerun()

if __name__ == "__main__":
    main()