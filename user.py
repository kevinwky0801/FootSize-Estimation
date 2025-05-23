
import sqlite3
import streamlit as st
# Database functions
def get_db_connection():
    return sqlite3.connect('foot_measurement.db')

def initialize_database():
    conn = get_db_connection()
    try:
        c = conn.cursor()

        # Create tables if they don't exist
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                name TEXT,
                email TEXT,
                password TEXT,
                role TEXT
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS measurements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                foot_length REAL,
                foot_width REAL,  -- Still exists but not used
                size_vn TEXT,
                size_uk TEXT,
                size_us TEXT,
                size_eu TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (username) REFERENCES users(username)
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                message TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (username) REFERENCES users(username)
            )
        """)

        # Create feedback_responses table if it doesn't exist
        c.execute("""
            CREATE TABLE IF NOT EXISTS feedback_responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feedback_id INTEGER,
                admin_username TEXT,
                response TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (feedback_id) REFERENCES feedback(id),
                FOREIGN KEY (admin_username) REFERENCES users(username)
            )
        """)

        conn.commit()
    finally:
        conn.close()

# Measurement functions (unchanged)
def add_measurement(username, foot_length, size_vn, size_uk, size_us, size_eu):
    conn = get_db_connection()
    try:
        c = conn.cursor()
        c.execute("""
            INSERT INTO measurements 
            (username, foot_length, size_vn, size_uk, size_us, size_eu) 
            VALUES (?, ?, ?, ?, ?, ?)
        """, (username, foot_length, size_vn, size_uk, size_us, size_eu))
        conn.commit()
    finally:
        conn.close()

def get_user_measurements(username):
    conn = get_db_connection()
    try:
        c = conn.cursor()
        c.execute("""
            SELECT foot_length, timestamp, size_vn, size_uk, size_us, size_eu
            FROM measurements 
            WHERE username = ? 
            ORDER BY timestamp DESC
        """, (username,))
        return c.fetchall()
    finally:
        conn.close()

# User functions (unchanged)
def get_user_info(username):
    conn = get_db_connection()
    try:
        c = conn.cursor()
        c.execute("SELECT username, name, email, role FROM users WHERE username = ?", (username,))
        return c.fetchone()
    finally:
        conn.close()

# Feedback functions - updated to include responses
def add_feedback(username, message):
    conn = get_db_connection()
    try:
        c = conn.cursor()
        c.execute("INSERT INTO feedback (username, message) VALUES (?, ?)", 
                 (username, message))
        conn.commit()
    finally:
        conn.close()

def get_user_feedback(username):
    conn = get_db_connection()
    try:
        c = conn.cursor()
        c.execute("""
            SELECT f.id, f.message, f.timestamp, 
                   (SELECT COUNT(*) FROM feedback_responses fr WHERE fr.feedback_id = f.id) as response_count
            FROM feedback f
            WHERE f.username = ?
            ORDER BY f.timestamp DESC
        """, (username,))
        return c.fetchall()
    finally:
        conn.close()

def get_feedback_responses(feedback_id):
    conn = get_db_connection()
    try:
        c = conn.cursor()
        c.execute("""
            SELECT admin_username, response, timestamp 
            FROM feedback_responses 
            WHERE feedback_id = ?
            ORDER BY timestamp
        """, (feedback_id,))
        return c.fetchall()
    finally:
        conn.close()

# Display functions - updated feedback section
def show_home(username):
    st.title("Foot Measurement System")
    st.write(f"Welcome, {username}!")
    
    st.header("Your Measurement History")
    measurements = get_user_measurements(username)
    
    if measurements:
        # Create a styled container for the table
        with st.container():
            st.markdown("""
               <style>
                .measurement-table {
                    width: 100%;
                    border-collapse: collapse;
                    background-color: #000000;  /* Black background */
                    color: #ffffff;  /* White text */
                }
                .measurement-table th {
                    background-color: #333333;  /* Dark gray header */
                    padding: 10px;
                    text-align: left;
                    border-bottom: 2px solid #555555;
                    color: #ffffff;
                }
                .measurement-table td {
                    padding: 10px;
                    border-bottom: 1px solid #555555;
                    color: #ffffff;
                }
                .measurement-table tr:hover {
                    background-color: #222222;  /* Slightly lighter black on hover */
                }
            </style>
            """, unsafe_allow_html=True)
            
            # Create table header
            st.markdown("""
                <table class="measurement-table">
                    <tr>
                        <th>#</th>
                        <th>Date & Time</th>
                        <th>Length (cm)</th>
                        <th>VN Size</th>
                        <th>UK Size</th>
                        <th>US Size</th>
                        <th>EU Size</th>
                    </tr>
            """, unsafe_allow_html=True)
            
            # Add measurement rows
            for i, (length, timestamp, size_vn, size_uk, size_us, size_eu) in enumerate(measurements, 1):
                st.markdown(f"""
                    <tr>
                        <td>{i}</td>
                        <td>{timestamp}</td>
                        <td>{length:.1f}</td>
                        <td>{size_vn}</td>
                        <td>{size_uk}</td>
                        <td>{size_us}</td>
                        <td>{size_eu}</td>
                    </tr>
                """, unsafe_allow_html=True)
            
            st.markdown("</table>", unsafe_allow_html=True)
    else:
        st.write("No measurements found.")

def show_profile(username):
    st.title("User Profile")
    user_info = get_user_info(username)
    if user_info:
        st.write(f"Username: {user_info[0]}")
        st.write(f"Name: {user_info[1]}")
        st.write(f"Email: {user_info[2]}")
        st.write(f"Role: {user_info[3]}")
    else:
        st.error("User not found")
def update_user_profile(username, name=None, email=None, password=None):
    conn = get_db_connection()
    try:
        c = conn.cursor()
        
        # Build the update query based on which fields are provided
        updates = []
        params = []
        
        if name is not None:
            updates.append("name = ?")
            params.append(name)
        if email is not None:
            updates.append("email = ?")
            params.append(email)
        if password is not None:
            updates.append("password = ?")
            params.append(password)
            
        if not updates:
            return False  # Nothing to update
            
        update_query = f"UPDATE users SET {', '.join(updates)} WHERE username = ?"
        params.append(username)
        
        c.execute(update_query, params)
        conn.commit()
        return c.rowcount > 0
    except sqlite3.Error as e:
        st.error(f"Error updating profile: {e}")
        return False
    finally:
        conn.close()

# Updated profile display function with editing capability
def show_profile(username):
    st.title("User Profile")
    user_info = get_user_info(username)
    
    if not user_info:
        st.error("User not found")
        return
    
    # Display current info
    st.write(f"Username: {user_info[0]}")
    st.write(f"Role: {user_info[3]}")  # Role shouldn't be editable by user
    
    # Edit form
    with st.expander("Edit Profile", expanded=False):
        with st.form("edit_profile_form"):
            new_name = st.text_input("Name", value=user_info[1])
            new_email = st.text_input("Email", value=user_info[2])
            new_password = st.text_input("New Password", type="password", placeholder="Leave blank to keep current")
            confirm_password = st.text_input("Confirm Password", type="password")
            
            submitted = st.form_submit_button("Save Changes")
            
            if submitted:
                # Validate inputs
                if new_password and new_password != confirm_password:
                    st.error("Passwords don't match!")
                    return
                
                # Prepare update data
                update_data = {}
                if new_name != user_info[1]:
                    update_data['name'] = new_name
                if new_email != user_info[2]:
                    update_data['email'] = new_email
                if new_password:
                    update_data['password'] = new_password
                
                if not update_data:
                    st.info("No changes made")
                    return
                
                # Update profile
                if update_user_profile(username, **update_data):
                    st.success("Profile updated successfully!")
                    st.rerun()  # Refresh to show updated info
                else:
                    st.error("Failed to update profile")
def show_user_feedback(username):
    st.title("Feedback")
    
    # Feedback submission form
    with st.expander("Submit New Feedback", expanded=True):
        with st.form("feedback_form"):
            message = st.text_area("Your feedback", height=150, placeholder="Type your message here...")
            submitted = st.form_submit_button("Submit Feedback")
            if submitted and message:
                add_feedback(username, message)
                st.success("Thank you for your feedback!")
                st.rerun()
    
    # Display user's feedback history with responses
    st.header("Your Feedback History")
    feedback_list = get_user_feedback(username)
    
    if not feedback_list:
        st.info("You haven't submitted any feedback yet")
        return
    
    for fb in feedback_list:
        with st.expander(f"Feedback from {fb[2]} (Responses: {fb[3]})"):
            st.write(fb[1])  # The feedback message
            
            # Show responses if any
            responses = get_feedback_responses(fb[0])
            if responses:
                st.subheader("Admin Responses")
                for response in responses:
                    st.markdown(f"""
                        <div style="background-color: #333333; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                            <strong>{response[0]}</strong> ({response[2]})
                            <p style="margin-top: 5px;">{response[1]}</p>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No responses yet")

def show_user_panel():
    st.sidebar.title("User Menu")
    menu_options = {
        "Home": show_home,
        "Profile": show_profile,
        "Feedback": show_user_feedback
    }
    
    # Adding a unique key to the radio button
    choice = st.sidebar.radio("Navigation", list(menu_options.keys()), key="unique_navigation_key")
    
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()
    
    menu_options[choice](st.session_state['username'])

# Initialize database
initialize_database()

# Main app logic
if 'username' not in st.session_state:
    st.title("Foot Measurement System")
    st.warning("Please login to access the system")
else:
    show_user_panel()