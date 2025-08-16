# app.py

from flask import Flask, flash, render_template, Response, redirect, url_for, request, jsonify, send_file, session
import face_recognition
import cv2
import numpy as np
from PIL import Image
import csv
import datetime
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Get the absolute path of the current script's directory (where app.py is located)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the directories to save the CSV files and photos inside the project directory
attendance_dir = os.path.join(current_dir, 'attendance_records')
photos_dir = os.path.join(current_dir, 'photos')

# Check if the directories exist, if not, create them
if not os.path.exists(attendance_dir):
    os.makedirs(attendance_dir)
if not os.path.exists(photos_dir):
    os.makedirs(photos_dir)

# Global variables for attendance tracking
attendance_tracking = {
    "Entry": set(),  # Set to track Entrys
    "Exit": set(),  # Set to track Exits
    "messages": set()  # Set to store displayed messages in this session
}

def load_users():
    """Load users and their face encodings."""
    known_face_names = []
    known_face_encodings = []

    for file_name in os.listdir(photos_dir):
        if file_name.endswith('.jpg'):
            name = os.path.splitext(file_name)[0]
            file_path = os.path.join(photos_dir, file_name)
            image = face_recognition.load_image_file(file_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_names.append(name)
                known_face_encodings.append(encodings[0])

    return known_face_names, known_face_encodings
                
# Get current date for the CSV file
current_date = datetime.datetime.now().strftime("%Y-%m-%d")

def load_attendance_records_for_date(date):
    """Load existing attendance records for a specific date from the CSV file."""
    attendance_file_path = os.path.join(attendance_dir, f"{date}.csv")
    if os.path.exists(attendance_file_path):
        with open(attendance_file_path, 'r') as f:
            csv_reader = csv.reader(f)
            try:
                next(csv_reader)  # Skip the header row
            except StopIteration:
                return []  # If the file is empty, return an empty list
            return list(csv_reader)  # Return the rest of the rows without the header
    return []

def save_attendance_record(name, Entry_time, Exit_time=""):
    """Save attendance record in the CSV file for the current date."""
    attendance_csv_file_path = os.path.join(attendance_dir, f"{current_date}.csv")
    
    # Load attendance records for the current day (if any)
    attendance_data = load_attendance_records_for_date(current_date)
    
    # Prevent marking attendance multiple times for the same person on the same day
    for record in attendance_data:
        if record[0] == name:
            return  # Prevent re-marking
    
    # Create or append to the file
    file_exists = os.path.exists(attendance_csv_file_path)
    with open(attendance_csv_file_path, 'a', newline='') as attendance_file:
        writer = csv.writer(attendance_file)
        if not file_exists:  # If the file doesn't exist, write the header
            writer.writerow(["Name", "Entry Time", "Exit Time"])
        writer.writerow([name, Entry_time, Exit_time])  # Save attendance record

def save_Exit_time(name, Exit_time):
    """Update the Exit time for a given name in the CSV file."""
    attendance_csv_file_path = os.path.join(attendance_dir, f"{current_date}.csv")
    
    # Load the existing attendance data
    attendance_data = load_attendance_records_for_date(current_date)
    
    # Find the record for the user and update the Exit time
    updated = False
    for record in attendance_data:
        if record[0] == name and record[2] == "":  # Exit time is empty
            record[2] = Exit_time
            updated = True
            break
    
    # If updated, write the new data back to the CSV file
    if updated:
        with open(attendance_csv_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Entry Time", "Exit Time"])  # Header
            writer.writerows(attendance_data)  # Save the updated rows

@app.route('/mark/<type_of_attendance>', methods=['GET', 'POST'])
def mark_attendance(type_of_attendance):
    displayed_messages = set()  # Set to store displayed messages for this session

    if request.method == 'POST':
        # Clear attendance tracking for the selected type of attendance
        attendance_tracking[type_of_attendance].clear()
        displayed_messages.clear()
        return redirect(url_for('index'))
    
    return render_template('attendance.html', type_of_attendance=type_of_attendance,)

# Global list to store attendance messages
attendance_messages = []

@app.route('/video_feed/<type_of_attendance>')
def video_feed(type_of_attendance):
    def generate_frames():
        video_capture = cv2.VideoCapture(0)
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, video_capture.get(cv2.CAP_PROP_FRAME_WIDTH) // 2)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT) // 2)
        known_face_names, known_face_encodings = load_users()
        
        attendance_messages.clear()
        displayed_messages = set()  # Set to track displayed messages for each session

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.4)  
                name = "Unknown"

                if True in matches:
                    best_match_index = np.argmin(face_recognition.face_distance(known_face_encodings, face_encoding))
                    name = known_face_names[best_match_index]

                if name != "Unknown" and name not in displayed_messages:
                    # Load the current day's attendance records
                    attendance_data = load_attendance_records_for_date(current_date)

                    user_Entry = False
                    user_Exit = False
                    
                    for record in attendance_data:
                        if record[0] == name:  # If the user exists in today's attendance records
                            if record[1] !="":
                                user_Entry=True # If the user has marked Entry
                                if type_of_attendance == "Entry":
                                    print(f"{name} already marked Entry.")
                                    attendance_messages.append(f"{name} already marked Entry.")
                            if record[2] !="":  # If the user has marked  Exit
                                user_Exit = True
                                if type_of_attendance == "Exit":
                                    print(f"{name} already marked Exit.")
                                    attendance_messages.append(f"{name} already marked Exit.")

                    if not user_Entry:
                        # If the user hasn't marked Entry today
                        if type_of_attendance == "Entry":
                            attendance_tracking["Entry"].add(name)
                            save_attendance_record(name, datetime.datetime.now().strftime("%H:%M:%S"))
                            print(f"{name} marked Entry.")
                            attendance_messages.append(f"{name} marked Entry.")
                        elif type_of_attendance == "Exit":
                            print(f"{name}, please mark your Entry first before Exit.")
                            attendance_messages.append(f"{name} please mark your Entry first before Exit.")
                    
                    elif user_Entry and not user_Exit:  # User has Entry marked but not Exit
                        # Allow the user to mark Exit
                        if type_of_attendance == "Exit":
                            attendance_tracking["Exit"].add(name)
                            save_Exit_time(name, datetime.datetime.now().strftime("%H:%M:%S"))
                            print(f"{name} marked Exit.")
                            attendance_messages.append(f"{name} marked Exit.")
                                            
                    # Add user to displayed messages to prevent re-displaying in the same session
                    displayed_messages.add(name)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        video_capture.release()

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Remaining routes and functions unchanged...

# Route to fetch messages from the global list
@app.route('/get_messages', methods=['GET'])
def get_messages():
    return jsonify(attendance_messages)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/admin', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Validate credentials
        if username == 'admin' and password == 'admin123':
            return redirect(url_for('admin_panel'))
        else:
            return render_template('admin_login.html', error="Incorrect username or password")
    
    return render_template('admin_login.html')

@app.route('/admin_panel', methods=['GET', 'POST'])
def admin_panel():
    if request.method == 'POST':
        action = request.form.get('action')

        # Add new user
        if action == 'add_user':
            name = request.form.get('name')
            photo = request.files.get('photo')
            
            # Check if the username already exists
            photo_path = os.path.join(photos_dir, f"{name}.jpg")
            if os.path.exists(photo_path):
                flash("Username already exists. Please choose a different name.", 'add_user_error')
                return redirect(url_for('admin_panel'))
            
            if photo and (photo.filename.endswith('.jpg') or photo.filename.endswith('.jpeg')):
                photo.save(photo_path)

                try:
                    # Open the image
                    image = Image.open(photo_path)

                    # Resize the image if it's too large, but maintain aspect ratio
                    max_size = 800  # Max size for resizing
                    width, height = image.size
                    if width > max_size or height > max_size:
                        aspect_ratio = width / height
                        if width > height:
                            new_width = max_size
                            new_height = int(new_width / aspect_ratio)
                        else:
                            new_height = max_size
                            new_width = int(new_height * aspect_ratio)
                        image = image.resize((new_width, new_height))  # Resize while preserving aspect ratio
                        image.save(photo_path)  # Save the resized image back

                    # Convert image to grayscale (optional, can help in some cases)
                    image = image.convert("L")  # "L" mode is for grayscale
                    image.save(photo_path)  # Save back grayscale image

                    # Attempt to create face encoding for the new user
                    new_image = face_recognition.load_image_file(photo_path)
                    new_encodings = face_recognition.face_encodings(new_image)

                    # Debugging: print the number of faces detected
                    print(f"Detected {len(new_encodings)} faces.")

                    if not new_encodings:  # If no face found
                        if os.path.exists(photo_path):
                            os.remove(photo_path)  # Remove the photo
                        flash("No face detected in the uploaded photo. Please try again with a valid photo.", 'add_user_error')
                    elif len(new_encodings) > 1:  # If multiple faces found
                        if os.path.exists(photo_path):
                            os.remove(photo_path)  # Remove the photo
                        flash("Multiple faces detected in the uploaded photo. Please upload a photo with only one face.", 'add_user_error')
                    else:
                        flash("User added successfully!", 'add_user_success')  # Flash success message

                except Exception as e:
                    # Handle any other exceptions (e.g., invalid image format)
                    if os.path.exists(photo_path):
                        os.remove(photo_path)  # Remove the photo
                    flash(f"An error occurred: {str(e)}", 'add_user_error')
            else:
                flash("Invalid file format. Please upload a JPG or JPEG photo.", 'add_user_error')
            
            return redirect(url_for('admin_panel'))

        # Delete user
        elif action == 'delete_user':
            name = request.form.get('delete_name')
            photo_path = os.path.join(photos_dir, f"{name}.jpg")
            if os.path.exists(photo_path):
                os.remove(photo_path)
                flash("User deleted successfully!", 'delete_user_success')
            else:
                flash("User not found", 'delete_user_error')

    return render_template('admin_panel.html')



@app.route('/generate_attendance_data', methods=['GET', 'POST'])
def generate_attendance_data():
    dates = list_all_dates()  # Get available dates
    all_users = user_names()  # Fetch all users

    if request.method == 'POST':
        selected_date = request.form.get('date')

        # Load the attendance data for the selected date
        attendance_data = load_attendance_records_for_date(selected_date)

        if not attendance_data:
            return render_template('attendance_data.html', dates=dates, message="No records found for the selected date.", selected_date=selected_date)

        present_members = []
        absent_members = []

        # Loop through all users
        for user in all_users:
            user_present = False

            # Check each attendance record for the current user
            for record in attendance_data:
                name = record[0]
                Entry_time = record[1]
                Exit_time = record[2]

                if name == user:
                    # If both Entry and Exit times are recorded, mark as present
                    if Entry_time and Exit_time:
                        present_members.append(name)
                        user_present = True
                    # If only Entry time is recorded, mark as absent
                    elif Entry_time and not Entry_time:
                        absent_members.append(name)
                        user_present = True
                    break  # No need to check further records for this user
            
            # If the user is not in the attendance record (no matching name), mark as absent
            if not user_present:
                absent_members.append(user)

        # Generate CSV content with two columns: Present members and Absent members
        csv_content = "Present members, Absent members\n"
        max_length = max(len(present_members), len(absent_members))

        for i in range(max_length):
            present = present_members[i] if i < len(present_members) else ""
            absent = absent_members[i] if i < len(absent_members) else ""
            csv_content += f"{present},{absent}\n"

        # Store CSV content and filename in session
        session['csv_content'] = csv_content
        session['filename'] = f"attendance_data_{selected_date}.csv"

        # After selecting a date, show the "View Data" and enable the download button
        return render_template('attendance_data.html', 
                               dates=dates, 
                               selected_date=selected_date, 
                               present_members=present_members, 
                               absent_members=absent_members, 
                               download_link=True)

    return render_template('attendance_data.html', dates=dates)

@app.route('/download/<filename>')
def download(filename):
    # Retrieve CSV content from session
    csv_content = session.get('csv_content')

    if not csv_content:
        return "No data available to download", 400

    # Save the CSV content to a temporary file for download
    temp_folder = os.path.join(current_dir, 'attendance_data')
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    file_path = os.path.join(temp_folder, filename)

    # Save the CSV content to a file
    with open(file_path, 'w') as f:
        f.write(csv_content)

    # Return the file for download
    return send_file(file_path, as_attachment=True, download_name=filename)

@app.route('/all_users')
def all_users():
    users=user_names()
    return render_template('all_users.html', users=users)

def user_names():
    # Get the list of all known user names
    users=[]
    for file_name in os.listdir(photos_dir):
        if file_name.endswith('.jpg'):
            nam = os.path.splitext(file_name)[0]
            if nam:
                users.append(nam)
    return users


@app.route('/records', methods=['GET', 'POST'])
def records():
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    dates = list_all_dates()

    if request.method == 'POST':
        selected_date = request.form.get('date')

        attendance_data = load_attendance_records_for_date(selected_date)
        if not attendance_data:
            return render_template('records.html', dates=dates, selected_date=selected_date, message="No records found for selected date.", current_date=current_date)
        return render_template('records.html', records=attendance_data, dates=dates, selected_date=selected_date, current_date=current_date)

    return render_template('records.html', dates=dates, current_date=current_date)

def list_all_dates():
    dates = []
    for filename in os.listdir(attendance_dir):
        if filename.endswith(".csv"):
            date = filename.split('.')[0]
            dates.append(date)
    return sorted(dates, reverse=True)

if __name__ == '__main__':
    app.run(debug=True)