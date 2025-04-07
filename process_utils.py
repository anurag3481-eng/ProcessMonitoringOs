import psutil
from datetime import datetime

def get_process_details(pid):
    """Get detailed information about a process"""
    try:
        process = psutil.Process(pid)
        
        # Get detailed information
        details = {
            "PID": pid,
            "Name": process.name(),
            "Status": process.status(),
            "Created": datetime.fromtimestamp(process.create_time()).strftime('%Y-%m-%d %H:%M:%S'),
            "CPU %": f"{process.cpu_percent():.1f}%",
            "Memory": f"{process.memory_info().rss / (1024 * 1024):.1f} MB",
            "Username": process.username(),
            "Executable": process.exe(),
            "Command Line": " ".join(process.cmdline()),
            "Threads": process.num_threads(),
            "Priority": process.nice()
        }
        
        return details
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None

def kill_process(pid):
    """Kill a process by PID"""
    try:
        process = psutil.Process(pid)
        process.terminate()
        return True, f"Process {pid} terminated."
    except psutil.NoSuchProcess:
        return False, f"Process {pid} not found."
    except psutil.AccessDenied:
        return False, f"Access denied to terminate process {pid}."

def change_process_priority(pid, priority):
    """Change the priority of a process"""
    try:
        process = psutil.Process(pid)
        process.nice(priority)
        return True, f"Priority of process {pid} changed."
    except psutil.NoSuchProcess:
        return False, f"Process {pid} not found."
    except psutil.AccessDenied:
        return False, f"Access denied to change priority of process {pid}." 
