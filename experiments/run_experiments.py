# experiments/run_experiment.py
import subprocess
import time

def run_server():
    return subprocess.Popen(["python", "federated/server.py"])

def run_client(model_type="dnn"):
    return subprocess.Popen(["python", "federated/client.py", "--model", model_type])

if __name__ == "__main__":
    # Start the Flower server
    server_proc = run_server()
    time.sleep(5)  # Wait for the server to start
    
    # Launch multiple clients (change "dnn" to "cnn" or "rnn" as desired)
    clients = [run_client("cnn") for _ in range(3)]
    
    # Wait for clients to finish
    for client in clients:
        client.wait()
    
    server_proc.terminate()
