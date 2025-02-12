# federated/server.py
import flwr as fl

def main():
    fl.server.start_server(server_address="[::]:8080", config={"num_rounds": 10})

if __name__ == "__main__":
    main()

# Run the server.py script in a terminal:
# $ python federated/server.py
# Now run the client.py script in another terminal:
# $ python federated/client.py --model dnn --batch_size 32 --device cpu
# $ python federated/client.py --model cnn --batch_size 32 --device cpu
# $ python federated/client.py --model rnn --batch_size 32 --device cpu
# Run the client.py script in another terminal:
# $ python federated/client.py --model dnn --batch_size 32 --device cuda
# $ python federated/client.py --model cnn --batch_size 32 --device cuda
# $ python federated/client.py --model rnn --batch_size 32 --device cuda
# The server will print the aggregated results after each round.
# The clients will print the training loss and test accuracy after each round.
# The server will print the aggregated results after each round.
# The clients will print the training loss and test accuracy after each round.  
