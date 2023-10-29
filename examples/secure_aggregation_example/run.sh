echo "1"
nohup python -m examples.secure_aggregation_example.server > examples/secure_aggregation_example/server.out 
echo "2"
sleep 5
echo "3"
nohup python -m examples.secure_aggregation_example.client > examples/secure_aggregation_example/client.out
echo "4"

# python -m examples.secure_aggregation_example.client
