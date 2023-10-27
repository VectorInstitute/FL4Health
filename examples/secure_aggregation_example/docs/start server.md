# How the Server Starts

To kick off the server we use the `start_server()` method in `server/app.py`. This method 
calls `run_fl()` which calls `Server.fit(num_rounds)`. 

