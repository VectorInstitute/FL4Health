# Local Differential Privacy 

### Local DP Client 

```py
# fl4health/clients/scaffold_client.py
class DPScaffoldLoggingClient
```

1. This client class inherits from `DPScaffoldClient` which is an `InstanceLevelPrivacyClient`. This client logs metrics to a JSON like the other CDP and DDP clients. 

### Local DP Server

```py
# fl4health/server/scaffold_server.py
class DPScaffoldLoggingServer
```

1 . Logs metrics to JSON.