import pymongo

# Replace the uri string with your MongoDB deployment's connection string.
conn_str = "mongodb://1.15.118.125:27017/"

# set a 5-second connection timeout
client = pymongo.MongoClient(conn_str, serverSelectionTimeoutMS=5000)

try:
    print(client.server_info())
except Exception:
    print("Unable to connect to the server.")

collection_user = client.get_database("NIC").get_collection("Mission")






