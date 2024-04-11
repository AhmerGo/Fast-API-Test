from pymongo import MongoClient
from bson.objectid import ObjectId

"""
Helper file for HossBot MongoDB database operations.

@author: Steven Barnas
@version: March 28, 2024 
"""

class Connection(object):

    def __init__(self):
        self.client = None
        self.database_name = None
        self.hostname = None
        self.password = None
        self.username = None
        self.chat_logs_collection = None


    """
    Connects to the MongoDB database using the provided credentials
    @param username: username of MongoDB user (String)
    @param password: password of MongoDB user (String)
    @param database_name: which MongoDB database to access (String)
    """
    def connect(self, username, password, database_name):
        self.username = username
        self.password = password
        self.hostname = "10.72.8.178:27017"
        self.database_name = database_name

        # MongoDB connection URI
        uri = f"mongodb://{self.username}:{self.password}@{self.hostname}/{self.database_name}"
        self.client = MongoClient(uri)

        # Create the chat_logs collection if it doesn't exist
        self.create_chat_logs_collection()
    """
    Reads and prints all the documents inside the provided directory
    @param db_choice: which MongoDB database to access (String)
    @param collection_choice: which MongoDB collection to access (String)
    """
    def read(self, db_choice, collection_choice):
        db = self.client[db_choice]
        collection = db[collection_choice]
        result = collection.find({})

        t = []
        for document in result:
            #print(document)
            t.append(document)
        
        return t

    """
    Inserts one user into the user table after checking for duplicates
    @param name: Name of user (String)
    @param email: Email of user (String)
    @return 1 (Int) if duplicate is found, inserts new document into collection if unique
    """
    def insert_users(self, name, email):
        db = self.client["chatbot"]
        collection = db["users"]
        if self.find_email(email) == 0:
            strip_email = email.strip()
            lower_email = strip_email.lower()
            collection.insert_one({'name': name, 'email': lower_email})
        else:
            return 1

    """
    Inserts one link into the links table after checking for duplicates. If duplicate is found, tally is updated
    @param link_url: URL of pre-made link (String)
    """
    def insert_links(self, link_url):
        db = self.client["chatbot"]
        collection = db["links"]
        if self.find_link(link_url) == 0:
            collection.insert_one({'link_url': link_url, 'link_tally': 1})
        else:
            self.update_link_tally(link_url)

    """
    Inserts a chat log session into the chatlog table.
    @param user_id: User ID of user that generated the chat log (Int)
    @param chat_log: Chat log that's being saved (String)
    @param response_flag_1: User reports as good answer (Int)
    @param response_flag_2: User reports as incorrect/low quality answer (Int)
    @param response_flag_3: User reports as inappropriate answer (Int)
    @param save_flag: Flags that chat was saved by user (Int)
    """
    def insert_chat_log(self, user_id, chat_log, response_flag_1, response_flag_2, response_flag_3, save_flag):
        db = self.client["chatbot"]
        collection = db["chatlog"]
        collection.insert_one(
            {'user_id': user_id, 'chat_log': chat_log,
             'flags': {'response_flag_1': response_flag_1, 'response_flag_2': response_flag_2,
                       'response_flag_3': response_flag_3, 'save_flag': save_flag}})

    """
    Inserts a premade question in to the premade_questions table after checking for duplicates.
    If duplicate exists, update tally instead.
    @param question: Pre-made question that was clicked/being inserted.
    """
    def insert_question(self, question):
        db = self.client["chatbot"]
        collection = db["premade_questions"]
        if self.find_question(question) == 0:
            collection.insert_one({'question': question, 'q_tally': 1})
        else:
            self.update_question_tally(question)

    """
    Checks if inserted email already exists
    @param input_email: Email that needs to be checked
    @return 1 (Int) if email already exists, 0 (Int) if email does not exist yet  
    """
    def find_email(self, input_email):
        strip_email = input_email.strip()
        lower_email = strip_email.lower()
        db = self.client["chatbot"]
        collection = db["users"]
        record = collection.find({"email": lower_email})
        return record[0]['email']
    
    """"""
    
    """
    Gets chat history of a given HSU.class instance 
    @param self the instance of HSU we're using to access said chat history
    @param user_id the id associated with a given chat log history 
    """
    def get_chat_history(self, user_id):
        try:
            chat_log = self.chat_logs_collection.find_one({"user_id": user_id})
            if chat_log:
                chat_history = chat_log["chat_history"]
                formatted_chat_history = [
                    (entry["user_input"], entry["bot_response"]) for entry in chat_history
                ]
                return formatted_chat_history
            else:
                return []
        except Exception as e:
            raise Exception(f"Error retrieving chat history: {str(e)}")
    """
    Appends chat history to the associated user_id
    @param self the instance of HSU.py 
    @param user_id the id associated with the given chat log history
    @param chat_history the chat history we're goign to append to that given chat log 
    """        
    def update_chat_history(self, user_id, chat_history):
        try:
            formatted_chat_history = [
                {"user_input": entry[0], "bot_response": entry[1]} for entry in chat_history
            ]
            self.chat_logs_collection.update_one(
                {"user_id": user_id},
                {"$set": {"chat_history": formatted_chat_history}},
                upsert=True
            )
        except Exception as e:
            raise Exception(f"Error updating chat history: {str(e)}")
        
    """
    Temporary method used to create the chat_log section
    @param self the instance of this connection 
    """
    def create_chat_logs_collection(self):
        db = self.client[self.database_name]
        if "chat_logs" in db.list_collection_names():
            db.drop_collection("chat_logs")
        self.chat_logs_collection = db.create_collection("chat_logs")


    """
    Checks if inserted link already exists
    @param link_url: URL that needs to be checked
    @return 1 (Int) if URL already exists, 0 (Int) if URL does not exist yet  
    """
    def find_link(self, link_url):
        db = self.client["chatbot"]
        collection = db["links"]
        result = collection.find({'link_url': link_url})
        return len(list(result))

    """
    Checks if inserted question already exists
    @param question: Question that needs to be checked
    @return 1 (Int) if question already exists, 0 (Int) if question does not exist yet  
    """
    def find_question(self, question):
        db = self.client["chatbot"]
        collection = db["premade_questions"]
        result = collection.find({'question': question})
        return len(list(result))

    """
    Gets current tally of provided URL
    @param link_url: URL whose tally needs to be checked
    @return tally of provided URL (Int)
    """
    def find_tally(self, link_url):
        db = self.client["chatbot"]
        collection = db["links"]
        record = collection.find({"link_url": link_url})
        return record[0]['link_tally']

    """
    Adds 1 to tally of provided URL
    @param link_url: URL whose tally needs to be updated
    """
    def update_link_tally(self, link_url):
        db = self.client["chatbot"]
        collection = db["links"]
        record = collection.find({"link_url": link_url})
        tally_total = record[0]['link_tally'] + 1
        collection.update_one({'link_url': link_url}, {'$set': {'link_tally': tally_total}})

    """
    Adds 1 to tally of provided question
    @param question: question whose tally needs to be updated
    """
    def update_question_tally(self, question):
        db = self.client["chatbot"]
        collection = db["premade_questions"]
        record = collection.find({"question": question})
        tally_total = record[0]['q_tally'] + 1
        collection.update_one({'question': question}, {'$set': {'q_tally': tally_total}})

    """
    Delete a user from the users table
    @param object_id: MongoDB Object ID of the user who needs to be deleted
    """
    def delete_users(self, object_id):
        db = self.client["chatbot"]
        collection = db["users"]
        collection.delete_one({'_id': ObjectId(object_id)})

    """
    Delete a chat log from the chatlog table
    @param object_id: MongoDB Object ID of the chat that needs to be deleted
    """
    def delete_chat_log(self, object_id):
        db = self.client["chatbot"]
        collection = db["chatlog"]
        collection.delete_one({'_id': ObjectId(object_id)})

    """
    Delete a link from the links table
    @param object_id: MongoDB Object ID of the pre-provided link that needs to be deleted
    """
    def delete_link(self, object_id):
        db = self.client["chatbot"]
        collection = db["links"]
        collection.delete_one({'_id': ObjectId(object_id)})

    """
    Delete a pre-made question from the premade_questions table
    @param object_id: MongoDB Object ID of the pre-made question that needs to be deleted
    """
    def delete_question(self, object_id):
        db = self.client["chatbot"]
        collection = db["premade_questions"]
        collection.delete_one({'_id': ObjectId(object_id)})

    """
    Closes connection to MongoDB
    """
    def close(self):
        # Close the connection
        self.client.close()

test = Connection()
Connection.connect(test, "admin", "Stevencantremember", "admin")
# Connection.insert_users(test, 2, "Steven Barnas", "steven@hsutx.edu")
# Connection.insert_links(test, "www.hsutx.edu")
# Connection.update_tally(test, "www.hsutx.edu")
Connection.read(test,"chatbot", "users")
# Connection.read(test,"chatbot", "links")
# Connection.delete_chat_log(test, '65d7863a3a91bedf9928cf8f')
# Connection.read(test,"chatbot", "chatlog")
# print(Connection.find_tally(test, "www.hsutx.edu"))
# Connection.insert_chat_log(test, 1, "Lorem Ipsum 2", 0, 0, 0, 1)
# print(Connection.find_link(test, "www.hsutx.edu"))
# print(Connection.test_object(test, "Steven Barnas"))
# Connection.delete_users(test, 2, "Steven Barnas", "steven@hsutx.edu")
# Connection.read(test,"chatbot", "users")
# print(jsonify(Connection.read(test,"chatbot", "users")))
# Connection.close(test)

# upd = users.update_one({'user_id': 1}, {'$set':{'name': 'Jaden Barnwell', 'email': 'jabster@hsutx.edu'}})
