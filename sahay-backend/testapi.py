
# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask, request, jsonify
from flask_cors import CORS,cross_origin

# from openaitest import index
# import chatbot
import legalbot
# Flask constructor takes the name of
# current module (__name__) as argument.

app = Flask(__name__)

CORS(app, origins="http://localhost:3000")


# Endpoint to create a new guide
@app.route('/init', methods=['POST', 'GET'])
@cross_origin()
def initial():
    # print(session.sid)
    id = request.json['id']
    print(id)
    legalbot.init_mem(id)

    return jsonify("done")


@app.route('/message', methods=['POST', 'GET'])
@cross_origin()
def add_guide():
    # print(session.sid)
    title = request.json['input']
    id = request.json['id']
    
    # try:
    # print(session.get('recieving_details'))
    output = legalbot.chat_qa(title, id)
    # except:
    #     output = "Oops, There seems to be some problem. Please try again later."
    # if(done) :
    #     done, output = chatbot.chat_qa("")
    # print(output)
    return jsonify(output)
   

# def add_guide():
#     title = request.json['input']

#     return jsonify(index.query(title))


# main driver function
if __name__ == '__main__':

    # run() method of Flask class runs the application
    # on the local development server.
    app.run(port=8001, debug=True)
